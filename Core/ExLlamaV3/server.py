import argparse
import os
import sys
import json
import io
import time
import uvicorn
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path

# [Tuning] OOM 방지를 위한 PyTorch 메모리 설정
os.environ[
    "PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"

try:
    from exllamav3 import model_init, Generator, Model
    from exllamav3.generator.sampler import ComboSampler
    from exllamav3.model import Config
    from chat_templates import prompt_formats
except ImportError as e:
    print("❌ ExLlamaV3 라이브러리를 찾을 수 없습니다. 경로를 확인해주세요.")
    print(e)
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("❌ PIL(Pillow) 라이브러리가 설치되지 않았습니다. 'pip install pillow'를 실행하세요.")
    sys.exit(1)

# =====================================================================
# 1. 설정 파싱 (Arguments)
# =====================================================================
parser = argparse.ArgumentParser(description="ExLlamaV3 Binary Image Server")
model_init.add_args(parser, cache=True)

parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
parser.add_argument("--port", type=int, default=8000, help="Port number")
parser.add_argument("-mode", "--mode", type=str, default="gemma",
                    help="Chat template mode (e.g., gemma, llama3, chatml)")
parser.add_argument("-modes", "--modes", action="store_true", help="List available prompt modes and exit")
parser.add_argument("-gpu", "--gpu_index", type=str, help="Comma-separated list of GPU indices (e.g., '0,1')")

# 샘플링 기본값 설정
parser.add_argument("-temp", "--temperature", type=float, help="Default temperature", default=0.8)
parser.add_argument("-topk", "--top_k", type=int, help="Default Top-K", default=50)
parser.add_argument("-topp", "--top_p", type=float, help="Default Top-P", default=0.8)
parser.add_argument("-minp", "--min_p", type=float, help="Default Min-P", default=0.0)
parser.add_argument("-repp", "--repetition_penalty", type=float, help="Default repetition penalty", default=1.05)
parser.add_argument("-presp", "--presence_penalty", type=float, help="Default presence penalty", default=0.0)
parser.add_argument("-freqp", "--frequency_penalty", type=float, help="Default frequency penalty", default=0.0)

# 전역 변수 선언 (초기화는 main 블록에서 수행)
args = None
model = None
config = None
cache = None
tokenizer = None
vision_model = None
generator = None
prompt_format = None


# =====================================================================
# 2. 데이터 모델 (Pydantic)
# =====================================================================
class Message(BaseModel):
    role: str
    content: str


class ChatRequestConfig(BaseModel):
    messages: List[Message]
    max_tokens: int = Field(default=1000, alias="max_response_tokens")

    thinking_budget: Optional[int] = None
    no_think: bool = False
    tools: Optional[List[Dict[str, Any]]] = None

    # 샘플링 옵션 (기본값은 나중에 args가 로드된 후 설정됨)
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


# =====================================================================
# 3. FastAPI 서버 설정
# =====================================================================
app = FastAPI(title="ExLlamaV3 Binary Vision Server")


@app.post("/v1/chat/completions")
async def chat_endpoint_binary(
        images: List[UploadFile] = File(None),
        request_json: str = Form(...)
):
    # 전역 변수 참조
    global generator, tokenizer, vision_model, prompt_format, args

    if generator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        t_start = time.time()

        # 1. JSON 설정 파싱
        try:
            req_dict = json.loads(request_json)
            req = ChatRequestConfig(**req_dict)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format in 'request_json'")

        # 2. 이미지 처리 (Vision Logic)
        image_embeddings = []
        placeholders = ""

        if images:
            if vision_model:
                print(f"🖼️ 이미지 {len(images)}장 수신됨")

                for img_file in images:
                    file_bytes = await img_file.read()
                    pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    emb = vision_model.get_image_embeddings(tokenizer=tokenizer, image=pil_image)
                    image_embeddings.append(emb)

                placeholders = "".join([ie.text_alias for ie in image_embeddings])
                print(f"✅ 이미지 임베딩 완료. Placeholders: {placeholders}")
            else:
                print("⚠️ 이미지가 업로드되었으나, 서버에 Vision 모델이 로드되지 않았습니다.")

        # 3. 프롬프트 구성
        spc = {}
        if req.thinking_budget is not None:
            spc["thinking_budget"] = req.thinking_budget
        prompt_format.set_special(spc)

        banned_strings = []
        if req.no_think:
            tt = prompt_format.thinktag()
            if tt and tt[0]: banned_strings.append(tt[0])
            if tt and tt[1]: banned_strings.append(tt[1])

        # 🟢 클라이언트에서 no_think=True를 보내면 꺼지도록 수정 (권장)
        use_think = not req.no_think
        # use_think = False #수동 설정
        print(f"🧠 [Think 모드 상태]: {'켜짐 (ON) - 추론/사고 중...' if use_think else '꺼짐 (OFF) - 초고속 대답 모드!'}")

        system_prompt = prompt_format.default_system_prompt(think=use_think)
        chat_history = []
        current_user_msg = None

        for msg in req.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                if current_user_msg is not None:
                    chat_history.append((current_user_msg, None))
                current_user_msg = msg.content
            elif msg.role == "assistant":
                if current_user_msg is None:
                    continue
                chat_history.append((current_user_msg, msg.content))
                current_user_msg = None

        if current_user_msg is not None:
            chat_history.append((current_user_msg, None))

        if placeholders and chat_history:
            last_user_msg, last_asst_msg = chat_history[-1]
            new_msg = f"{placeholders}\n{last_user_msg}"
            chat_history[-1] = (new_msg, last_asst_msg)

        # 🟢 [수정됨] chat_templates.py의 인자 충돌을 피하기 위해 tools 인자 전달 제거
        full_prompt = prompt_format.format(
            system_prompt=system_prompt,
            messages=chat_history,
            think=use_think
        )

        prompt_ids = tokenizer.encode(full_prompt)
        prompt_tokens = prompt_ids.shape[-1]

        print(f"📝 Prompt Tokens: {prompt_tokens}")

        # 4. 추론 실행
        # 샘플링 파라미터: 요청값이 없으면 args 기본값 사용
        sampler = ComboSampler(
            temperature=req.temperature if req.temperature is not None else args.temperature,
            top_k=req.top_k if req.top_k is not None else args.top_k,
            top_p=req.top_p if req.top_p is not None else args.top_p,
            min_p=req.min_p if req.min_p is not None else args.min_p,
            rep_p=req.repetition_penalty if req.repetition_penalty is not None else args.repetition_penalty,
            pres_p=req.presence_penalty if req.presence_penalty is not None else args.presence_penalty,
            freq_p=req.frequency_penalty if req.frequency_penalty is not None else args.frequency_penalty,
        )

        stop_conditions = [tokenizer.eos_token_id]
        stop_conditions.extend(prompt_format.stop_conditions(tokenizer))
        stop_conditions = [sc for sc in stop_conditions if sc is not None]

        t_gen_start = time.time()

        output = generator.generate(
            prompt=full_prompt,
            max_new_tokens=req.max_tokens,
            sampler=sampler,
            stop_conditions=stop_conditions,
            embeddings=image_embeddings if image_embeddings else None,
            add_bos=prompt_format.add_bos(),
            encode_special_tokens=True,
            decode_special_tokens=True,
            banned_strings=banned_strings
        )

        t_end = time.time()

        # 5. 결과 처리
        answer = output[len(full_prompt):]
        for sc in stop_conditions:
            if isinstance(sc, str) and answer.endswith(sc):
                answer = answer[:-len(sc)]
        answer = answer.replace("<eos>", "").strip()

        output_ids = tokenizer.encode(output)
        total_tokens = output_ids.shape[-1]
        new_tokens = total_tokens - prompt_tokens
        time_gen = t_end - t_gen_start
        tps = new_tokens / time_gen if time_gen > 0 else 0

        print(f"🤖 생성 완료: {new_tokens} tokens ({tps:.2f} t/s)")

        return {
            "choices": [{"message": {"role": "assistant", "content": answer}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": new_tokens,
                "total_tokens": total_tokens,
                "tps": round(tps, 2)
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# 메인 실행 블록 (모델 로딩 및 서버 시작)
# =====================================================================
if __name__ == "__main__":
    # 병렬 연산(TP) 사용 시 Linux/WSL 환경에서 'spawn' 모드 강제 설정 필요
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 1. 인자 파싱
    args = parser.parse_args()

    if args.modes:
        print("Available modes:")
        for k, v in prompt_formats.items():
            print(f" - {k:16} {v.description}")
        sys.exit(0)

    # -----------------------------------------------------------------
    # [Manual Split Logic] 사용자가 지정한 인덱스와 스플릿 값 필터링 및 적용
    # -----------------------------------------------------------------
    if args.gpu_index:
        # 1. 시스템 레벨에서 사용할 GPU만 노출시킴 (예: "1,2")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
        print(f"🎯 활성화된 GPU 인덱스: {args.gpu_index}")

        # 2. Split 배열 필터링 로직
        if hasattr(args, "gpu_split") and args.gpu_split:
            try:
                # 콤마로 구분된 문자열을 리스트로 변환 (공백 제거)
                target_indices = [int(idx.strip()) for idx in args.gpu_index.split(",")]
                all_splits = [val.strip() for val in args.gpu_split.split(",")]

                filtered_splits = []
                for idx in target_indices:
                    # 전체 스플릿 배열에서 내가 사용할 인덱스 위치의 값만 쏙 뽑아옴
                    if idx < len(all_splits):
                        filtered_splits.append(all_splits[idx])
                    else:
                        # 혹시 배열 길이가 안 맞을 경우를 대비한 방어 코드
                        filtered_splits.append("24.0")

                        # ExLlamaV3가 인식할 수 있도록 다시 콤마 문자열로 병합 (예: "24.0,24.0")
                args.gpu_split = ",".join(filtered_splits)
                print(f"✅ 최종 적용된 gpu_split (필터링 완료): {args.gpu_split}")

            except Exception as e:
                print(f"❌ gpu_split 필터링 중 오류 발생: {e}")

    # -----------------------------------------------------------------

    # 2. 모델 로딩 (메인 프로세스에서만 실행)
    print(f"⏳ 모델 로딩 시작... {args.model_dir}")

    try:
        model, config, cache, tokenizer = model_init.init(args)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        sys.exit(1)

    if args.mode not in prompt_formats:
        print(f"⚠️ 경고: 알 수 없는 모드 '{args.mode}'. 'gemma' 모드로 대체합니다.")
        args.mode = "gemma"
    prompt_format = prompt_formats[args.mode]("User", "Assistant")
    print(f"✅ Prompt Template: {args.mode}")

    print("👀 Vision Encoder 로딩 시도...")
    try:
        vision_model = Model.from_config(config, component="vision")
        vision_model.load()
        print("✅ Vision Encoder 로드 성공! (이미지 처리 가능)")
    except Exception as e:
        vision_model = None
        print(f"ℹ️ Vision Encoder 로드 실패 (텍스트 전용 모델일 수 있음): {e}")

    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    # 3. 서버 시작
    print(f"🚀 서버 시작: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)