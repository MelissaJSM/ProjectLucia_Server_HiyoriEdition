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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import server_config  # 이제 정상적으로 import 됩니다!

# [공사 1] Hugging Face AutoTokenizer 추가 (chat_templates 제거)
from transformers import AutoTokenizer

# [Tuning] OOM 방지를 위한 PyTorch 메모리 설정
os.environ[
    "PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"

try:
    from exllamav3 import model_init, Generator, Model
    from exllamav3.generator.sampler import ComboSampler
    # from chat_templates import prompt_formats <-- 삭제 완료!
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
# -mode 인자는 더 이상 쓰이지 않지만 호환성을 위해 남겨둠
parser.add_argument("-mode", "--mode", type=str, default="auto", help="Ignored (handled by AutoTokenizer)")
parser.add_argument("-gpu", "--gpu_index", type=str, help="Comma-separated list of GPU indices (e.g., '0,1')")

# 샘플링 기본값 설정
parser.add_argument("-temp", "--temperature", type=float, help="Default temperature", default=0.8)
parser.add_argument("-topk", "--top_k", type=int, help="Default Top-K", default=50)
parser.add_argument("-topp", "--top_p", type=float, help="Default Top-P", default=0.8)
parser.add_argument("-minp", "--min_p", type=float, help="Default Min-P", default=0.0)
parser.add_argument("-repp", "--repetition_penalty", type=float, help="Default repetition penalty", default=1.05)
parser.add_argument("-presp", "--presence_penalty", type=float, help="Default presence penalty", default=0.0)
parser.add_argument("-freqp", "--frequency_penalty", type=float, help="Default frequency penalty", default=0.0)

# 전역 변수 선언
args = None
model = None
config = None
cache = None
tokenizer = None  # ExLlama 내부 토크나이저 (토큰 ID 변환용)
hf_tokenizer = None  # [공사 2] HF 토크나이저 추가 (프롬프트 템플릿용)
vision_model = None
generator = None


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

    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None


# =====================================================================
# 3. FastAPI 서버 설정
# =====================================================================
app = FastAPI(title="ExLlamaV3 Binary Vision Server")


@app.post("/v1/chat/completions")
async def chat_endpoint_binary(
        images: List[UploadFile] = File(None),
        request_json: str = Form(...)
):
    global generator, tokenizer, hf_tokenizer, vision_model, args

    if generator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        t_start = time.time()

        # 1. JSON 파싱
        try:
            req_dict = json.loads(request_json)
            req = ChatRequestConfig(**req_dict)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format in 'request_json'")

        # 2. 이미지 처리
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

        # 3. [공사 3] HF AutoTokenizer를 활용한 완벽한 프롬프트 구성
        # 클라이언트가 보낸 메시지를 그대로 딕셔너리 리스트로 변환
        messages_for_template = [{"role": msg.role, "content": msg.content} for msg in req.messages]

        # 이미지 Placeholder가 있다면 마지막 user 메시지 맨 앞에 삽입
        if placeholders:
            for i in range(len(messages_for_template) - 1, -1, -1):
                if messages_for_template[i]["role"] == "user":
                    messages_for_template[i]["content"] = f"{placeholders}\n{messages_for_template[i]['content']}"
                    break

        # HF 토크나이저로 템플릿 적용 (Jinja 템플릿 기반으로 완벽한 형태 생성)
        full_prompt = hf_tokenizer.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_ids = tokenizer.encode(full_prompt)
        prompt_tokens = prompt_ids.shape[-1]
        print(f"📝 Prompt Tokens: {prompt_tokens}")

        # 4. 추론 파라미터 설정
        sampler = ComboSampler(
            temperature=req.temperature if req.temperature is not None else args.temperature,
            top_k=req.top_k if req.top_k is not None else args.top_k,
            top_p=req.top_p if req.top_p is not None else args.top_p,
            min_p=req.min_p if req.min_p is not None else args.min_p,
            rep_p=req.repetition_penalty if req.repetition_penalty is not None else args.repetition_penalty,
            pres_p=req.presence_penalty if req.presence_penalty is not None else args.presence_penalty,
            freq_p=req.frequency_penalty if req.frequency_penalty is not None else args.frequency_penalty,
        )

        # [공사 4] 정지 조건 간소화 (아래처럼 덮어씌워 주세요)
        stop_conditions = [tokenizer.eos_token_id]
        if hasattr(hf_tokenizer, "eos_token") and hf_tokenizer.eos_token:
            stop_conditions.append(hf_tokenizer.eos_token)

        # 자주 쓰이는 찌꺼기 방지용 하드코딩 정지 토큰 추가
        for stop_str in ["<|eot_id|>", "<|im_end|>", "<end_of_turn>", "</s>", "user\n", "user:", "User:"]:
            stop_conditions.append(stop_str)

        # 🟢 [여기에 추가!] llm_handler.py가 보내준 stop 배열을 최종적으로 병합
        if req.stop:
            stop_conditions.extend(req.stop)

        t_gen_start = time.time()

        output = generator.generate(
            prompt=full_prompt,
            max_new_tokens=req.max_tokens,
            sampler=sampler,
            stop_conditions=stop_conditions,
            embeddings=image_embeddings if image_embeddings else None,
            add_bos=False,  # HF 템플릿이 이미 BOS를 처리함
            encode_special_tokens=True,
            decode_special_tokens=True
        )

        t_end = time.time()

        # 5. 결과 처리
        answer = output[len(full_prompt):]
        for sc in stop_conditions:
            if isinstance(sc, str) and answer.endswith(sc):
                answer = answer[:-len(sc)]
        answer = answer.strip()

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
# 메인 실행 블록
# =====================================================================
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    args = parser.parse_args()

    if args.gpu_index:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
        print(f"🎯 활성화된 GPU 인덱스: {args.gpu_index}")

        if hasattr(args, "gpu_split") and args.gpu_split:
            try:
                target_indices = [int(idx.strip()) for idx in args.gpu_index.split(",")]
                all_splits = [val.strip() for val in args.gpu_split.split(",")]

                filtered_splits = []
                for idx in target_indices:
                    if idx < len(all_splits):
                        filtered_splits.append(all_splits[idx])
                    else:
                        filtered_splits.append("24.0")

                args.gpu_split = ",".join(filtered_splits)
                print(f"✅ 최종 적용된 gpu_split: {args.gpu_split}")

            except Exception as e:
                print(f"❌ gpu_split 필터링 중 오류 발생: {e}")

    model_name_lower = (server_config.LLM.LOCATION_MODEL).lower()
    if "gemma-4" in model_name_lower or "gemma4" in model_name_lower:
        print("⚠️ [우회 적용] Gemma-4 모델 감지됨: 멀티모달 OOM 및 어텐션 버그 방지를 위해 Flash Attention을 차단합니다.")
        # ExLlamaV3의 args 객체에 no_flash_attn 속성을 강제로 주입/True 설정
        setattr(args, "no_flash_attn", True)

    # 모델 & ExLlama 내부 토크나이저 로딩
    print(f"⏳ 모델 로딩 시작... {args.model_dir}")
    try:
        model, config, cache, tokenizer = model_init.init(args)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        sys.exit(1)

    # [공사 5] 이전 로그의 오류를 잡기 위해 fix_mistral_regex=True 플래그 추가 로딩
    print("⏳ HF AutoTokenizer (프롬프트 템플릿용) 로딩 시도...")
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, fix_mistral_regex=True)
        print("✅ HF AutoTokenizer 로드 완료!")
    except Exception as e:
        print(f"❌ HF AutoTokenizer 로드 실패: {e}")
        sys.exit(1)

    print("👀 Vision Encoder 로딩 시도...")
    try:
        vision_model = Model.from_config(config, component="vision")
        vision_model.load()
        print("✅ Vision Encoder 로드 성공! (이미지 처리 가능)")
    except Exception as e:
        vision_model = None
        print(f"ℹ️ Vision Encoder 로드 실패 (텍스트 전용 모델일 수 있음): {e}")

    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    print(f"🚀 서버 시작: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)