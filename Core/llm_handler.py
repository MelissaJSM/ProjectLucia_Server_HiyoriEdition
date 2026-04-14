# Core/llm_handler.py
import os
import time
import json
import threading
import requests
import re
from datetime import datetime
from typing import List, Optional
from contextlib import ExitStack
from enum import IntEnum
import pytz

from Core.rag_search import preprocess_webrag
from Core.sql import MySQLManager
from transformers import AutoTokenizer
import Core.server_config as server_config

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8000")
LLAMA_SERVER_MODEL = os.environ.get("LLAMA_SERVER_MODEL", "lucia_gemma-4_31b_8_0")

model_lock = threading.Lock()
db_manager = MySQLManager()


class InputTypeValue(IntEnum):
    CHAT = 0
    FEEDBACK = 3


GLOBAL_TOKENIZER = None


def init_tokenizer():
    global GLOBAL_TOKENIZER
    if GLOBAL_TOKENIZER is None:
        try:
            model_path = os.path.join(server_config.LLM.LOCATION, server_config.LLM.LOCATION_MODEL)
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
            print(f"✅ Tokenizer 로드 완료: {model_path}")
        except Exception as e:
            print(f"⚠️ Tokenizer 로드 실패: {e}")


def _call_llama_server(data: dict, files: list = None) -> dict:
    url = LLAMA_SERVER_URL.rstrip("/") + "/v1/chat/completions"
    timeout = getattr(server_config.LLM, "TIMEOUT", 120)
    resp = requests.post(url, data=data, files=files, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def check_llm_backend_status() -> dict:
    test_req_dict = {"model": LLAMA_SERVER_MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1,
                     "temperature": 0.0, "stream": False}
    data = {"request_json": json.dumps(test_req_dict)}
    start = time.time()
    try:
        resp_json = _call_llama_server(data)
        latency_ms = int((time.time() - start) * 1000)
        if isinstance(resp_json, dict) and "choices" in resp_json: return {"status": "ok", "latency_ms": latency_ms}
        return {"status": "error", "detail": "invalid response", "latency_ms": latency_ms}
    except Exception as e:
        return {"status": "down", "detail": str(e)}


def _build_chat_messages(user_input, history_log, emotion, model_type, user_info=None, has_image=False):
    tz = pytz.timezone("Asia/Seoul")
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    rag_context = ""
    if not has_image:
        search_result = preprocess_webrag(question=user_input, search_query=user_input, max_results_search=5,
                                          use_embedding=True, select_top=True)
        rag_context = search_result.get("best_text", "")

    rag_prompt = ""
    if rag_context and "검색 결과가 없습니다" not in rag_context:
        rag_prompt = (
            f"\n\n[참고 정보 (RAG Search Result)]\n{rag_context}\n※ 위 검색 결과를 분석하여 사용자의 질문에 답변하십시오.\n※ 만약 정확한 정보가 없다면, 검색된 내용(제목, 출처 등)을 언급하며 '이런 정보들만 보인다'고 설명하십시오.\n※ 검색 결과를 무시하지 말고 반드시 답변에 포함시키십시오.\n")
        print(f"🔍 RAG 정보 추가됨: {len(rag_context)} chars")

    emotion_prompt = f" 현재 대화하는 사람의 감정 상태는 {emotion} 입니다. 감정 상태를 고려하여 대화하여 주십시오." if emotion else ""

    user_info_str = ""
    if user_info:
        parts = []
        if user_info.get("name"): parts.append(f"Name: {user_info['name']}")
        if user_info.get("gender"): parts.append(f"Gender: {user_info['gender']}")
        if user_info.get("birth_date"): parts.append(f"Birthday: {user_info['birth_date']}")
        if parts: user_info_str = f"[About the user you are talking to] {', '.join(parts)}\n"

    base_system_content = server_config.LLM.LLM_CHAT_FORMAT.format(recent_conversation=history_log,
                                                                   userEmotion=emotion_prompt, now=now)
    final_system_content = f"{user_info_str}{base_system_content}{rag_prompt}"
    final_system_content += "\n\n[시스템 중요 지시사항]\n당신은 사용자에게 정보를 제공하고 돕는 텍스트 기반 AI 어시스턴트입니다.\n절대로 소설이나 대본처럼 괄호 '( )'나 특수기호를 사용하여 당신의 행동, 감정, 표정, 동작 등을 지문으로 묘사하지 마십시오.\n오직 대화체의 답변 텍스트만 깔끔하게 출력하십시오."

    if "gemma" in model_type: final_system_content += "\n\n대화시작:"

    return [{"role": "system", "content": final_system_content}, {"role": "user", "content": user_input}]


def _build_feedback_messages(feedback_input, log_id, model_type):
    print(f"📝 피드백 데이터 조회 (ID: {log_id})")
    feedback_data = db_manager.feedback_call(log_id)
    formatted_input = f"질문: {feedback_data['user']}\n모델의 응답 : {feedback_data['assistant']}\n피드백: {feedback_input}"
    system_content = server_config.LLM.LLM_FEEDBACK_FORMAT
    system_content += "\n\n[시스템 중요 지시사항]\n절대로 괄호 '( )'를 사용하여 행동이나 감정을 묘사하지 마십시오. 오직 피드백에 대한 명확한 텍스트 답변만 제공하십시오."
    if "gemma" in model_type: system_content += "\n\n대화시작:"
    return [{"role": "system", "content": system_content}, {"role": "user", "content": formatted_input}]


def _execute_llm_request(messages, model_type, image_paths: Optional[List[str]] = None):
    # 🚨 VRAM 및 GPU에 따른 안전 마진 계산 (방어 로직 강화)
    try:
        gpu_split_str = getattr(server_config.LLM, "GPU_SPLIT", "0,24.0,24.0")
        total_vram_gb = sum(float(x) for x in gpu_split_str.split(','))
    except (ValueError, AttributeError):
        total_vram_gb = 24.0

    estimated_model_size_gb = 28.0
    free_vram = total_vram_gb - estimated_model_size_gb

    if free_vram > 15:
        base_margin = 4096
    elif free_vram > 5:
        base_margin = 2048
    else:
        base_margin = 1024

    if image_paths: base_margin += 1024

    try:
        base_context = int(server_config.LLM.CONTEXT)
    except (ValueError, TypeError):
        base_context = 4096
    aligned_context = ((base_context + 255) // 256) * 256

    if GLOBAL_TOKENIZER:
        prompt_text = " ".join([m["content"] for m in messages])
        prompt_tokens = len(GLOBAL_TOKENIZER.encode(prompt_text))
        dynamic_max_tokens = aligned_context - prompt_tokens - base_margin
        print(f"📊 [하드웨어 기반 계산] 총 VRAM: {total_vram_gb}GB / 마진: {base_margin} / 응답 한도: {dynamic_max_tokens}")
    else:
        dynamic_max_tokens = aligned_context - base_margin
        print(f"📊 [토큰 고정 계산] 응답 한도: {dynamic_max_tokens}")

    req_dict = {
        "model": LLAMA_SERVER_MODEL,
        "messages": messages,
        "stream": False,
        "max_response_tokens": dynamic_max_tokens,
        "stop": ["<end_of_turn>", "<eos>", "<|endoftext|>", "user:", "대화시작:", "<turn|>", "<|turn>", "<|/think|>"]
    }

    data = {"request_json": json.dumps(req_dict)}
    print(f"🚀 LLM 요청 시작 (Model: {model_type})")

    try:
        with ExitStack() as stack:
            files = []
            if image_paths:
                for path in image_paths:
                    if os.path.exists(path):
                        try:
                            f = stack.enter_context(open(path, "rb"))
                            files.append(("images", (os.path.basename(path), f, "image/jpeg")))
                            print(f"📷 [이미지 첨부] {path}")
                        except Exception as e:
                            print(f"⚠️ 이미지 파일 열기 실패 ({path}): {e}")
                    else:
                        print(f"⚠️ 이미지 파일 없음: {path}")

            with model_lock:
                response = _call_llama_server(data, files)

        result = response["choices"][0]["message"]["content"].strip()
        result = re.sub(r'<\|channel\|>.*?<\|channel\|>', '', result, flags=re.DOTALL)
        result = re.sub(r'<\|channel>.*?<channel\|>', '', result, flags=re.DOTALL)
        result = re.sub(r'<\|channel\|>.*', '', result, flags=re.DOTALL)

        result = result.replace("<end_of_turn>", "").replace("<turn|>", "").replace("<|turn>", "").replace("<bos>",
                                                                                                           "").replace(
            "<eos>", "").replace("<|/think|>", "")
        result = result.strip()

        print(f"🔹 LLM 응답: {result}")
        return result

    except (KeyError, IndexError, TypeError) as e:
        print(f"❌ LLM 응답 파싱 오류: {e}")
        if 'response' in locals(): print(f"Raw Response: {response}")
        raise


def generate_llm_response(user_input, recent_conversation, inputType, emotion, image_paths: Optional[List[str]] = None,
                          user_info: dict = None):
    model_type = server_config.LLM.MODEL_TYPE
    if inputType == InputTypeValue.CHAT:
        has_image = bool(image_paths)
        messages = _build_chat_messages(user_input, recent_conversation, emotion, model_type, user_info,
                                        has_image=has_image)
    elif inputType == InputTypeValue.FEEDBACK:
        messages = _build_feedback_messages(user_input, recent_conversation, model_type)
    else:
        print(f"❌ 알 수 없는 inputType: {inputType}")
        return None
    return _execute_llm_request(messages, model_type, image_paths)