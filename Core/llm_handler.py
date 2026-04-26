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
    OBSERVE = 4
    NOTIFICATION = 5  # 🔔 알림 반응 모드 (웹 검색/RAG 강제 스킵용) 추가!


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
    # 💡 이미지 없을 때 백그라운드에서 조용히 검색어(RAG)를 가져옴
    if not has_image:
        search_result = preprocess_webrag(question=user_input, search_query=user_input, max_results_search=5,
                                          use_embedding=True, select_top=True)
        rag_context = search_result.get("best_text", "")

    rag_prompt = ""
    if rag_context and "검색 결과가 없습니다" not in rag_context:
        rag_prompt = (
            f"\n\n[참고 정보 (네트워크 검색 결과)]\n{rag_context}\n"
            "※ 위 검색 결과를 바탕으로 자연스럽게 대답해. 모르는 내용이면, 검색 결과가 없으면 '검색해 봤는데 그런 건 안 보이는데요?'라고 말해.\n"
        )
        print(f"🔍 RAG 정보 추가됨: {len(rag_context)} chars")

    if emotion and emotion.strip():
        emotion_prompt = f" 현재 대화하는 사람의 감정 상태는 {emotion} 입니다. 이를 고려하여 반응하세요."
    else:
        emotion_prompt = ""

    time_prompt = f"현재 시간은 {now} 입니다. 시간 정보가 필요하면 이 시간을 참고해서 대화 하십시오."

    user_info_str = ""
    if user_info:
        # 각 정보가 없을 경우를 대비해 기본값(Default)을 설정해두면 더 안전해요!
        user_name = user_info.get("name", server_config.LLM.LLM_USER_NAME)
        user_gender = "남성" if user_info.get("gender") == "Man" else "여성"
        user_birthday = user_info.get("birth_date", "비공개")

        # 루시아의 시스템 프롬프트에 들어갈 서사적인 문구로 구성
        user_info_str = f"""
    ### [대화 하고있는 상대방  {server_config.LLM.LLM_USER_NAME}님에 대한 인적 정보 (상대방이 누군지 반드시 인지해주세요!)]
    - **성함(혹은 닉네임):** {user_name}
    - **성별:** {user_gender}
    - **생년월일:** {user_birthday}
    \n\n"""

    base_system_content = server_config.LLM.LLM_CHAT_FORMAT.format(userName=server_config.LLM.LLM_USER_NAME,
                                                                   characterName=server_config.LLM.LLM_CHARACTER_NAME)

    final_system_content = f"{base_system_content}{user_info_str}{time_prompt}{emotion_prompt}{rag_prompt}\n\n{history_log}"

    if "gemma" in model_type: final_system_content += "\n\n대화시작:"

    return [{"role": "system", "content": final_system_content}, {"role": "user", "content": user_input}]


def _build_feedback_messages(feedback_input, log_id, model_type):
    feedback_data = db_manager.feedback_call(log_id)
    formatted_input = f"질문: {feedback_data['user']}\n모델의 응답 : {feedback_data['assistant']}\n피드백: {feedback_input}"
    system_content = server_config.LLM.LLM_FEEDBACK_FORMAT
    if "gemma" in model_type: system_content += "\n\n대화시작:"
    return [{"role": "system", "content": system_content}, {"role": "user", "content": formatted_input}]


def _build_observe_messages(user_input, emotion, model_type, user_info=None):
    system_content = (
        "[시스템: 비전 분석 엔진]\n"
        "당신은 캐릭터 자아를 버리고 오직 시각적 변화를 JSON 포맷으로 출력하는 분석 엔진입니다.\n"
        "입력된 이미지를 비교하여 사건의 중요도와 구체적인 상황을 분석하십시오.\n\n"
        "[절대 규칙]\n"
        "1. 반드시 순수 JSON 텍스트만 출력하십시오.\n"
        "2. 마크다운 기호(```json)나 한국어 설명, 인사말을 절대 포함하지 마십시오.\n"
        "3. '{'로 시작해서 '}'로 끝나는 데이터 외에는 어떠한 텍스트도 출력해서는 안 됩니다.\n"
        "4. [활성 화면 집중]: 메인 화면 내부의 콘텐츠 변화를 중점적으로 확인하십시오.\n"
        "5. [점수 부여]: 동적이거나 흥미로운 변화가 있다면 8점이나 9점의 높은 점수를 부여하십시오.\n"
        "6. [상세 요약 필수]: 'summary' 항목에 단순히 짧은 단어를 적지 마십시오. 다음 단계의 챗봇이 텍스트만 읽고도 상황을 완벽히 머릿속에 그릴 수 있도록, 화면에서 어떤 텍스트, 이미지, 동작이 확인되는지 최대한 구체적인 문장으로 서술하십시오.\n"
        "7. [반복 금지]: 이전과 동일한 요약(Summary)을 반복하지 마십시오.\n\n"
        "[출력 포맷 (오직 JSON만)]\n"
        "{\n"
        '  "score": 정수(0~10),\n'
        '  "summary": "구체적이고 상세한 현재 상황 묘사 (문장형)",\n'
        '  "reason": "해당 점수를 부여한 논리적 이유"\n'
        "}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]


# 🔔 새로 추가된 알림 전용 메시지 빌더 (RAG 스킵)
def _build_notification_messages(user_input, emotion, model_type, user_info=None):
    tz = pytz.timezone("Asia/Seoul")
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    user_info_str = ""
    if user_info:
        # 각 정보가 없을 경우를 대비해 기본값(Default)을 설정해두면 더 안전해요!
        user_name = user_info.get("name", server_config.LLM.LLM_USER_NAME)
        user_gender = "남성" if user_info.get("gender") == "Man" else "여성"
        user_birthday = user_info.get("birth_date", "비공개")

        # 루시아의 시스템 프롬프트에 들어갈 서사적인 문구로 구성
        user_info_str = f"""
    ### [대화 하고있는 상대방  {server_config.LLM.LLM_USER_NAME}님에 대한 인적 정보 (상대방이 누군지 반드시 인지해주세요!)]
    - **성함(혹은 닉네임):** {user_name}
    - **성별:** {user_gender}
    - **생년월일:** {user_birthday}
    \n\n"""

    # RAG 검색 결과를 넣지 않고 페르소나만 깔끔하게 넣습니다.
    base_system_content = server_config.LLM.LLM_CHAT_FORMAT.format(userName=server_config.LLM.LLM_USER_NAME,
                                                                   characterName=server_config.LLM.LLM_CHARACTER_NAME)
    final_system_content = f"{user_info_str}{base_system_content}"

    if "gemma" in model_type: final_system_content += "\n\n대화시작:"

    return [{"role": "system", "content": final_system_content}, {"role": "user", "content": user_input}]


def _execute_llm_request(messages, model_type, image_paths: Optional[List[str]] = None, use_think: bool = False):
    try:
        base_context = int(getattr(server_config.LLM, "CONTEXT", 4096))
    except (ValueError, TypeError):
        base_context = 4096

    aligned_context = ((base_context + 255) // 256) * 256
    MIN_RESPONSE_TOKENS = 256

    if GLOBAL_TOKENIZER:
        prompt_text = " ".join([m["content"] for m in messages])
        prompt_tokens = len(GLOBAL_TOKENIZER.encode(prompt_text))
        available_space = aligned_context - prompt_tokens

        if image_paths:
            ideal_image_margin = len(image_paths) * 2048
            if available_space - ideal_image_margin < MIN_RESPONSE_TOKENS:
                dynamic_max_tokens = int(available_space * 0.3)
            else:
                dynamic_max_tokens = available_space - ideal_image_margin
        else:
            dynamic_max_tokens = int(available_space * 0.8)

        dynamic_max_tokens = max(dynamic_max_tokens, MIN_RESPONSE_TOKENS)
        print(f"📊 [유연한 토큰 계산] 컨텍스트: {aligned_context} / 입력: {prompt_tokens} / 응답 한도: {dynamic_max_tokens}")
    else:
        dynamic_max_tokens = int(aligned_context * (0.3 if image_paths else 0.7))
        dynamic_max_tokens = max(dynamic_max_tokens, MIN_RESPONSE_TOKENS)
        print(f"📊 [토크나이저 미탑재 폴백] 응답 한도: {dynamic_max_tokens}")

    req_dict = {
        "model": LLAMA_SERVER_MODEL,
        "messages": messages,
        "stream": False,
        "max_response_tokens": dynamic_max_tokens,
        "no_think": not use_think,
        "stop": ["<end_of_turn>", "<eos>", "<|endoftext|>", "user:", "대화시작:", "<turn|>", "<|turn>", "<|/think|>",
                 "<|tool_response>"]
    }

    data = {"request_json": json.dumps(req_dict)}
    print(f"🚀 LLM 요청 시작 (Model: {model_type} | Think 모드: {'켜짐' if use_think else '꺼짐'})")

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

        tags_to_remove = [
            "<|channel>thought", "<channel|>", "<|channel|>",
            "<|think|>", "</|think|>", "<bos>", "<eos>",
            "<end_of_turn>", "<turn|>", "<|turn>"
        ]
        for tag in tags_to_remove:
            result = result.replace(tag, "")

        result = result.strip()
        print(f"🔹 LLM 응답: {result}")
        return result

    except (KeyError, IndexError, TypeError) as e:
        print(f"❌ LLM 응답 파싱 오류: {e}")
        if 'response' in locals(): print(f"Raw Response: {response}")
        raise


def generate_llm_response(user_input, recent_conversation, inputType, emotion, image_paths: Optional[List[str]] = None,
                          user_info: dict = None, use_think: bool = False):
    model_type = getattr(server_config.LLM, "MODEL_TYPE", "gemma").lower()

    # 🟢 [1] 관찰 모드
    if inputType == InputTypeValue.OBSERVE:
        messages = _build_observe_messages(user_input, emotion, model_type, user_info)
        return _execute_llm_request(messages, model_type, image_paths, use_think=False)

    # 🟡 [2] 채팅 모드 (RAG 포함)
    elif inputType == InputTypeValue.CHAT:
        has_image = bool(image_paths)
        messages = _build_chat_messages(user_input, recent_conversation, emotion, model_type, user_info,
                                        has_image=has_image)
        print("💬 일반 대화 처리 (Legacy RAG 결합)")
        print(messages)
        return _execute_llm_request(messages, model_type, image_paths, use_think=use_think)

    # 🔵 [3] 피드백 모드
    elif inputType == InputTypeValue.FEEDBACK:
        messages = _build_feedback_messages(user_input, recent_conversation, model_type)
        return _execute_llm_request(messages, model_type, image_paths, use_think=use_think)

    # 🟠 [4] 알림 모드 (새로 추가)
    elif inputType == InputTypeValue.NOTIFICATION:
        messages = _build_notification_messages(user_input, emotion, model_type, user_info)
        print("🔔 알림 반응 처리 (웹 검색 스킵 적용됨)")
        return _execute_llm_request(messages, model_type, image_paths, use_think=use_think)

    else:
        print(f"❌ 알 수 없는 inputType: {inputType}")
        return None