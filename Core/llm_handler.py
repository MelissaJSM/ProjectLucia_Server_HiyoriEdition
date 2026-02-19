# ──────────────────────────────────────────────────────────────────────────────
# Core/llm_handler.py
# LLM 서버(ExLlamaV3 등)와 통신하여 텍스트 생성 요청을 처리하는 모듈입니다.
# ──────────────────────────────────────────────────────────────────────────────
import os
import time
import json
import threading
import requests
from datetime import datetime
from typing import List, Optional
from contextlib import ExitStack
from enum import IntEnum

import pytz

# 프로젝트 내부 모듈
from Core.rag_search import preprocess_webrag
from Core.sql import MySQLManager
import Core.server_config as server_config

# ──────────────────────────────────────────────────────────────────────────────
# 설정 및 상수
# ──────────────────────────────────────────────────────────────────────────────

# LLM 서버 URL 및 모델명 (환경변수 또는 기본값)
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8000")
LLAMA_SERVER_MODEL = os.environ.get("LLAMA_SERVER_MODEL", "lucia_gemma-3_27b_8_0")

# 요청 동시성 제어용 락 (Lock)
model_lock = threading.Lock()

# DB 매니저 인스턴스
db_manager = MySQLManager()


class InputTypeValue(IntEnum):
    """입력 유형을 정의하는 열거형 클래스"""
    CHAT = 0  # 일반 대화
    FEEDBACK = 3  # 피드백 처리


# ──────────────────────────────────────────────────────────────────────────────
# LLM 서버 통신 함수
# ──────────────────────────────────────────────────────────────────────────────

def _call_llama_server(data: dict, files: list = None) -> dict:
    """
    LLM 서버 API(/v1/chat/completions)를 호출합니다.
    Multipart/form-data 형식을 지원하여 이미지 전송이 가능합니다.
    """
    url = LLAMA_SERVER_URL.rstrip("/") + "/v1/chat/completions"
    timeout = getattr(server_config.LLM, "TIMEOUT", 120)

    # files는 [('images', (filename, file_obj, content_type)), ...] 형태여야 함
    resp = requests.post(url, data=data, files=files, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def check_llm_backend_status() -> dict:
    """
    LLM 서버의 상태를 확인합니다 (Health Check).
    간단한 'ping' 메시지를 보내 응답 시간과 성공 여부를 반환합니다.
    """
    test_req_dict = {
        "model": LLAMA_SERVER_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }

    data = {"request_json": json.dumps(test_req_dict)}

    start = time.time()
    try:
        resp_json = _call_llama_server(data)
        latency_ms = int((time.time() - start) * 1000)

        if isinstance(resp_json, dict) and "choices" in resp_json:
            return {"status": "ok", "latency_ms": latency_ms}
        return {"status": "error", "detail": "invalid response", "latency_ms": latency_ms}
    except Exception as e:
        return {"status": "down", "detail": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# 메시지 빌더 함수 (모드별 프롬프트 구성)
# ──────────────────────────────────────────────────────────────────────────────

def _build_chat_messages(user_input, history_log, emotion, model_type, user_info=None):
    """
    일반 대화(CHAT) 모드의 메시지를 구성합니다.
    시스템 프롬프트에 시간, 감정, 사용자 정보 등을 포함합니다.
    또한 RAG 검색을 수행하여 검색 결과가 있으면 프롬프트에 추가합니다.
    """
    tz = pytz.timezone("Asia/Seoul")
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    # 1. RAG 검색 수행 (상시 동작)
    # 검색어가 없으면 user_input 전체를 사용
    search_result = preprocess_webrag(
        question=user_input,
        search_query=user_input,
        max_results_search=5,
        use_embedding=True,
        select_top=True,
    )
    rag_context = search_result.get("best_text", "")

    # 검색 결과가 유효한 경우에만 프롬프트에 추가
    rag_prompt = ""
    if rag_context and "검색 결과가 없습니다" not in rag_context:
        rag_prompt = (
            f"\n\n[참고 정보 (RAG Search Result)]\n"
            f"{rag_context}\n"
            f"※ 위 검색 결과를 분석하여 사용자의 질문에 답변하십시오.\n"
            f"※ 만약 정확한 정보가 없다면, 검색된 내용(제목, 출처 등)을 언급하며 '이런 정보들만 보인다'고 설명하십시오.\n"
            f"※ 검색 결과를 무시하지 말고 반드시 답변에 포함시키십시오.\n"
        )
        print(f"🔍 RAG 정보 추가됨: {len(rag_context)} chars")

    # 2. 감정 정보 프롬프트
    emotion_prompt = ""
    if emotion:
        emotion_prompt = (
            f" 현재 대화하는 사람의 감정 상태는 {emotion} 입니다. "
            f"감정 상태를 고려하여 대화하여 주십시오."
        )

    # 3. 사용자 정보 프롬프트
    user_info_str = ""
    if user_info:
        parts = []
        if user_info.get("name"): parts.append(f"Name: {user_info['name']}")
        if user_info.get("gender"): parts.append(f"Gender: {user_info['gender']}")
        if user_info.get("birth_date"): parts.append(f"Birthday: {user_info['birth_date']}")

        if parts:
            user_info_str = f"[About the user you are talking to] {', '.join(parts)}\n"

    # 4. 시스템 프롬프트 조립
    # 순서 변경: 사용자 정보 -> 기본 설정(캐릭터) -> RAG 정보 (가장 최신/중요)
    base_system_content = server_config.LLM.LLM_CHAT_FORMAT.format(
        recent_conversation=history_log,
        userEmotion=emotion_prompt,
        now=now,
    )

    # RAG 정보를 캐릭터 설정 뒤에 배치하여 우선순위를 높임
    final_system_content = f"{user_info_str}{base_system_content}{rag_prompt}"

    print(f"📝 최종 시스템 프롬프트:\n{final_system_content}")

    # Gemma 모델 특화 처리
    if "gemma" in model_type:
        final_system_content += "\n\n대화시작:"

    return [
        {"role": "system", "content": final_system_content},
        {"role": "user", "content": user_input},
    ]


def _build_feedback_messages(feedback_input, log_id, model_type):
    """
    피드백(FEEDBACK) 모드의 메시지를 구성합니다.
    이전 대화 로그를 조회하여 피드백 내용을 반영한 답변을 생성합니다.
    """
    print(f"📝 피드백 데이터 조회 (ID: {log_id})")
    feedback_data = db_manager.feedback_call(log_id)

    formatted_input = (
        f"질문: {feedback_data['user']}\n"
        f"모델의 응답 : {feedback_data['assistant']}\n"
        f"피드백: {feedback_input}"
    )

    system_content = server_config.LLM.LLM_FEEDBACK_FORMAT

    if "gemma" in model_type:
        system_content += "\n\n대화시작:"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": formatted_input},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# LLM 요청 실행 및 응답 처리
# ──────────────────────────────────────────────────────────────────────────────

def _execute_llm_request(messages, model_type, image_paths: Optional[List[str]] = None):
    """
    구성된 메시지와 이미지를 사용하여 LLM 서버에 요청을 보내고 응답을 반환합니다.
    """
    # 1. 요청 데이터 구성
    req_dict = {
        "model": LLAMA_SERVER_MODEL,
        "messages": messages,
        "stream": False,
        "max_response_tokens": 1000
    }

    # 2. Multipart 데이터 준비 (JSON은 문자열로 변환)
    data = {
        "request_json": json.dumps(req_dict)
    }

    print(f"🚀 LLM 요청 시작 (Model: {model_type})")

    try:
        with ExitStack() as stack:
            files = []
            # 3. 이미지 파일 처리 (다중 이미지 지원)
            if image_paths:
                for path in image_paths:
                    if os.path.exists(path):
                        try:
                            f = stack.enter_context(open(path, "rb"))
                            # 키 이름을 'images'로 통일하여 리스트로 전송 (서버측 처리 방식에 따름)
                            files.append(("images", (os.path.basename(path), f, "image/jpeg")))
                            print(f"📷 [이미지 첨부] {path}")
                        except Exception as e:
                            print(f"⚠️ 이미지 파일 열기 실패 ({path}): {e}")
                    else:
                        print(f"⚠️ 이미지 파일 없음: {path}")

            # 4. 서버 호출 (Thread-safe)
            with model_lock:
                response = _call_llama_server(data, files)

        # 5. 응답 파싱
        result = response["choices"][0]["message"]["content"].strip()
        print(f"🔹 LLM 응답: {result}")
        return result

    except (KeyError, IndexError, TypeError) as e:
        print(f"❌ LLM 응답 파싱 오류: {e}")
        if 'response' in locals():
            print(f"Raw Response: {response}")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# 메인 진입점 (Facade)
# ──────────────────────────────────────────────────────────────────────────────

def generate_llm_response(user_input, recent_conversation, inputType, emotion, image_paths: Optional[List[str]] = None,
                          user_info: dict = None):
    """
    LLM 응답을 생성하는 메인 함수입니다.
    입력 타입(inputType)에 따라 적절한 메시지 빌더를 호출하고 요청을 실행합니다.

    Args:
        user_input (str): 사용자 입력 텍스트
        recent_conversation (str): 최근 대화 내역
        inputType (InputTypeValue): 입력 유형 (CHAT, FEEDBACK)
        emotion (str): 사용자 감정 상태
        image_paths (List[str], optional): 이미지 파일 경로 리스트
        user_info (dict, optional): 사용자 정보 (이름, 성별, 생년월일)

    Returns:
        str: LLM이 생성한 응답 텍스트
    """
    model_type = server_config.LLM.MODEL_TYPE

    if inputType == InputTypeValue.CHAT:
        # 일반 대화 모드에서 RAG 검색이 통합됨
        messages = _build_chat_messages(user_input, recent_conversation, emotion, model_type, user_info)

    elif inputType == InputTypeValue.FEEDBACK:
        # FEEDBACK 모드에서는 recent_conversation 인자를 로그 ID로 사용
        messages = _build_feedback_messages(user_input, recent_conversation, model_type)

    else:
        print(f"❌ 알 수 없는 inputType: {inputType}")
        return None

    return _execute_llm_request(messages, model_type, image_paths)
