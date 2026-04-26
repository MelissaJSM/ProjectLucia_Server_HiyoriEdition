# ──────────────────────────────────────────────────────────────────────────────
# Ui/ws_server.py
# FastAPI 기반의 WebSocket 서버 및 REST API 엔드포인트를 정의합니다.
# 클라이언트와의 실시간 통신, 파일 업로드, LLM/TTS 요청 처리 등을 담당합니다.
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import sys
import gc
import json
import uuid
import time
import asyncio
import traceback
import re
import threading
import http.server
import socketserver
import ast
import urllib.request
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime

# --- Core 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- FastAPI / WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks, HTTPException, Request

# --- Core 모듈
import Core.server_config as server_config
from Core.sql import MySQLManager
from Core.llm_handler import generate_llm_response, InputTypeValue, init_tokenizer
from Core.emotion_analyzer import analyze_emotion
from Core.tts_client import text_to_speech
from Core.runtime_control import restart_auto

# --- GPU Info (pynvml)
try:
    import pynvml
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, \
        nvmlDeviceGetName

    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

# ──────────────────────────────────────────────────────────────────────────────
# 설정 및 상수
# ──────────────────────────────────────────────────────────────────────────────

AUDIO_SAVE_PATH = os.environ.get("AUDIO_SAVE_PATH", "audio_files")
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

TEMP_IMAGE_PATH = os.path.join(os.path.dirname(AUDIO_SAVE_PATH), "temp_images")
os.makedirs(TEMP_IMAGE_PATH, exist_ok=True)

HEARTBEAT_INTERVAL = 1.0
HEARTBEAT_TIMEOUT = 60.0

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
FILE_SERVER_PORT = server_config.PORTS.AUDIO_SERVER_PORT

# [STATE STORE] 클라이언트별 화면 관찰 상태 저장소
client_states: Dict[str, Dict[str, Any]] = {}
SERVER_IS_READY = False


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수 & 파일 서버
# ──────────────────────────────────────────────────────────────────────────────

def get_all_vram_info() -> List[Dict[str, Any]]:
    infos = []
    if not _HAS_NVML: return infos
    try:
        nvmlInit()
        for i in range(nvmlDeviceGetCount()):
            h = nvmlDeviceGetHandleByIndex(i)
            m = nvmlDeviceGetMemoryInfo(h)
            name = nvmlDeviceGetName(h)
            if isinstance(name, bytes): name = name.decode("utf-8")
            infos.append({
                "gpu_id": i,
                "gpu_name": name,
                "gpu_used": m.used // 1024 ** 2,
                "gpu_total": m.total // 1024 ** 2
            })
        nvmlShutdown()
    except Exception:
        pass
    return infos


async def ws_error(ws: WebSocket, message: str, code: int = 500):
    try:
        await ws.send_json({"op": "error", "code": code, "message": message})
    except Exception:
        pass


def _cleanup_temp_images(paths: List[str]):
    if not paths: return
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


class AudioFileHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=AUDIO_SAVE_PATH, **kwargs)

    def log_message(self, format, *args): pass

    def do_GET(self):
        if self.path.startswith("/audio/"): self.path = self.path.replace("/audio/", "/", 1)
        return super().do_GET()


def start_file_server():
    retries = 5
    for i in range(retries):
        try:
            socketserver.TCPServer.allow_reuse_address = True
            server = socketserver.TCPServer(("", FILE_SERVER_PORT), AudioFileHandler)
            print(f"📂 File Server started on port {FILE_SERVER_PORT}")
            server.serve_forever()
            break
        except Exception:
            time.sleep(1)


_file_server_thread = None


# ──────────────────────────────────────────────────────────────────────────────
# Connection Manager (WebSocket 연결 관리)
# ──────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}
        self.last_seen: Dict[str, float] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active[client_id] = websocket
        self.last_seen[client_id] = time.time()
        print(f"[WS] connected: {client_id}")
        try:
            await websocket.send_json({"op": "hello", "client_id": client_id})
        except Exception:
            pass

    def disconnect(self, client_id: str):
        if client_id in self.active:
            del self.active[client_id]
            del self.last_seen[client_id]
            print(f"[WS] disconnected: {client_id}")
            if client_id in client_states:
                state = client_states.pop(client_id)
                stored_images = state.get("stored_images", [])
                if stored_images:
                    _cleanup_temp_images(stored_images)

    def mark_seen(self, client_id: str):
        if client_id in self.active: self.last_seen[client_id] = time.time()

    async def heartbeat_tick(self):
        now = time.time()
        for cid in list(self.active.keys()):
            if (now - self.last_seen.get(cid, 0)) > HEARTBEAT_TIMEOUT:
                try:
                    await self.active[cid].close()
                except Exception:
                    pass
                self.disconnect(cid)
            else:
                try:
                    await self.active[cid].send_json({"op": "server_ping", "ts": now})
                except Exception:
                    self.disconnect(cid)


manager = ConnectionManager()


async def _heartbeat_loop():
    while True:
        await manager.heartbeat_tick()
        await asyncio.sleep(HEARTBEAT_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# Service Check Helpers
# ──────────────────────────────────────────────────────────────────────────────

async def check_port_open(host: str, port: int) -> bool:
    try:
        future = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(future, timeout=0.5)
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


async def wait_for_services(timeout: int = 60) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        llm_ok = await check_port_open(server_config.PORTS.LLAMA_HOST, server_config.PORTS.LLAMA_PORT)
        tts_ok = True
        if server_config.TTS.TTS_ENABLE:
            tts_ok = await check_port_open(server_config.PORTS.TTS_HOST, server_config.PORTS.TTS_PORT)
        audio_ok = await check_port_open("127.0.0.1", FILE_SERVER_PORT)
        if llm_ok and tts_ok and audio_ok: return True
        await asyncio.sleep(1.0)
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Init & Lifespan
# ──────────────────────────────────────────────────────────────────────────────

def init_server_settings():
    try:
        db = MySQLManager()
        settings = db.fetch_server_settings()
        server_config.LLM.COMMU_LOG_TIME = bool(settings.get("commu_log_time", 0))
        server_config.LLM.COMMU_LOG_INTERVAL = int(settings.get("commu_log_interval", 10))
        db.close()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SERVER_IS_READY
    init_server_settings()
    global _file_server_thread
    _file_server_thread = threading.Thread(target=start_file_server, daemon=True)
    _file_server_thread.start()

    async def do_warmup():
        global SERVER_IS_READY
        is_ready = await wait_for_services(timeout=120)
        if is_ready:
            try:
                def _run():
                    init_tokenizer()
                    try:
                        analyze_emotion("테스트")
                    except Exception:
                        pass
                    try:
                        req_data = json.dumps(
                            {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}).encode('utf-8')
                        req = urllib.request.Request(
                            f"http://{server_config.PORTS.LLAMA_HOST}:{server_config.PORTS.LLAMA_PORT}/v1/chat/completions",
                            data=req_data, headers={'Content-Type': 'application/json'})
                        with urllib.request.urlopen(req, timeout=30) as _:
                            pass
                    except Exception:
                        pass
                    if server_config.TTS.TTS_ENABLE:
                        try:
                            text_to_speech("아", "Neutral")
                        except Exception:
                            pass

                await asyncio.to_thread(_run)
            except Exception:
                pass
        SERVER_IS_READY = True

    asyncio.create_task(do_warmup())
    hb_task = asyncio.create_task(_heartbeat_loop())
    try:
        yield
    finally:
        hb_task.cancel()


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI Endpoints
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Lucia WS Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
async def health():
    if not SERVER_IS_READY: return JSONResponse(status_code=503, content={"status": "warming_up"})
    return PlainTextResponse("ok")


@app.get("/hb/state")
async def hb_state():
    if not SERVER_IS_READY: return JSONResponse(status_code=503, content={"ok": False})
    return JSONResponse({"ok": True, "active_count": len(manager.active), "gpus": get_all_vram_info()})


@app.post("/restart")
async def restart(request: Request, bg: BackgroundTasks):
    bg.add_task(restart_auto)
    return JSONResponse({"ok": True})


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        image_id = f"{uuid.uuid4()}{ext}"
        path = os.path.join(TEMP_IMAGE_PATH, image_id)
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
        return {"ok": True, "image_id": image_id}
    except Exception as e:
        raise HTTPException(500, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Handlers (핵심 비즈니스 로직)
# ──────────────────────────────────────────────────────────────────────────────

async def handle_observe(websocket: WebSocket, client_id: str, data: dict):
    # 1. 상태 초기화 및 Busy Lock 체크
    if client_id not in client_states:
        client_states[client_id] = {
            "summary": "No record",
            "stored_images": [],
            "cooldown": 0,
            "last_spoken": "None",
            "is_processing": False  # 🟢 작업 중 플래그
        }

    state = client_states[client_id]

    # 🚨 처리 중이면 새로운 요청 무시 (대기열 체증 방지)
    if state.get("is_processing"):
        return

    current_image_id = data.get("image_id")
    if not current_image_id: return
    current_image_path = os.path.join(TEMP_IMAGE_PATH, current_image_id)
    if not os.path.exists(current_image_path): return

    # 🔒 분석 작업 시작 잠금
    state["is_processing"] = True

    try:
        last_summary = state.get("summary", "No previous record")
        cooldown = state.get("cooldown", 0)
        stored_images = list(state.get("stored_images", []))
        last_spoken = state.get("last_spoken", "None")

        image_paths_for_llm = stored_images + [current_image_path]
        count = len(image_paths_for_llm)
        SPEECH_THRESHOLD = 6

        if count >= 3:
            img_context_str = "Input: [Img1(Past)] -> [Img2(Past)] -> [Img3(CURRENT)]."
            task_str = "Task: Compare [CURRENT] with [Past]. Rate the DEGREE OF CHANGE/EVENT."
        else:
            img_context_str = "Input: [Img1(CURRENT)]."
            task_str = "Task: Analyze the current screen."

        # 🟢 [수정됨] 불필요한 시스템 명령어(JSON 규칙 등) 제거. 순수 동적 데이터(상황과 목표)만 전달
        analyze_prompt = f"{img_context_str}\n{task_str}"

        def step1_analyze():
            return generate_llm_response(
                user_input=analyze_prompt,
                recent_conversation=[],
                inputType=InputTypeValue.OBSERVE,  # 🟢 자아를 분리한 순수 분석 모드
                emotion="Neutral",
                image_paths=image_paths_for_llm
            )

        try:
            # 🟢 타임아웃 대폭 연장 (31B 모델의 이미지 3장 처리 대응)
            raw_res_1 = await asyncio.wait_for(asyncio.to_thread(step1_analyze), timeout=30.0)
        except asyncio.TimeoutError:
            print(f"⚠️ [DEBUG] 1단계 분석 타임아웃 발생 (30초 초과)")
            raw_res_1 = '{"score": 0, "summary": "Timeout", "reason": "System Timeout"}'

        # 🟢 JSON 추출 및 방어적 파싱 로직
        res_json = {}
        try:
            start_idx = raw_res_1.find('{')
            if start_idx != -1:
                json_str = raw_res_1[start_idx:raw_res_1.rfind('}') + 1]
                res_json = json.loads(json_str)
        except:
            pass

        # JSON이 깨졌거나 점수가 없을 때 정규식 억지 추출
        if not res_json or "score" not in res_json:
            score_match = re.search(r'score["\']?\s*[:=]\s*(\d+)', raw_res_1, re.I)
            if not score_match:
                score_match = re.search(r'(\d+)\s*점', raw_res_1)

            if score_match:
                res_json["score"] = int(score_match.group(1))
            else:
                nums = re.findall(r'\b([0-9]|10)\b', raw_res_1)
                res_json["score"] = int(nums[0]) if nums else 0

            res_json["summary"] = raw_res_1[:50] + "..." if not res_json.get("summary") else res_json["summary"]
            res_json["reason"] = "Regex Fallback"

        score = int(res_json.get("score", 0))
        summary = res_json.get("summary", "")
        reason = res_json.get("reason", "Regex Fallback")

        # 큐 정리
        next_stored_images = stored_images + [current_image_path]
        images_to_delete = []
        while len(next_stored_images) > 2:
            images_to_delete.append(next_stored_images.pop(0))

        if count < 3:
            print(f"👀 Observe[{client_id}]: Warm-up ({count}/3). Ignored.")
            state.update({"summary": summary, "stored_images": next_stored_images})
            _cleanup_temp_images(images_to_delete)
            await websocket.send_json(
                {"op": "observe_result", "should_speak": False, "message": None, "reason": "Warm-up"})
            return

        should_speak = False
        message_to_send = None
        final_reason = reason
        emotion_res = "Neutral"
        audio_filename = None
        audio_url = None

        if score >= SPEECH_THRESHOLD:
            if cooldown > 0 and score < 8:
                should_speak = False
                final_reason = f"[Skipped by Cooldown: {cooldown}] {reason}"
                cooldown -= 1
            else:
                should_speak = True
                cooldown = 3 if score < 8 else 2
        else:
            if cooldown > 0: cooldown -= 1

        # 2단계: 루시아 대사 생성
        if should_speak:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lucia_persona = server_config.LLM.LLM_CHAT_FORMAT.format(userName=server_config.LLM.LLM_USER_NAME,
                                                                     characterName=server_config.LLM.LLM_CHARACTER_NAME)

            actor_prompt = (
                f"{lucia_persona}\n"
                "--------------------------------------------------\n"
                "!!! SYSTEM INSTRUCTION: REACT TO THE SCREEN !!!\n"
                f"CONTEXT SUMMARY (Detection): {summary}\n"
                f"REASON: {reason} (Score: {score})\n"
                f"Your Last Message: \"{last_spoken}\" (Avoid repeating this semantic content.)\n"
                "--------------------------------------------------\n"
                "Task: Look at the [CURRENT IMAGE] and the [CONTEXT SUMMARY].\n"
                "If score is 6-7: Say something mild/encouraging.\n"
                "If score is 8-9: React with excitement/surprise! The visual is interesting.\n"
                "\n"
                "As Lucia, say something spontaneous based on the CURRENT situation.\n"
                "\n"
                "Output: Just one sentence in Korean. (No JSON)."
            )

            def step2_act():
                return generate_llm_response(
                    user_input=actor_prompt,
                    recent_conversation=[],
                    inputType=InputTypeValue.CHAT,  # 🟡 여기는 정상적인 대화 모드로 자아 발동
                    emotion="Neutral",
                    image_paths=[current_image_path]
                )

            try:
                raw_res_2 = await asyncio.wait_for(asyncio.to_thread(step2_act), timeout=15.0)
                message_to_send = raw_res_2.replace('"', '').replace("Lucia:", "").strip()

                if message_to_send:
                    def process_tts(text):
                        tts_text = re.sub(r'[^\u0000-\uFFFF]', '', text)
                        emo = analyze_emotion(text)
                        wav_bytes = text_to_speech(tts_text, emo) if server_config.TTS.TTS_ENABLE else None
                        return emo, wav_bytes

                    emo, wav_bytes = await asyncio.to_thread(process_tts, message_to_send)
                    emotion_res = emo

                    if wav_bytes is not None:
                        fname = f"{uuid.uuid4()}.wav"
                        with open(os.path.join(AUDIO_SAVE_PATH, fname), "wb") as f:
                            f.write(wav_bytes)

                        host = websocket.url.hostname or "localhost"
                        if host == "0.0.0.0": host = "localhost"

                        audio_filename = fname
                        audio_url = f"http://{host}:{FILE_SERVER_PORT}/{fname}"

            except asyncio.TimeoutError:
                print(f"⚠️ Actor Timeout. Message skipped.")
                message_to_send = None
                should_speak = False
            except Exception as e:
                print(f"⚠️ Actor/TTS Error: {e}")
                if not message_to_send: should_speak = False

        state.update({
            "summary": summary if summary else last_summary,
            "stored_images": next_stored_images,
            "cooldown": cooldown,
            "last_spoken": message_to_send if should_speak and message_to_send else last_spoken
        })

        _cleanup_temp_images(images_to_delete)

        await websocket.send_json({
            "op": "observe_result",
            "should_speak": should_speak,
            "llm_response": message_to_send,
            "reason": f"[Score: {score}] {final_reason}",
            "emotion": emotion_res,
            "audio_filename": audio_filename,
            "audio_url": audio_url
        })
        print(
            f"👀 Observe[{client_id}]: Score={score} (Cool={cooldown}) -> Speak={should_speak} / Msg: {message_to_send}")

    except Exception as e:
        print(f"⚠️ Observe Error: {e}")
        traceback.print_exc()
        _cleanup_temp_images([current_image_path])

    finally:
        # 🔓 모든 작업 종료 후 락 해제 (다음 프레임 대기)
        if client_id in client_states:
            client_states[client_id]["is_processing"] = False


async def handle_notification(websocket: WebSocket, client_id: str, data: dict):
    app_name = data.get("app_name", "System")
    content = data.get("content", "")
    image_id = data.get("image_id")

    if not content and not image_id:
        return

    # 이미지가 넘어왔을 경우 경로 확인
    valid_paths = []
    if image_id:
        img_path = os.path.join(TEMP_IMAGE_PATH, image_id)
        if os.path.exists(img_path):
            valid_paths.append(img_path)

    # 1. 루시아 페르소나 및 알림 상황 프롬프트 설정
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lucia_persona = server_config.LLM.LLM_CHAT_FORMAT.format(userName=server_config.LLM.LLM_USER_NAME,
                                                             characterName=server_config.LLM.LLM_CHARACTER_NAME)

    prompt = (
        f"{lucia_persona}\n"
        "--------------------------------------------------\n"
        "!!! 시스템 지침: 알림에 대한 응답!!!\n"
        f"App Name: {app_name}\n"
        f"Content Summary / Image Context: {content}\n"
        "--------------------------------------------------\n"
        "작업: 방금 화면에서 시스템 알림 또는 메시지를 받았습니다.\n"
        "캐릭터 컨셉을 반드시 맞춰서 행동하고 알림과 관련된 말을 하세요. 짧게 하세요\n"
        "JSON을 출력하지 말고 자연스럽게 말하세요."
    )

    def process_notification():
        # LLM 호출
        llm_out = generate_llm_response(
            user_input=prompt,
            recent_conversation=[],
            inputType=InputTypeValue.NOTIFICATION,  # 🔥 [여기!] CHAT 대신 NOTIFICATION 으로 수정
            emotion="Neutral",
            image_paths=valid_paths
        )

        # 앞뒤 따옴표 및 Lucia: 같은 태그 제거
        message = llm_out.replace('"', '').replace("Lucia:", "").strip()

        # TTS 변환 및 감정 추출
        tts_text = re.sub(r'[^\u0000-\uFFFF]', '', message)
        emo_out = analyze_emotion(message)

        wav_bytes = None
        if server_config.TTS.TTS_ENABLE and message:
            wav_bytes = text_to_speech(tts_text, emo_out)

        return message, emo_out, wav_bytes

    try:
        # 비동기 스레드에서 처리
        message, emo_out, wav_bytes = await asyncio.to_thread(process_notification)

        audio_filename = None
        audio_url = None

        if wav_bytes is not None:
            fname = f"{uuid.uuid4()}.wav"
            with open(os.path.join(AUDIO_SAVE_PATH, fname), "wb") as f:
                f.write(wav_bytes)

            host = websocket.url.hostname or "localhost"
            if host == "0.0.0.0": host = "localhost"

            audio_filename = fname
            audio_url = f"http://{host}:{FILE_SERVER_PORT}/{fname}"

        # 2. 클라이언트로 결과 전송 (observe_result와 동일한 포맷을 보장하기 위해 should_speak 등 포함)
        await websocket.send_json({
            "op": "notification_result",
            "should_speak": True if message else False,
            "llm_response": message,
            "reason": f"[{app_name}] 알림 감지 반응",
            "emotion": emo_out,
            "audio_filename": audio_filename,
            "audio_url": audio_url
        })
        print(f"🔔 Notification[{client_id}] App: {app_name} -> Msg: {message}")

    except Exception as e:
        print(f"⚠️ Notification Handling Failed: {e}")
    finally:
        # 사용 완료된 임시 이미지 삭제
        _cleanup_temp_images(valid_paths)


async def handle_chat(websocket: WebSocket, data: dict):
    text = data.get("text", "")
    is_emotion = bool(data.get("emotion"))
    img_ids = data.get("image_ids") or []
    if data.get("image_id"): img_ids.append(data.get("image_id"))

    # 🟢 [추가] 클라이언트에서 'think' 값을 받아옵니다. (안 보내면 기본값 False로 초고속 모드)
    use_think = bool(data.get("think", False))

    user_info = {
        "name": data.get("user_name"),
        "gender": data.get("user_gender"),
        "birth_date": data.get("user_birth_date")
    }

    server_config.LLM.LLM_USER_NAME = user_info["name"]

    if not text and not img_ids: return

    await websocket.send_json({"op": "status", "stage": "fetch_context"})
    db = MySQLManager()
    recent_logs = await asyncio.to_thread(db.fetch_recent_logs)

    valid_paths = [os.path.join(TEMP_IMAGE_PATH, i) for i in img_ids if
                   os.path.exists(os.path.join(TEMP_IMAGE_PATH, i))]

    await websocket.send_json({"op": "status", "stage": "processing"})

    def process_chat():
        llm_out = generate_llm_response(
            user_input=text,
            recent_conversation=recent_logs,
            inputType=InputTypeValue.CHAT,
            emotion="Neutral" if is_emotion else "",
            image_paths=valid_paths,
            user_info=user_info,
            use_think=use_think  # 🟢 [추가] 추출한 값을 핸들러로 넘김
        )
        tts_text = re.sub(r'[^\u0000-\uFFFF]', '', llm_out)
        emo_out = analyze_emotion(llm_out)

        if server_config.TTS.TTS_ENABLE:
            wav_bytes = text_to_speech(tts_text, emo_out)
        else:
            wav_bytes = None

        return llm_out, emo_out, wav_bytes

    try:
        llm_res, emo_res, wav_bytes = await asyncio.to_thread(process_chat)

        audio_filename = None
        audio_url = None

        if wav_bytes is not None:
            fname = f"{uuid.uuid4()}.wav"
            with open(os.path.join(AUDIO_SAVE_PATH, fname), "wb") as f:
                f.write(wav_bytes)

            host = websocket.url.hostname or "localhost"
            if host == "0.0.0.0": host = "localhost"

            audio_filename = fname
            audio_url = f"http://{host}:{FILE_SERVER_PORT}/{fname}"

        try:
            await websocket.send_json({
                "op": "chat_result",
                "llm_response": llm_res,
                "emotion": emo_res,
                "audio_filename": audio_filename,
                "audio_url": audio_url
            })
        except RuntimeError as e:
            if "once a close message has been sent" in str(e):
                print(f"⚠️ Client disconnected before receiving the chat result.")
            else:
                raise e

    except Exception as e:
        print(f"⚠️ Chat Failed: {e}")
        await ws_error(websocket, str(e))
    finally:
        _cleanup_temp_images(valid_paths)


async def handle_feedback(websocket: WebSocket, data: dict):
    fb_text = data.get("feedback", "")
    num = data.get("number")
    if fb_text and num is not None:
        res = await asyncio.to_thread(generate_llm_response, fb_text, num, InputTypeValue.FEEDBACK, "")
        await websocket.send_json({"op": "feedback_result", "result": res})


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("⏳ Waiting for internal services to be ready before accepting WS...")
    is_ready = await wait_for_services(timeout=60)

    if not is_ready:
        print("⚠️ Connection rejected: Internal services not ready.")
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    client_id = websocket.query_params.get("client_id") or f"client-{uuid.uuid4().hex[:8]}"
    await manager.connect(websocket, client_id)

    try:
        while True:
            try:
                data_str = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break

            manager.mark_seen(client_id)

            try:
                obj = json.loads(data_str)
            except Exception:
                continue

            op = obj.get("op", "")
            data = obj.get("data", {})

            if op == "client_pong": continue
            if op == "ping":
                await websocket.send_json({"op": "pong", "ts": time.time()})
                continue
            if op == "monitoring":
                await websocket.send_json({"op": "monitoring", "gpus": get_all_vram_info()})
                continue

            if op == "observe":
                await handle_observe(websocket, client_id, data)
                continue

            if op == "notification":
                await handle_notification(websocket, client_id, data)
                continue

            if op == "chat":
                await handle_chat(websocket, data)
                continue

            if op == "feedback":
                await handle_feedback(websocket, data)
                continue

    finally:
        manager.disconnect(client_id)