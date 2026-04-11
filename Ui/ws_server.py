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
import urllib.request  # 워밍업 직접 호출을 위해 추가
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
from Core.llm_handler import generate_llm_response, InputTypeValue
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

# 오디오 및 임시 이미지 저장 경로
AUDIO_SAVE_PATH = os.environ.get("AUDIO_SAVE_PATH", "audio_files")
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

TEMP_IMAGE_PATH = os.path.join(os.path.dirname(AUDIO_SAVE_PATH), "temp_images")
os.makedirs(TEMP_IMAGE_PATH, exist_ok=True)

# 하트비트 설정
HEARTBEAT_INTERVAL = 1.0
HEARTBEAT_TIMEOUT = 60.0

# 관리자 토큰
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# 파일 서버 포트 (server_config에서 로드)
FILE_SERVER_PORT = server_config.PORTS.AUDIO_SERVER_PORT

# [STATE STORE] 클라이언트별 화면 관찰 상태 저장소
client_states: Dict[str, Dict[str, Any]] = {}

# [서버 상태 플래그] 워밍업 완료 여부를 추적하여 UI 동기화에 사용
SERVER_IS_READY = False


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수 & 파일 서버
# ──────────────────────────────────────────────────────────────────────────────

def get_all_vram_info() -> List[Dict[str, Any]]:
    """모든 GPU의 VRAM 사용량 정보를 조회합니다."""
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
                "gpu_used": m.used // 1024 ** 2,  # MB 단위
                "gpu_total": m.total // 1024 ** 2
            })
        nvmlShutdown()
    except Exception:
        pass
    return infos


async def ws_error(ws: WebSocket, message: str, code: int = 500):
    """WebSocket을 통해 에러 메시지를 전송합니다."""
    try:
        await ws.send_json({"op": "error", "code": code, "message": message})
    except Exception:
        pass


def _cleanup_temp_images(paths: List[str]):
    """임시 이미지 파일들을 삭제합니다."""
    if not paths: return
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


class AudioFileHandler(http.server.SimpleHTTPRequestHandler):
    """생성된 오디오 파일을 제공하기 위한 HTTP 핸들러"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=AUDIO_SAVE_PATH, **kwargs)

    def log_message(self, format, *args): pass  # 로그 출력 억제

    def do_GET(self):
        if self.path.startswith("/audio/"): self.path = self.path.replace("/audio/", "/", 1)
        return super().do_GET()


def start_file_server():
    """백그라운드 스레드에서 파일 서버를 시작합니다."""
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
        await websocket.send_json({"op": "hello", "client_id": client_id})

    def disconnect(self, client_id: str):
        if client_id in self.active:
            del self.active[client_id]
            del self.last_seen[client_id]
            print(f"[WS] disconnected: {client_id}")

            if client_id in client_states:
                state = client_states.pop(client_id)
                stored_images = state.get("stored_images", [])
                if stored_images:
                    print(f"🧹 Cleaning up {len(stored_images)} temp images for {client_id}")
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
# Service Check Helpers (내부 서비스 상태 확인)
# ──────────────────────────────────────────────────────────────────────────────

async def check_port_open(host: str, port: int) -> bool:
    """비동기로 특정 호스트:포트 연결 가능 여부를 확인합니다."""
    try:
        future = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(future, timeout=0.5)
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


async def wait_for_services(timeout: int = 60) -> bool:
    """LLM, TTS, Audio 서버가 모두 준비될 때까지 대기합니다."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        llm_ok = await check_port_open(server_config.PORTS.LLAMA_HOST, server_config.PORTS.LLAMA_PORT)

        tts_ok = True
        if server_config.TTS.TTS_ENABLE:
            tts_ok = await check_port_open(server_config.PORTS.TTS_HOST, server_config.PORTS.TTS_PORT)

        audio_ok = await check_port_open("127.0.0.1", FILE_SERVER_PORT)

        if llm_ok and tts_ok and audio_ok:
            return True

        await asyncio.sleep(1.0)
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Init & Lifespan (서버 시작/종료 시 처리 및 스텔스 워밍업)
# ──────────────────────────────────────────────────────────────────────────────

def init_server_settings():
    try:
        db = MySQLManager()
        settings = db.fetch_server_settings()
        server_config.LLM.COMMU_LOG_TIME = bool(settings.get("commu_log_time", 0))
        server_config.LLM.COMMU_LOG_INTERVAL = int(settings.get("commu_log_interval", 10))
        db.close()
    except Exception as e:
        print(f"⚠️ Settings load failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 앱의 수명 주기 관리 및 비동기 워밍업"""
    global SERVER_IS_READY
    init_server_settings()

    # 파일 서버 스레드 시작
    global _file_server_thread
    _file_server_thread = threading.Thread(target=start_file_server, daemon=True)
    _file_server_thread.start()

    # 🚀 백그라운드 스텔스 워밍업
    async def do_warmup():
        global SERVER_IS_READY
        print("⏳ Waiting for internal services to be ready for warmup...")

        # LLM/TTS 포트가 열릴 때까지 최대 120초 대기
        is_ready = await wait_for_services(timeout=120)

        if is_ready:
            print("🔥 Services are online! Executing stealth warmup...")
            try:
                def _run():
                    # 1. LLM 직접 API 호출 (로그 오염 방지, 1토큰 생성으로 시간 단축)
                    try:
                        req_data = json.dumps({
                            "messages": [{"role": "user", "content": "Wake up"}],
                            "max_tokens": 1
                        }).encode('utf-8')

                        req = urllib.request.Request(
                            f"http://{server_config.PORTS.LLAMA_HOST}:{server_config.PORTS.LLAMA_PORT}/v1/chat/completions",
                            data=req_data,
                            headers={'Content-Type': 'application/json'}
                        )
                        with urllib.request.urlopen(req, timeout=30) as response:
                            pass
                    except Exception as e:
                        print(f"⚠️ LLM Warmup warning (Ignored): {e}")

                    # 2. TTS 호출 (사전 다운로드 및 텐서 초기화)
                    if server_config.TTS.TTS_ENABLE:
                        try:
                            _ = text_to_speech("아", "Neutral")
                        except Exception as e:
                            print(f"⚠️ TTS Warmup warning (Ignored): {e}")

                await asyncio.to_thread(_run)
                print("✅ Models are warmed up silently and ready!")
            except Exception as e:
                print(f"⚠️ Warmup failed: {e}")
        else:
            print("⚠️ Could not warmup. Internal services timeout.")

        # 워밍업 성공 여부와 관계없이 프로세스가 끝났으므로 서비스 레디 상태로 전환
        SERVER_IS_READY = True

    # 워밍업을 백그라운드 태스크로 등록하여 FastAPI 시작을 블로킹하지 않음
    asyncio.create_task(do_warmup())

    # 하트비트 태스크 시작
    hb_task = asyncio.create_task(_heartbeat_loop())
    print("✅ Server Started")

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
    """서버 헬스 체크 엔드포인트"""
    if not SERVER_IS_READY:
        return JSONResponse(status_code=503, content={"status": "warming_up"})
    return PlainTextResponse("ok")


@app.get("/hb/state")
async def hb_state():
    """현재 서버 상태(활성 연결 수, GPU 정보 등) 반환"""
    if not SERVER_IS_READY:
        return JSONResponse(status_code=503, content={"ok": False})
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
# WebSocket Handlers (비즈니스 로직)
# ──────────────────────────────────────────────────────────────────────────────

async def handle_observe(websocket: WebSocket, client_id: str, data: dict):
    current_image_id = data.get("image_id")
    if not current_image_id: return

    current_image_path = os.path.join(TEMP_IMAGE_PATH, current_image_id)
    if not os.path.exists(current_image_path): return

    state = client_states.get(client_id, {})

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

    analyze_prompt = (
        "!!! SYSTEM COMMAND: ACT AS A SENSITIVE EVENT DETECTOR !!!\n"
        f"{img_context_str}\n"
        f"{task_str}\n"
        "\n"
        "RULES:\n"
        "1. Output valid JSON only. NO COMMENTS.\n"
        "2. **FOCUS ON THE ACTIVE WINDOW**: Check content changes inside the main window (e.g., Video scene change, Character pose, New tab).\n"
        "3. **USE FULL RANGE (6-9)**: Do NOT stick to score 7. If it's cool/visual, give 8. If amazing, give 9.\n"
        "4. **AVOID REPETITION**: Do not use the same summary/reason as the previous turn.\n"
        "\n"
        "SCORING GUIDELINES (Target: [CURRENT]):\n"
        "- 0-3: [Static] Absolutely nothing changed. AFK.\n"
        "- 4-5: [Minor] Mouse movement, slow scrolling, idle animation. (No Speech)\n"
        "- 6-7: [Standard Change] Active window switch, New web page loaded, Character pose changed. (Speak Mildly)\n"
        "- 8-9: [Exciting Update] Visuals became colorful, Game action started, Zoom-in, Video scene highlight. (Speak Excitedly)\n"
        "- 10: [Critical] Game Over, Victory, Error Popup, Huge visual explosion.\n"
        "\n"
        "*** TIP: If the visual is 'Beautiful' or 'Dynamic', give 8 or 9. Don't be shy. ***\n"
        "\n"
        "OUTPUT JSON ONLY:\n"
        "{\n"
        '  "score": Integer(0-10),\n'
        '  "summary": "Specific description of WHAT changed.",\n'
        '  "reason": "Why did you pick this score? (Explain the visual intensity)"\n'
        "}"
    )

    def step1_analyze():
        return generate_llm_response(
            user_input=analyze_prompt,
            recent_conversation=[],
            inputType=InputTypeValue.CHAT,
            emotion="Neutral",
            image_paths=image_paths_for_llm
        )

    try:
        try:
            raw_res_1 = await asyncio.wait_for(asyncio.to_thread(step1_analyze), timeout=8.0)
        except asyncio.TimeoutError:
            print(f"⚠️ Observe Timeout (Infinite Gen blocked). Score set to 0.")
            raw_res_1 = '{"score": 0, "summary": "Timeout", "reason": "System Timeout"}'

        res_json = {}
        json_str = ""
        start_idx = raw_res_1.find('{')

        if start_idx != -1:
            json_str = raw_res_1[start_idx:]
            json_str = json_str.replace("\u2581", " ")
            json_str = re.sub(r'(?<![:"])\/\/.*', '', json_str)
            json_str = re.sub(r'(?<![:"])\#.*', '', json_str)
            json_str = re.sub(r'(?<=[{,])\s*[^"a-zA-Z0-9\s{]+', "", json_str)
            try:
                res_json, _ = json.JSONDecoder().raw_decode(json_str)
            except:
                try:
                    res_json = ast.literal_eval(json_str)
                except:
                    pass

        if not res_json or "score" not in res_json:
            score_match = re.search(r'["\']score["\']\s*:\s*(\d+)', raw_res_1)
            if score_match:
                res_json["score"] = int(score_match.group(1))
            else:
                res_json["score"] = 0

            sum_match = re.search(r'["\']summary["\']\s*:\s*["\'](.*?)["\']', raw_res_1)
            if sum_match:
                res_json["summary"] = sum_match.group(1)
            else:
                res_json["summary"] = "화면의 시각적 정보를 참고하세요."

            res_json["reason"] = "Regex Fallback"

        score = int(res_json.get("score", 0))
        summary = res_json.get("summary", "")
        reason = res_json.get("reason", "")

        next_stored_images = stored_images + [current_image_path]
        images_to_delete = []
        while len(next_stored_images) > 2:
            oldest = next_stored_images.pop(0)
            images_to_delete.append(oldest)

        if count < 3:
            print(f"👀 Observe[{client_id}]: Warm-up ({count}/3). Ignored.")
            client_states[client_id] = {
                "summary": summary,
                "stored_images": next_stored_images,
                "cooldown": 0,
                "last_spoken": last_spoken
            }
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

        if should_speak:
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lucia_persona = server_config.LLM.LLM_CHAT_FORMAT.format(
                recent_conversation="", userEmotion="", now=current_time_str
            )

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
                "As Lucia, say something spontaneous to Melissa based on the CURRENT situation.\n"
                "\n"
                "Output: Just one sentence in Korean. (No JSON)."
            )

            def step2_act():
                return generate_llm_response(
                    user_input=actor_prompt,
                    recent_conversation=[],
                    inputType=InputTypeValue.CHAT,
                    emotion="Neutral",
                    image_paths=[current_image_path]
                )

            try:
                raw_res_2 = await asyncio.wait_for(asyncio.to_thread(step2_act), timeout=10.0)
                message_to_send = raw_res_2.replace('"', '').replace("Lucia:", "").strip()

                if message_to_send:
                    def process_tts(text):
                        tts_text = re.sub(r'[^\u0000-\uFFFF]', '', text)
                        emo = analyze_emotion(text)

                        if server_config.TTS.TTS_ENABLE:
                            wav_bytes = text_to_speech(tts_text, emo)
                        else:
                            wav_bytes = None

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

        client_states[client_id] = {
            "summary": summary if summary else last_summary,
            "stored_images": next_stored_images,
            "cooldown": cooldown,
            "last_spoken": message_to_send if should_speak and message_to_send else last_spoken
        }

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


async def handle_chat(websocket: WebSocket, data: dict):
    text = data.get("text", "")
    is_emotion = bool(data.get("emotion"))
    img_ids = data.get("image_ids") or []
    if data.get("image_id"): img_ids.append(data.get("image_id"))

    user_info = {
        "name": data.get("user_name"),
        "gender": data.get("user_gender"),
        "birth_date": data.get("user_birth_date")
    }

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
            user_info=user_info
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

        # 🚀 클라이언트가 이미 끊겼을 경우 발생하는 에러를 캡처
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
# WebSocket Endpoint (메인 진입점)
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("⏳ Waiting for internal services to be ready before accepting WS...")
    # 연결 수락 전 내부 서비스 상태 확인 (최대 60초 대기)
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

            if op == "chat":
                await handle_chat(websocket, data)
                continue

            if op == "feedback":
                await handle_feedback(websocket, data)
                continue

    finally:
        manager.disconnect(client_id)