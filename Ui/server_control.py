# ──────────────────────────────────────────────────────────────────────────────
# Ui/server_control.py
# 백그라운드 서버 프로세스(Main, LLM, TTS)를 관리하고 상태를 모니터링하는 클래스입니다.
# ──────────────────────────────────────────────────────────────────────────────
import os
import platform
import sys
import socket
import json
import logging
import time
import signal
from typing import Optional, Set
from datetime import datetime
import pytz

from PyQt5.QtCore import QObject, QProcess, QTimer, pyqtSignal, QUrl, QProcessEnvironment, QByteArray
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PyQt5.QtWidgets import QPushButton, QTextBrowser

# psutil (프로세스 관리용, 선택적)
try:
    import psutil
except ImportError:
    psutil = None

# Core 접근 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Core.server_config as server_config

# 로거 설정
logger = logging.getLogger("system.server_control")
logger_llm = logging.getLogger("llm.server_control")

# 로그 필터링 키워드 (불필요한 로그 제외)
NOISY_KEYWORDS = (
    "GET /health",
    "GET /hb/state",
    "200 OK",
    "🤖 생성 완료:",
    "📝 Prompt Tokens:",
)

# 포트 설정 (server_config에서 로드)
LLAMA_HOST = server_config.PORTS.LLAMA_HOST
LLAMA_PORT = server_config.PORTS.LLAMA_PORT
TTS_HOST = server_config.PORTS.TTS_HOST
TTS_PORT = server_config.PORTS.TTS_PORT
AUDIO_SERVER_HOST = server_config.PORTS.AUDIO_SERVER_HOST
AUDIO_SERVER_PORT = server_config.PORTS.AUDIO_SERVER_PORT
MAIN_SERVER_PORT = server_config.PORTS.MAIN_SERVER_PORT

# 관리자 토큰 (선택적)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def init_server_settings() -> None:
    """
    MySQL DB에서 서버 설정을 로드하여 server_config를 갱신합니다.
    서버 시작 시 호출됩니다.
    """
    from Core.sql import MySQLManager  # 지연 임포트 (순환 참조 방지)
    db = MySQLManager()
    try:
        settings = db.fetch_server_settings()
        # 통신 로그 관련 설정 반영
        server_config.LLM.COMMU_LOG_TIME = bool(settings.get("commu_log_time", 0))
        server_config.LLM.COMMU_LOG_INTERVAL = int(settings.get("commu_log_interval", 10))

        print(
            f"✅ 서버 설정 로드: commu_log_time={server_config.LLM.COMMU_LOG_TIME}, "
            f"commu_log_interval={server_config.LLM.COMMU_LOG_INTERVAL}"
        )
    except Exception as e:
        logger.error(f"서버 설정 로드 실패: {e}")
    finally:
        db.close()


class ServerControl(QObject):
    """
    서버 프로세스(Main, LLM, TTS)의 생명주기를 관리하고 상태를 UI에 반영하는 클래스입니다.
    """
    statusChanged = pyqtSignal(str) # 상태 변경 시그널

    def __init__(
            self,
            workdir: str,
            module_spec: str,
            host: str = "127.0.0.1",
            port: int = MAIN_SERVER_PORT,
            venv_dir: Optional[str] = None,
            python_exe: Optional[str] = None,
            parent=None,
    ):
        super().__init__(parent)
        self.workdir = os.path.abspath(workdir)
        self.module_spec = module_spec
        self.host = host
        self.port = port
        self.venv_dir = venv_dir
        self.python_exe = python_exe
        self.parent = parent

        # ─────────────────────────────────────────────
        # 프로세스 객체 초기화
        # ─────────────────────────────────────────────
        self.proc = QProcess(self)      # 메인 서버 (FastAPI/Uvicorn)
        self.proc_llm = QProcess(self)  # LLM 서버 (ExLlamaV3)
        self.proc_tts = QProcess(self)  # TTS 서버 (GPT-SoVITS)

        # 프로세스 시그널 연결
        for p in (self.proc, self.proc_llm, self.proc_tts):
            p.setProcessChannelMode(QProcess.MergedChannels) # stdout/stderr 병합
            p.readyReadStandardOutput.connect(self._on_output)
            p.readyReadStandardError.connect(self._on_output)
            p.finished.connect(self._on_finished)
            p.errorOccurred.connect(self._on_error)

        self.proc.started.connect(self._on_started)
        self.proc_llm.started.connect(self._on_llm_started)
        self.proc_tts.started.connect(self._on_tts_started)

        # ─────────────────────────────────────────────
        # 네트워크 및 상태 모니터링 초기화
        # ─────────────────────────────────────────────
        self.net = QNetworkAccessManager(self)
        probe_host = self.host if self.host not in ("0.0.0.0", "::", "") else "127.0.0.1"
        self._url_health = QUrl(f"http://{probe_host}:{self.port}/health")
        self._url_hb = QUrl(f"http://{probe_host}:{self.port}/hb/state")
        self._url_restart = QUrl(f"http://{probe_host}:{self.port}/restart")

        # UI 컨트롤 참조 저장소
        self._controls = {
            "startstop": None, 
            "restart": None, 
            "status_in": None, 
            "status_out": None, 
            "status_llm": None,
            "status_tts": None,
            "status_audio": None
        }
        
        self._status = "stopped"
        self._timer = None
        self._pending_restart = False
        self._intentional_stop = False # 의도적인 정지 여부 플래그

        # 초기 상태 전파
        self.statusChanged.connect(self._update_controls)
        self.statusChanged.emit(self._status)

    def bind_controls(self, startstop_btn=None, restart_btn=None, 
                      status_in_widget=None, status_out_widget=None, 
                      status_llm_widget=None, status_tts_widget=None, 
                      status_audio_widget=None):
        """UI 위젯들을 컨트롤러에 연결합니다."""
        self._controls["startstop"] = startstop_btn
        self._controls["restart"] = restart_btn
        self._controls["status_in"] = status_in_widget
        self._controls["status_out"] = status_out_widget
        self._controls["status_llm"] = status_llm_widget
        self._controls["status_tts"] = status_tts_widget
        self._controls["status_audio"] = status_audio_widget

        if startstop_btn:
            startstop_btn.clicked.connect(self._on_click_startstop)
        if restart_btn:
            restart_btn.clicked.connect(self._on_click_restart)

        self._update_controls(self._status)
        
        # 초기 상태 표시 (정지됨)
        self._update_status_llm("stopped")
        self._update_status_tts("stopped")
        self._update_status_audio("stopped")

        # [UI 연동] 이제 프리셋/커스텀 구분이 없으므로, LLMAdvancedButton은 무조건 활성화합니다.
        if self.parent and hasattr(self.parent, "LLMAdvancedButton"):
            self.parent.LLMAdvancedButton.setEnabled(True)

    def start(self):
        """모든 서버 프로세스를 시작합니다."""
        if self.proc.state() != QProcess.NotRunning:
            logger.info("이미 메인 서버가 실행 중입니다.")
            return

        # 1. 포트 점유 확인 및 정리
        ports_to_check = [self.port, AUDIO_SERVER_PORT]
        for p in ports_to_check:
            if not self._is_port_free(self.host, p):
                logger.warning(f"포트 {p}가 사용 중입니다. 프로세스 종료를 시도합니다...")
                if not self._force_free_port(p, retries=3):
                    logger.error(f"포트 {p}를 확보할 수 없습니다. 시작을 중단합니다.")
                    self._mark_error_hint()
                    self._set_status("오류 발생")
                    return

        self._intentional_stop = False
        logger.info(f"서버 시작 (WorkDir: {self.workdir})")

        # 2. 환경 변수 설정
        env = self._prepare_environment()
        env_llm = QProcessEnvironment(env)
        env_tts = QProcessEnvironment(env)

        # GPU 설정 적용
        if server_config.LLM.GPU_INDEX:
            env_llm.insert("CUDA_VISIBLE_DEVICES", str(server_config.LLM.GPU_INDEX))
        if hasattr(server_config.TTS, "GPU_TTS"):
            env_tts.insert("CUDA_VISIBLE_DEVICES", str(server_config.TTS.GPU_TTS))

        pyexe = self._pick_python_exe()
        project_root = os.path.abspath(os.path.join(self.workdir, ".."))

        # ─────────────────────────────────────────────
        # 3. LLM 서버 실행 (ExLlamaV3)
        # ─────────────────────────────────────────────
        # ─────────────────────────────────────────────
        # 3. LLM 서버 실행 (ExLlamaV3)
        # ─────────────────────────────────────────────
        llama_model_path = f"{server_config.LLM.LOCATION}/{server_config.LLM.LOCATION_MODEL}"
        llm_script = os.path.join(project_root, "Core", "ExLlamaV3", "server.py")
        
        # Context Size 보정 (256의 배수)
        try:
            context_val = int(server_config.LLM.CONTEXT)
        except (ValueError, TypeError):
            context_val = 4096
        if context_val % 256 != 0:
            new_context = ((context_val + 255) // 256) * 256
            logger.warning(f"Context size {context_val} -> {new_context} (256 배수 보정)")
            context_val = new_context

        # 설정된 파라미터 무조건 적용 (프리셋 무시)
        gen_params = {
            "temperature": server_config.LLM.TEMPERATURE,
            "top_k": server_config.LLM.TOP_K,
            "top_p": server_config.LLM.TOP_P,
            "min_p": server_config.LLM.MIN_P,
            "repetition_penalty": server_config.LLM.REPETITION_PENALTY,
            "presence_penalty": server_config.LLM.PRESENCE_PENALTY,
            "frequency_penalty": server_config.LLM.FREQUENCY_PENALTY,
        }

        # LLM 실행 인자 구성 (-mode 인자 완전 삭제)
        llama_args = [
            llm_script,
            "-m", llama_model_path,
            "--host", LLAMA_HOST,
            "--port", str(LLAMA_PORT),
            "--cache_size", str(context_val),
            "--gpu_index", str(server_config.LLM.GPU_INDEX),
            "--temperature", str(gen_params["temperature"]),
            "--top_k", str(gen_params["top_k"]),
            "--top_p", str(gen_params["top_p"]),
            "--min_p", str(gen_params["min_p"]),
            "--repetition_penalty", str(gen_params["repetition_penalty"]),
            "--presence_penalty", str(gen_params["presence_penalty"]),
            "--frequency_penalty", str(gen_params["frequency_penalty"]),
        ]

        if server_config.LLM.GPU_SPLIT:
            llama_args.extend(["--gpu_split", str(server_config.LLM.GPU_SPLIT)])
        if str(server_config.LLM.CACHE_QUANT) != "0":
            llama_args.extend(["--cache_quant", str(server_config.LLM.CACHE_QUANT)])
        
        # Tensor Parallelism (UI 체크박스 확인)
        if self.parent and hasattr(self.parent, "ParallelCheckBox") and self.parent.ParallelCheckBox.isChecked():
            llama_args.append("--tensor_parallel")
            logger.info("Tensor Parallelism 활성화됨")

        self.proc_llm.setWorkingDirectory(project_root)
        self.proc_llm.setProcessEnvironment(env_llm)
        logger.info(f"LLM 시작: {' '.join(llama_args)}")
        self._update_status_llm("booting")
        self.proc_llm.start(pyexe, llama_args)

        # ─────────────────────────────────────────────
        # 4. TTS 서버 실행 (GPT-SoVITS)
        # ─────────────────────────────────────────────
        tts_script = os.path.join(project_root, "Core", "GptSoVits", "api_v2.py")
        tts_args = [tts_script, "-a", TTS_HOST, "-p", str(TTS_PORT)]
        tts_cwd = os.path.join(project_root, "Core", "GptSoVits")
        
        self.proc_tts.setWorkingDirectory(tts_cwd)
        self.proc_tts.setProcessEnvironment(env_tts)
        logger.info(f"TTS 시작: {' '.join(tts_args)}")
        self._update_status_tts("booting")
        self.proc_tts.start(pyexe, tts_args)

        # ─────────────────────────────────────────────
        # 5. 메인 서버 실행 (FastAPI)
        # ─────────────────────────────────────────────
        program, args = self._best_server_cmd_program_args(pyexe)
        self.proc.setWorkingDirectory(self.workdir)
        self.proc.setProcessEnvironment(env)
        logger.info(f"Main 시작: {program} {' '.join(args)}")
        self._set_status("running(process)")
        self.proc.start(program, args)
        
        # 헬스 체크 시작
        self._poll_health_ready()

    def stop(self, timeout_ms=2000):
        """모든 서버 프로세스를 종료합니다."""
        self._intentional_stop = True
        
        self._stop_process(self.proc, "Main", self.port, timeout_ms)
        self._stop_process(self.proc_llm, "LLM", LLAMA_PORT, timeout_ms)
        self._stop_process(self.proc_tts, "TTS", TTS_PORT, timeout_ms)

        self._set_status("stopped")
        self._update_status_in_widget("stopped")
        self._update_status_out_widget("stopped")
        self._update_status_llm("stopped")
        self._update_status_tts("stopped")
        self._update_status_audio("stopped")
        logger.info("모든 서버가 종료되었습니다.")

    def restart(self, delay_ms=400):
        """서버를 재시작합니다 (Stop -> Start)."""
        logger.info("서버 재시작 요청됨.")
        self._set_status("restarting")
        self._pending_restart = True
        self._fallback_local_restart(delay_ms)

    def status(self) -> str:
        return self._status

    def refresh_status(self):
        """현재 상태를 강제로 갱신합니다."""
        if self.proc.state() == QProcess.NotRunning:
            self._set_status("stopped")
            self._update_status_in_widget("stopped")
            self._update_status_out_widget("stopped")
            self._update_status_llm("stopped")
            self._update_status_tts("stopped")
            self._update_status_audio("stopped")
            return
        self._probe_health_once()
        self._probe_hb_once()

    # ─────────────────────────────────────────────────────────────
    # 내부 헬퍼 메서드
    # ─────────────────────────────────────────────────────────────
    def _prepare_environment(self) -> QProcessEnvironment:
        """공통 환경 변수를 설정합니다."""
        env = QProcessEnvironment.systemEnvironment()
        
        # GPU 순서 고정 (PCI Bus ID 기준)
        env.insert("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        # Python 경로 보정
        pyexe = self._pick_python_exe()
        pyroot = os.path.dirname(pyexe)
        path_parts = [env.value("PATH", "")]
        for sub in ("Library\\bin", "Scripts"):
            p = os.path.join(pyroot, sub)
            if os.path.isdir(p):
                path_parts.insert(0, p)
        path_parts.insert(0, pyroot)
        env.insert("PATH", os.pathsep.join(path_parts))

        # DB 설정 주입 (config.json)
        try:
            with open("./config.json", "r", encoding="utf-8") as f:
                db_config = json.load(f)
            env.insert("MYSQL_HOST", str(db_config.get("host") or db_config.get("MYSQL_HOST")))
            env.insert("MYSQL_DATABASE", str(db_config.get("database") or db_config.get("MYSQL_DATABASE")))
            env.insert("MYSQL_USER", str(db_config.get("user") or db_config.get("MYSQL_USER")))
            env.insert("MYSQL_PASSWORD", str(db_config.get("password") or db_config.get("MYSQL_PASSWORD")))
            env.insert("MYSQL_PORT", str(db_config.get("port") or db_config.get("MYSQL_PORT") or 3306))
        except Exception as e:
            logger.warning(f"config.json 로드 실패: {e}")

        # 내부 서버 URL 환경변수
        env.insert("LLAMA_SERVER_URL", f"http://{LLAMA_HOST}:{LLAMA_PORT}")
        env.insert("LLAMA_SERVER_MODEL", server_config.LLM.LOCATION_MODEL)
        env.insert("TTS_SERVER_URL", f"http://{TTS_HOST}:{TTS_PORT}")
        
        return env

    def _stop_process(self, process: QProcess, name: str, port: int, timeout: int):
        """단일 프로세스를 종료하고 포트를 정리합니다."""
        if process.state() != QProcess.NotRunning:
            logger.info(f"{name} 서버 종료 중...")
            pid = process.processId()
            
            if pid > 0:
                self._terminate_tree(pid)
            
            process.terminate()
            if not process.waitForFinished(timeout):
                logger.warning(f"{name} 서버 강제 종료 시도")
                process.kill()
                process.waitForFinished(1000)

            self._force_free_port(port, retries=2)

    def _on_click_startstop(self):
        if self.proc.state() == QProcess.NotRunning:
            self.start()
        else:
            self.stop()

    def _on_click_restart(self):
        self.restart()

    def _restart_after_stop(self):
        self.stop()
        logger.info("재시작을 위해 모든 서버를 다시 시작합니다.")
        QTimer.singleShot(1000, self.start)
        self._pending_restart = False

    def _set_status(self, s: str):
        if s != self._status:
            logger.info(f"상태 변경: {self._status} -> {s}")
            self._status = s
            self.statusChanged.emit(s)

    # ─────────────────────────────────────────────────────────────
    # 프로세스 이벤트 핸들러
    # ─────────────────────────────────────────────────────────────
    def _on_started(self):
        logger.info("[Main] 프로세스 시작됨")
        self._set_status("running(process)")
        init_server_settings()

    def _on_llm_started(self):
        logger.info("[LLM] 프로세스 시작됨")
        self._update_status_llm("booting")

    def _on_tts_started(self):
        logger.info("[TTS] 프로세스 시작됨")
        self._update_status_tts("booting")

    def _on_finished(self, code, status):
        sender = self.sender()
        prefix_map = {self.proc: "[Main]", self.proc_llm: "[LLM]", self.proc_tts: "[TTS]"}
        prefix = prefix_map.get(sender, "[Server]")
        
        logger.info(f"{prefix} 종료됨 (code={code}, status={status})")

        if not self._intentional_stop:
            # 비정상 종료 처리
            if sender == self.proc_llm:
                logger.error("❌ LLM 서버 비정상 종료")
                self._update_status_llm("error")
                self._handle_unexpected_exit()
            elif sender == self.proc_tts:
                logger.error("❌ TTS 서버 비정상 종료")
                self._update_status_tts("error")
                self._handle_unexpected_exit()
            elif sender == self.proc:
                if self._pending_restart:
                    self._set_status("restarting")
                    self._restart_after_stop()
                else:
                    self._set_status("stopped")
                    self._update_status_in_widget("stopped")
                    self._update_status_out_widget("stopped")
                    self._update_status_audio("stopped")
        else:
            # 정상 종료 처리
            if sender == self.proc:
                self._set_status("stopped")
                self._update_status_in_widget("stopped")
                self._update_status_out_widget("stopped")
                self._update_status_audio("stopped")
            elif sender == self.proc_llm:
                self._update_status_llm("stopped")
            elif sender == self.proc_tts:
                self._update_status_tts("stopped")

    def _handle_unexpected_exit(self):
        """서브 프로세스 비정상 종료 시 전체 중단"""
        self._mark_error_hint()
        self._set_status("오류 발생")
        self.stop()

    def _on_error(self, err):
        if self._intentional_stop:
            return

        sender = self.sender()
        prefix_map = {self.proc: "[Main]", self.proc_llm: "[LLM]", self.proc_tts: "[TTS]"}
        prefix = prefix_map.get(sender, "[Server]")
        
        logger.error(f"{prefix} 프로세스 오류 발생: {err}")
        self._mark_error_hint()
        self._set_status("오류 발생")
        self.stop()

    def _on_output(self):
        """프로세스 표준 출력을 로그로 기록합니다."""
        sender = self.sender()
        if not isinstance(sender, QProcess): return

        data = bytes(sender.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if not data.strip(): return

        for line in data.splitlines():
            if not line.strip() or any(k in line for k in NOISY_KEYWORDS):
                continue
            
            tz = pytz.timezone("Asia/Seoul")
            now = datetime.now(tz).strftime("[%Y-%m-%d %H:%M:%S] ")

            prefix_map = {self.proc: "[main]", self.proc_llm: "[llm-proc]", self.proc_tts: "[tts-proc]"}
            prefix = prefix_map.get(sender, "[server]")

            # LLM 관련 로그는 별도 로거 사용
            if "🔸" in line or "🔹" in line or line.startswith(("[llm]", "[LLM]")):
                logger_llm.info(now + line)
            else:
                logger.info(f"{now}{prefix} {line}")

    # ─────────────────────────────────────────────────────────────
    # 헬스 체크 및 모니터링
    # ─────────────────────────────────────────────────────────────
    def _poll_health_ready(self, interval_ms=1000):
        if self._timer: self._timer.stop()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_tick)
        self._timer.start(interval_ms)

    def _poll_tick(self):
        if self.proc.state() == QProcess.NotRunning:
            if self._timer: self._timer.stop()
            self._set_status("stopped")
            self._update_status_in_widget("stopped")
            self._update_status_out_widget("stopped")
            self._update_status_audio("stopped")
            return
        
        # 메인 서버 체크
        self._probe_health_once()
        self._probe_hb_once()

        # 서브 서버 체크 (소켓 연결)
        self._check_sub_server_status(self.proc_llm, LLAMA_HOST, LLAMA_PORT, self._update_status_llm)
        self._check_sub_server_status(self.proc_tts, TTS_HOST, TTS_PORT, self._update_status_tts)
        
        # 오디오 서버 체크
        if self.proc.state() != QProcess.NotRunning:
            if self._can_connect(AUDIO_SERVER_HOST, AUDIO_SERVER_PORT):
                self._update_status_audio("running")
            else:
                self._update_status_audio("booting")
        else:
            self._update_status_audio("stopped")

    def _check_sub_server_status(self, proc, host, port, update_func):
        if proc.state() != QProcess.NotRunning:
            if self._can_connect(host, port):
                update_func("running")
            else:
                update_func("booting")

    def _can_connect(self, host, port) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    def _probe_health_once(self):
        reply = self.net.get(QNetworkRequest(self._url_health))
        reply.finished.connect(lambda r=reply: self._on_health_done(r))

    def _probe_hb_once(self):
        reply = self.net.get(QNetworkRequest(self._url_hb))
        reply.finished.connect(lambda r=reply: self._on_hb_done(r))

    def _on_health_done(self, reply):
        ok = (reply.error() == reply.NoError and reply.attribute(QNetworkRequest.HttpStatusCodeAttribute) == 200)
        reply.deleteLater()

        if self.proc.state() == QProcess.NotRunning:
            self._set_status("stopped")
            self._update_status_in_widget("stopped")
            return

        if ok:
            self._update_status_in_widget("running(ready)")
            if self._status in ("running(process)", "running(booting)", "restarting"):
                self._set_status("running(process)")
        else:
            self._update_status_in_widget("running(booting)")
            if self._status != "restarting":
                self._set_status("running(booting)")

    def _on_hb_done(self, reply):
        ok = False
        try:
            if reply.error() == reply.NoError:
                data = json.loads(bytes(reply.readAll()))
                ok = (reply.attribute(QNetworkRequest.HttpStatusCodeAttribute) == 200 and int(data.get("active_count", 0)) >= 1)
        except Exception:
            ok = False
        finally:
            reply.deleteLater()

        if self.proc.state() == QProcess.NotRunning:
            self._set_status("stopped")
            self._update_status_out_widget("stopped")
            return

        status_str = "running(ready)" if ok else "running(booting)"
        self._update_status_out_widget(status_str)
        if ok and self._status == "running(booting)":
            self._set_status("running(process)")

    # ─────────────────────────────────────────────────────────────
    # 유틸리티 메서드
    # ─────────────────────────────────────────────────────────────
    def _fallback_local_restart(self, delay_ms: int):
        self.stop()
        QTimer.singleShot(delay_ms, self.start)

    def _is_port_free(self, host, port) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return True
            except OSError:
                return False

    def _pick_python_exe(self) -> str:
        if self.python_exe and os.path.exists(self.python_exe):
            return self.python_exe
        if self.venv_dir:
            sub = "Scripts" if platform.system() == "Windows" else "bin"
            exe = "python.exe" if platform.system() == "Windows" else "python"
            cand = os.path.join(self.venv_dir, sub, exe)
            if os.path.exists(cand):
                return cand
        return sys.executable

    def _best_server_cmd_program_args(self, pyexe: str):
        program = pyexe
        workers = 1 # uvicorn 워커는 1로 고정 (내부 Executor 사용)
        args = ["-m", "uvicorn", self.module_spec, "--host", self.host, "--port", str(self.port), "--workers", str(workers)]
        return program, args

    def _pids_listening_on(self, port: int) -> Set[int]:
        if not psutil: return set()
        pids: Set[int] = set()
        try:
            for c in psutil.net_connections(kind='inet'):
                if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                    pids.add(c.pid)
        except psutil.AccessDenied:
            pass
        return pids

    def _terminate_tree(self, pid: int, timeout: float = 2.0):
        """psutil을 사용하여 프로세스 트리 전체를 종료합니다."""
        if not psutil:
            logger.warning("psutil 미설치: 프로세스 트리 종료 불가")
            return
        try:
            proc = psutil.Process(pid)
            children = proc.children(recursive=True)
            for p in children:
                try: p.terminate()
                except psutil.NoSuchProcess: pass
            
            proc.terminate()
            gone, alive = psutil.wait_procs(children + [proc], timeout=timeout)
            
            for p in alive:
                try: p.kill()
                except psutil.NoSuchProcess: pass
            logger.info(f"PID {pid} 프로세스 트리 종료 완료")

        except psutil.NoSuchProcess:
            pass

    def _force_free_port(self, port: int, retries: int = 2) -> bool:
        if not psutil: return False
        for i in range(retries):
            pids = self._pids_listening_on(port)
            if not pids: return True
            logger.info(f"포트 {port} 점유 PID 종료 시도 ({i+1}/{retries}): {pids}")
            for pid in pids:
                self._terminate_tree(pid)
            time.sleep(1.0)
        return not self._pids_listening_on(port)

    # ─────────────────────────────────────────────────────────────
    # UI 업데이트 메서드
    # ─────────────────────────────────────────────────────────────
    def _update_controls(self, status: str):
        startstop: QPushButton = self._controls["startstop"]
        restart: QPushButton = self._controls["restart"]

        self._update_status_in_widget(status)
        self._update_status_out_widget("running(booting)" if "running" in status else status)

        mapping = {
            "stopped": {"startstop": ("시작", True), "restart": False},
            "running(process)": {"startstop": ("중지", True), "restart": True},
            "running(booting)": {"startstop": ("중지", True), "restart": False},
            "running(ready)": {"startstop": ("중지", True), "restart": True},
            "restarting": {"startstop": ("재시작중…", False), "restart": False},
            "오류 발생": {"startstop": ("시작", True), "restart": False},
        }
        spec = mapping.get(status, mapping["stopped"])

        if startstop:
            startstop.setText(spec["startstop"][0])
            startstop.setEnabled(spec["startstop"][1])
        if restart:
            restart.setEnabled(spec["restart"])

    def _update_status_widget(self, widget, text_map, color_map, status, prefix, error_hint=False):
        if not widget: return
        
        color = "#d9534f" if error_hint and status != "오류 발생" else color_map.get(status, "#333")
        text = text_map.get(status, status)
        full_text = f"{prefix} {text}".strip()

        if isinstance(widget, QTextBrowser):
            line = f'<span style="color:{color}">{full_text}</span>'
            widget.setHtml(line)
        else:
            widget.setText(full_text)
            widget.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _update_status_in_widget(self, status: str, error_hint: bool=False):
        text_map = {
            "stopped": "", "running(process)": "",
            "running(booting)": "", "running(ready)": "",
            "restarting": "", "오류 발생": "",
        }
        color_map = {
            "stopped": "#d9534f", "running(process)": "#0275d8", "running(booting)": "#f0ad4e",
            "running(ready)": "#5cb85c", "restarting": "#5bc0de", "오류 발생": "#d9534f",
        }
        self._update_status_widget(self._controls["status_in"], text_map, color_map, status, "■", error_hint)

    def _update_status_out_widget(self, status: str, error_hint: bool=False):
        text_map = {
            "stopped": "", "running(booting)": "",
            "running(ready)": "", "오류 발생": "",
            "running(process)": "", "restarting": "",
        }
        color_map = {
            "stopped": "#d9534f", "running(booting)": "#f0ad4e", "running(ready)": "#5cb85c",
            "오류 발생": "#d9534f", "running(process)": "#0275d8", "restarting": "#5bc0de",
        }
        self._update_status_widget(self._controls["status_out"], text_map, color_map, status, "■", error_hint)

    def _update_status_llm(self, status: str):
        text_map = {"stopped": "", "running": "", "booting": "", "error": ""}
        color_map = {"stopped": "#d9534f", "running": "#5cb85c", "booting": "#f0ad4e", "error": "#d9534f"}
        self._update_status_widget(self._controls["status_llm"], text_map, color_map, status, "■")

    def _update_status_tts(self, status: str):
        text_map = {"stopped": "", "running": "", "booting": "", "error": ""}
        color_map = {"stopped": "#d9534f", "running": "#5cb85c", "booting": "#f0ad4e", "error": "#d9534f"}
        self._update_status_widget(self._controls["status_tts"], text_map, color_map, status, "■")

    def _update_status_audio(self, status: str):
        text_map = {"stopped": "", "running": "", "booting": "", "error": ""}
        color_map = {"stopped": "#d9534f", "running": "#5cb85c", "booting": "#f0ad4e", "error": "#d9534f"}
        self._update_status_widget(self._controls["status_audio"], text_map, color_map, status, "■")

    def _mark_error_hint(self):
        self._update_status_in_widget(self._status, error_hint=True)
        self._update_status_out_widget(self._status, error_hint=True)
        self._update_status_llm("error")
        self._update_status_tts("error")
        self._update_status_audio("error")
