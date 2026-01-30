# ──────────────────────────────────────────────────────────────────────────────
# Core/runtime_control.py
# 서버 프로세스의 재시작을 제어하는 모듈입니다.
# 실행 환경(Supervisor, Docker, 단독 실행 등)에 따라 적절한 재시작 방식을 선택합니다.
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import signal
import logging

# 로거 설정
logger = logging.getLogger("system.runtime_control")

def under_supervisor() -> bool:
    """
    현재 프로세스가 외부 관리자(Gunicorn, systemd, Docker, Supervisor 등)에 의해
    관리되고 있는지 확인합니다.
    
    Returns:
        bool: 외부 관리자 하에서 실행 중이면 True, 아니면 False
    """
    # 환경 변수를 통해 Supervisor 실행 여부 힌트 확인
    return (
            "GUNICORN_CMD_ARGS" in os.environ or
            "GUNICORN_WORKER_TMP_DIR" in os.environ or
            os.environ.get("RUN_UNDER_SUPERVISOR") == "1"
    )

def restart_worker():
    """
    워커 프로세스를 정상 종료(SIGTERM)합니다.
    외부 관리자가 프로세스 종료를 감지하고 자동으로 새 워커를 생성하도록 유도합니다.
    """
    logger.info("🔄 Supervisor 환경 감지: 워커 프로세스를 종료하여 재시작을 유도합니다.")
    time.sleep(0.5)  # 응답 전송 및 로그 플러시를 위한 여유 시간
    
    try:
        # 현재 프로세스에 종료 시그널 전송
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception as e:
        logger.warning(f"⚠️ SIGTERM 전송 실패 ({e}), 강제 종료(os._exit)를 시도합니다.")
        # SIGTERM이 실패하면 강제 종료
        os._exit(0)

def reexec_self():
    """
    단독 실행 환경(python main.py, uvicorn 직행 등)에서 자기 자신을 재실행(execv)합니다.
    현재 프로세스를 새로운 프로세스로 교체합니다.
    """
    logger.info("🔄 단독 실행 환경 감지: os.execv를 사용하여 프로세스를 재실행합니다.")
    time.sleep(0.5) # 로그 플러시 대기
    
    python = sys.executable
    # 현재 파이썬 인터프리터와 인자를 그대로 사용하여 재실행
    os.execv(python, [python] + sys.argv)

def restart_auto():
    """
    실행 환경을 자동으로 감지하여 최적의 재시작 방식을 수행합니다.
    - Supervisor/Gunicorn 환경: 워커 종료 -> 자동 재생성 유도
    - 단독 실행 환경: os.execv로 프로세스 교체
    """
    if under_supervisor():
        restart_worker()
    else:
        reexec_self()
