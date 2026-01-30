import sys
import os
import atexit
import logging
import ctypes  # <--- [중요] 윈도우 작업표시줄 아이콘 설정을 위해 추가

from PyQt5 import QtCore, uic
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDialog, QTextBrowser, QWidget
from PyQt5.QtNetwork import QNetworkAccessManager

# 프로젝트 내부 모듈 임포트 (사용자 환경에 맞게 유지)
from Ui.qt_log_handler import QtLogHandler
import Core.server_config as server_config
from Ui.controllers import NavController, SystemController


# (참고) MyApp 클래스는 run() 함수에서 사용되지 않으므로
# 실제 실행되는 MainWindow에 아이콘 설정을 해야 합니다.

class MainWindow(QDialog):
    """
    메인 윈도우 클래스.
    UI 초기화, 로거 설정, 컨트롤러 연결 및 애플리케이션 수명 주기를 관리합니다.
    """

    def __init__(self):
        super().__init__(parent=None)

        # 윈도우 플래그 설정 (최소화 버튼 포함, 기본 프레임)
        self.setWindowFlags(
            QtCore.Qt.Window |
            QtCore.Qt.CustomizeWindowHint |
            QtCore.Qt.WindowMinimizeButtonHint |
            QtCore.Qt.WindowCloseButtonHint  # 닫기 버튼 명시
        )

        # .ui 파일 로드
        uic.loadUi("Ui/untitled.ui", self)

        # ─────────────────────────────────────────────
        # 1. 공용 리소스 및 경로 설정
        # ─────────────────────────────────────────────
        # 현재 스크립트(ui_main.py)가 위치한 디렉토리를 기준으로 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # [수정됨] 아이콘 설정 부분 -----------------------------------------
        # 이미지 파일이 스크립트와 같은 폴더에 'sample.png'로 있다고 가정
        icon_path = os.path.join(script_dir, 'Ui/sample.png')
        self.setWindowIcon(QIcon(icon_path))
        # ------------------------------------------------------------------

        # 모델 다운로드 경로 설정
        self.save_dir = os.path.join(script_dir, "Models")
        os.makedirs(self.save_dir, exist_ok=True)

        # 전역 설정에 다운로드 경로 고정
        server_config.DOWNLOAD_DIR = self.save_dir

        # QNetworkAccessManager 초기화
        self.nam = QNetworkAccessManager(self)

        # ─────────────────────────────────────────────
        # 2. 로깅 시스템 초기화
        # ─────────────────────────────────────────────
        self._init_logging()

        # ─────────────────────────────────────────────
        # 3. 컨트롤러 초기화
        # ─────────────────────────────────────────────
        self.nav = NavController(self)
        self.sys = SystemController(self)

        # 초기화 완료 로그
        self.logger.info(f"MainWindow initialized (DOWNLOAD_DIR={server_config.DOWNLOAD_DIR})")
        self.llm_logger.info("LLM logger initialized")

    def _init_logging(self):
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)

        fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", "%H:%M:%S")

        # System 로거
        self.logger = logging.getLogger("system")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not any(isinstance(h, QtLogHandler) for h in self.logger.handlers):
            qt_handler = QtLogHandler()
            qt_handler.setFormatter(fmt)
            self.logger.addHandler(qt_handler)
            qt_handler.emitter.message.connect(self._on_log_message)

        # LLM 로거
        self.llm_logger = logging.getLogger("llm")
        self.llm_logger.setLevel(logging.INFO)
        self.llm_logger.propagate = False

        if not any(isinstance(h, QtLogHandler) for h in self.llm_logger.handlers):
            llm_handler = QtLogHandler()
            llm_handler.setFormatter(fmt)
            self.llm_logger.addHandler(llm_handler)
            llm_handler.emitter.message.connect(self._on_llm_log_message)

        self._system_browser: QTextBrowser = getattr(self, "SystemTextBrowser", None)
        self._llm_browser: QTextBrowser = getattr(self, "LLMTextBrowser", None)

    # ─────────────────────────────────────────────
    # 로그 출력 슬롯 (Slots)
    # ─────────────────────────────────────────────
    def _on_log_message(self, msg: str, levelno: int):
        self._append_to_browser(self._system_browser, msg, levelno)

    def _on_llm_log_message(self, msg: str, levelno: int):
        self._append_to_browser(self._llm_browser, msg, levelno)

    def _append_to_browser(self, browser: QTextBrowser, msg: str, levelno: int):
        if not isinstance(browser, QTextBrowser):
            print(msg)
            return

        style = "color: #212121;"

        if levelno >= logging.CRITICAL:
            style = "color: #B71C1C; font-weight: bold; text-decoration: underline;"
        elif levelno >= logging.ERROR:
            style = "color: #D32F2F; font-weight: bold;"
        elif levelno >= logging.WARNING:
            style = "color: #E65100; font-weight: bold;"
        elif levelno == logging.DEBUG:
            style = "color: #1565C0;"

        browser.append(f'<span style="{style}">{msg}</span>')

    # ─────────────────────────────────────────────
    # 종료 처리
    # ─────────────────────────────────────────────
    def graceful_shutdown(self):
        if getattr(self, "_shutdown_called", False):
            return
        self._shutdown_called = True

        print("종료 절차를 시작합니다...")
        try:
            if not hasattr(self, "sys"):
                return

            ctrl = getattr(self.sys, "ctrl", None)
            if ctrl:
                try:
                    timer = getattr(ctrl, "_timer", None)
                    if timer:
                        timer.stop()
                except RuntimeError:
                    pass

                ctrl._pending_restart = False

                try:
                    if ctrl.proc.state() != ctrl.proc.NotRunning:
                        ctrl.stop()
                except RuntimeError:
                    pass
        except Exception as e:
            print(f"종료 처리 중 예외 발생: {e}")

    def closeEvent(self, event, **kwargs):
        self.graceful_shutdown()
        event.accept()


def run():
    """
    애플리케이션 진입점 함수.
    """
    # [수정됨] 윈도우 작업표시줄 아이콘 분리 설정 -------------------------
    # 이 코드가 없으면 작업표시줄에는 기본 파이썬 로고가 뜹니다.
    try:
        myappid = 'project.lucia.server.hiyori.1.0'  # 임의의 고유 ID
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass  # 리눅스/맥 등에서는 무시
    # ------------------------------------------------------------------

    app = QApplication(sys.argv)

    # [수정됨] 앱 전체 기본 아이콘 설정 (안전장치)
    # MainWindow에서 설정했지만, 앱 전역으로도 설정해두면 팝업창 등에도 적용됩니다.
    # app.setWindowIcon(QIcon('sample.png'))

    w = MainWindow()

    app.aboutToQuit.connect(w.graceful_shutdown)
    atexit.register(lambda: hasattr(w, "graceful_shutdown") and w.graceful_shutdown())

    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()