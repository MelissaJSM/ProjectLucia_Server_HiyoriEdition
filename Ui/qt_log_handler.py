# ──────────────────────────────────────────────────────────────────────────────
# Ui/qt_log_handler.py
# Python의 logging 모듈 로그를 PyQt 시그널로 변환하여 UI에 전달하는 핸들러입니다.
# ──────────────────────────────────────────────────────────────────────────────
import logging
from PyQt5.QtCore import QObject, pyqtSignal

class QtLogEmitter(QObject):
    """
    로그 메시지를 PyQt 시그널로 방출(emit)하기 위한 헬퍼 클래스입니다.
    QObject를 상속받아야 pyqtSignal을 사용할 수 있습니다.
    """
    # 시그널 정의: (포맷팅된 메시지 문자열, 로그 레벨 정수값)
    message = pyqtSignal(str, int)


class QtLogHandler(logging.Handler):
    """
    logging.Handler를 상속받아 로그 레코드를 Qt 시그널로 전달하는 커스텀 핸들러입니다.
    이 핸들러를 로거에 추가하면, 로그 발생 시 QtLogEmitter를 통해 시그널이 발생합니다.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level=level)
        # 시그널 방출을 담당할 객체 생성
        self.emitter = QtLogEmitter()

    def emit(self, record: logging.LogRecord):
        """
        로그 레코드가 전달될 때 호출되는 메서드입니다.
        메시지를 포맷팅한 후 시그널을 통해 UI 스레드로 전달합니다.
        """
        try:
            # 설정된 포맷터(Formatter)를 사용하여 메시지 포맷팅
            msg = self.format(record)
        except Exception:
            # 포맷팅 실패 시 원본 메시지 사용
            msg = record.getMessage()
            
        # 시그널 방출 (메시지 내용, 로그 레벨)
        # 이 시그널은 UI 스레드의 슬롯(예: MainWindow._on_log_message)에 연결되어야 합니다.
        self.emitter.message.emit(msg, record.levelno)
