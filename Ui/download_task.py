# ──────────────────────────────────────────────────────────────────────────────
# Ui/download_task.py
# UI와 연동하여 Hugging Face 리포지토리를 직접 다운로드하는 클래스입니다.
# ──────────────────────────────────────────────────────────────────────────────
import time
import os
import requests
from typing import Optional, Callable

from PyQt5.QtCore import QTimer, QObject, QCoreApplication
from huggingface_hub import HfApi, hf_hub_url

from Ui.utils import human_bytes, human_time


class DownloadTask(QObject):
    def __init__(self, parent, save_dir: str,
                 key: str, repo_id: str,
                 btn_start, btn_cancel, bar, lbl_percent, lbl_speed, lbl_eta, lbl_status,
                 mirror_repo_id: Optional[str] = None,
                 mirror_state_getter: Optional[Callable[[], bool]] = None):
        super().__init__(parent)

        self.parent = parent
        self.key = key
        self.repo_id = repo_id
        self.mirror_repo_id = mirror_repo_id
        self.mirror_state_getter = mirror_state_getter

        # 모델 종류에 따라 하위 폴더 자동 분류
        sub_folder = ""
        if "Gemma" in key or "Phi4" in key:
            sub_folder = "LLM"
        elif "TTS" in key:
            sub_folder = "TTS"
        elif "Emotion" in key:
            sub_folder = "Emotion"

        # 최종 저장 경로 설정 (예: Models/LLM/Gemma3_27b_2_0)
        base_dir = os.path.join(save_dir, sub_folder) if sub_folder else save_dir
        self.save_dir = os.path.join(base_dir, self.key)

        # UI 위젯 참조
        self.btn_start = btn_start
        self.btn_cancel = btn_cancel
        self.bar = bar
        self.lbl_percent = lbl_percent
        self.lbl_speed = lbl_speed
        self.lbl_eta = lbl_eta
        self.lbl_status = lbl_status

        # 내부 상태 변수
        self.bytes_received = 0
        self.bytes_total = 0
        self.last_bytes = 0
        self.last_time = None
        self.start_time = None
        self._is_cancelled = False
        self._is_running = False

        # 타이머 (0.5초마다 UI 속도 및 ETA 갱신용)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._update_speed_eta)

        self._reset_ui()

        # 버튼 시그널 연결
        self.btn_start.clicked.connect(self.start)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_cancel.setEnabled(False)

    def _reset_ui(self):
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.lbl_percent.setText("")
        self.lbl_speed.setText("")
        self.lbl_eta.setText("")
        self.lbl_status.setText("상태: 대기")

    def start(self):
        """다운로드 작업을 시작합니다."""
        if self._is_running:
            return

        self._is_running = True
        self._is_cancelled = False

        # 미러(Uncensored) 사용 여부 확인
        use_mirror = False
        if callable(self.mirror_state_getter):
            try:
                use_mirror = bool(self.mirror_state_getter())
            except:
                pass

        chosen_repo = self.mirror_repo_id if (use_mirror and self.mirror_repo_id) else self.repo_id

        if use_mirror and not self.mirror_repo_id:
            self.lbl_status.setText("상태: 실패 (미지원 모델)")
            self._is_running = False
            return

        # 허깅페이스 다운로드 실행
        self.download_huggingface_repo(chosen_repo)

    def download_huggingface_repo(self, repo_id: str):
        """Hugging Face API를 통해 전체 폴더를 다운로드하고 UI에 반영합니다."""
        self.lbl_status.setText("상태: 파일 목록 조회 중...")
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        QCoreApplication.processEvents()

        try:
            # 1. 파일 목록 및 전체 크기 조회
            api = HfApi()
            model_info = api.model_info(repo_id=repo_id, files_metadata=True)
            siblings = model_info.siblings

            # 다운로드할 전체 용량 계산
            self.bytes_total = sum((f.size if f.size is not None else 0) for f in siblings)
            self.bytes_received = 0

            os.makedirs(self.save_dir, exist_ok=True)

            self.start_time = time.monotonic()
            self.last_time = self.start_time
            self.timer.start()  # 속도/ETA 계산 타이머 시작

            # 2. 각 파일 순차 다운로드
            for i, file_info in enumerate(siblings):
                if self._is_cancelled:
                    raise InterruptedError("사용자에 의해 취소됨")

                filename = file_info.rfilename
                self.lbl_status.setText(f"상태: [{i + 1}/{len(siblings)}] {os.path.basename(filename)}")
                QCoreApplication.processEvents()

                file_url = hf_hub_url(repo_id=repo_id, filename=filename)
                local_path = os.path.join(self.save_dir, filename)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # 이미 다운로드된 파일 스킵 (크기 비교로 무결성 검증 역할)
                if os.path.exists(local_path) and file_info.size is not None:
                    if os.path.getsize(local_path) == file_info.size:
                        self.bytes_received += file_info.size
                        self._on_progress(self.bytes_received, self.bytes_total)
                        continue

                # HTTP 스트리밍 다운로드 진행
                response = requests.get(file_url, stream=True)
                response.raise_for_status()

                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=32768):  # 32KB 청크
                        if self._is_cancelled:
                            raise InterruptedError("사용자에 의해 취소됨")
                        if chunk:
                            f.write(chunk)
                            self.bytes_received += len(chunk)
                            self._on_progress(self.bytes_received, self.bytes_total)
                            QCoreApplication.processEvents()  # UI 프리징 방지

            # 전체 완료 처리
            self.timer.stop()
            self.lbl_status.setText("상태: 다운로드 완료!")
            self.lbl_speed.setText("속도: -")
            self.lbl_eta.setText("")
            self.bar.setValue(100)

        except InterruptedError:
            self.timer.stop()
            self.lbl_status.setText("상태: 취소됨")
        except Exception as e:
            self.timer.stop()
            self.lbl_status.setText(f"상태: 에러 ({type(e).__name__})")
        finally:
            self.btn_start.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self._is_running = False

    def cancel(self):
        """다운로드 작업을 취소합니다."""
        self._is_cancelled = True
        self.lbl_status.setText("상태: 취소 중...")

    def _on_progress(self, recvd, total):
        """프로그레스 바와 진행률 라벨을 업데이트합니다."""
        if total > 0:
            pct = int((recvd / total) * 100)
            self.bar.setRange(0, 100)
            self.bar.setValue(pct)
            self.lbl_percent.setText(f"진행률: {pct:d}% ({human_bytes(recvd)} / {human_bytes(total)})")

    def _update_speed_eta(self):
        """타이머에 의해 0.5초마다 호출되어 속도와 남은 시간을 계산합니다."""
        now = time.monotonic()
        if self.last_time is None or self.start_time is None:
            self.last_time = now
            self.last_bytes = self.bytes_received
            return

        dt = now - self.last_time
        if dt > 0.1:
            delta = self.bytes_received - self.last_bytes
            speed_bps = max(0, delta / dt)
            self.lbl_speed.setText(f"속도: {human_bytes(speed_bps)}/s")

            if self.bytes_total > 0 and speed_bps > 0:
                remain_bytes = self.bytes_total - self.bytes_received
                if remain_bytes > 0:
                    remain_time = remain_bytes / speed_bps
                    self.lbl_eta.setText(f"남은시간: {human_time(remain_time)}")
                else:
                    self.lbl_eta.setText("")
            else:
                self.lbl_eta.setText("")

            self.last_time = now
            self.last_bytes = self.bytes_received