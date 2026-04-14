# ──────────────────────────────────────────────────────────────────────────────
# Ui/download_task.py
# UI와 연동하여 파일 다운로드(HTTP, Google Drive, Hugging Face)를 수행하는 클래스입니다.
# ──────────────────────────────────────────────────────────────────────────────
import time
import os
import hashlib
import zipfile
import re
import requests
from typing import Optional, Callable

from PyQt5.QtCore import QTimer, QObject, QUrl, QCoreApplication
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

# Hugging Face API
from huggingface_hub import HfApi, hf_hub_url

# 유틸리티 함수
from Ui.utils import human_bytes, human_time


class DownloadTask(QObject):
    """
    파일 다운로드 작업을 관리하는 클래스.
    - 일반 HTTP 다운로드 (QNetworkAccessManager 사용)
    - Google Drive 다운로드 (requests 사용)
    - Hugging Face 리포지토리 다운로드 (HfApi + requests 사용)
    - 진행률 표시, 속도/남은 시간 계산, 체크섬 검증, 압축 해제 기능 포함
    """
    def __init__(self, parent, nam: QNetworkAccessManager, save_dir: str,
                 key: str, url: str, sha256: str,
                 btn_start, btn_cancel, bar, lbl_percent, lbl_speed, lbl_eta, lbl_status,
                 custom_name: str,
                 mirror_url: Optional[str] = None,
                 mirror_state_getter: Optional[Callable[[], bool]] = None,
                 mirror_checksum: Optional[str] = None):
        super().__init__(parent)
        
        # 기본 설정
        self.parent = parent
        self.nam = nam
        self.key = key
        self.url = url
        self.custom_name = custom_name

        # [수정] 모델 종류에 따라 하위 폴더 자동 분류
        sub_folder = ""
        if "Gemma" in key or "Gemma4" in key:
            sub_folder = "LLM"
        elif "TTS" in key:
            sub_folder = "TTS"
        elif "Emotion" in key:
            sub_folder = "Emotion"

        if sub_folder:
            self.save_dir = os.path.join(save_dir, sub_folder)
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = save_dir

        # UI 위젯 참조
        self.btn_start = btn_start
        self.btn_cancel = btn_cancel
        self.bar = bar
        self.lbl_percent = lbl_percent
        self.lbl_speed = lbl_speed
        self.lbl_eta = lbl_eta
        self.lbl_status = lbl_status

        # 체크섬 및 미러 설정
        self.base_checksum = sha256
        self.mirror_checksum = mirror_checksum
        self.expected_sha256 = sha256
        self.mirror_url = mirror_url
        self.mirror_state_getter = mirror_state_getter

        # 내부 상태 변수
        self.final_path: Optional[str] = None
        self.part_path: Optional[str] = None
        self.reply: Optional[QNetworkReply] = None
        self.file = None
        self.hasher = None
        
        self.bytes_received = 0
        self.bytes_total = 0
        self.last_bytes = 0
        self.last_time = None
        self.start_time = None
        self._is_cancelled = False
        self._is_running = False  # 중복 실행 방지 플래그

        # 타이머 (속도 및 ETA 갱신용)
        self.timer = QTimer(self)
        self.timer.setInterval(500) # 0.5초마다 갱신
        self.timer.timeout.connect(self._update_speed_eta)

        # 초기 UI 상태 설정
        self._reset_ui()

        # 버튼 시그널 연결
        self.btn_start.clicked.connect(self.start)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_cancel.setEnabled(False)

    def _reset_ui(self):
        """UI 위젯을 초기 상태로 리셋합니다."""
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.lbl_percent.setText("")
        self.lbl_speed.setText("")
        self.lbl_eta.setText("")
        self.lbl_status.setText("상태: 대기")

    def start(self):
        """다운로드 작업을 시작합니다."""
        if self._is_running:
            self.lbl_status.setText("상태: 이미 진행 중")
            return
        
        self._is_running = True
        self._is_cancelled = False
        
        # 미러 사이트 사용 여부 확인 (Uncensored 등)
        use_mirror = False
        if callable(self.mirror_state_getter):
            try:
                use_mirror = bool(self.mirror_state_getter())
            except Exception:
                use_mirror = False

        # URL 및 체크섬 결정
        chosen_url = self.mirror_url if (use_mirror and self.mirror_url) else self.url
        self.expected_sha256 = (self.mirror_checksum if (use_mirror and self.mirror_checksum)
                                else self.base_checksum)
        
        if use_mirror and not self.mirror_url:
            self.lbl_status.setText("상태: 실패 (미지원 모델)")
            self._is_running = False
            return

        # 1. Hugging Face 리포지토리 다운로드 (예: Emotion 모델)
        if self.key in ["Emotion"]:
            # Emotion 모델은 이미 하위 폴더(Emotion)가 save_dir에 포함되어 있으므로
            # 추가적인 하위 폴더 생성을 방지하기 위해 folder_name을 빈 문자열로 전달하거나
            # save_dir 구조에 맞게 조정해야 함.
            # 여기서는 save_dir 자체가 이미 .../Emotion 이므로 folder_name="" 로 전달
            self.download_huggingface_repo_with_ui(chosen_url, "")
            return
            
        # 2. Google Drive 다운로드
        if "drive.google.com" in chosen_url:
            self.download_google_drive(chosen_url)
            return

        # 3. 일반 HTTP 다운로드
        self._start_http_download(chosen_url, use_mirror)

    def _start_http_download(self, url: str, use_mirror: bool):
        """일반 HTTP 파일 다운로드를 시작합니다."""
        # 파일명 결정 (_uncensored 접미사 처리)
        fname = self.custom_name
        if use_mirror:
            root, ext = os.path.splitext(fname)
            fname = f"{root}_uncensored{ext}"

        self.final_path = os.path.join(self.save_dir, fname)
        self.part_path = self.final_path + ".part"

        qurl = QUrl(url)
        if not qurl.isValid():
            self.lbl_status.setText("상태: 실패 (URL 오류)")
            self._is_running = False
            return

        # 임시 파일 준비
        try:
            if os.path.exists(self.part_path):
                os.remove(self.part_path)
            os.makedirs(self.save_dir, exist_ok=True)
            self.file = open(self.part_path, "wb")
        except Exception:
            self.lbl_status.setText("상태: 실패 (파일 생성 오류)")
            self._is_running = False
            return

        # 상태 초기화
        self.bytes_received = 0
        self.bytes_total = 0
        self.last_bytes = 0
        self.start_time = time.monotonic()
        self.last_time = self.start_time
        self.hasher = hashlib.sha256()
        
        # UI 업데이트
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.lbl_percent.setText("진행률: 0%")
        self.lbl_speed.setText("속도: -")
        self.lbl_eta.setText("남은시간: -")
        self.lbl_status.setText("상태: 다운로드 중...")
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        # 네트워크 요청 시작
        req = QNetworkRequest(qurl)
        req.setRawHeader(b"User-Agent", b"PyQtDownloader/2.1")
        req.setAttribute(QNetworkRequest.FollowRedirectsAttribute, True)
        
        self.reply = self.nam.get(req)
        self.reply.readyRead.connect(self._on_ready_read)
        self.reply.downloadProgress.connect(self._on_progress)
        
        # 에러 시그널 연결 (Qt 버전에 따른 호환성 처리)
        if hasattr(self.reply, "errorOccurred"):
            self.reply.errorOccurred.connect(self._on_error)
        else:
            self.reply.error.connect(self._on_error)
            
        self.reply.finished.connect(self._on_finished)
        self.timer.start()

    def download_huggingface_repo_with_ui(self, repo_id: str, folder_name: str):
        """
        Hugging Face 리포지토리의 파일들을 다운로드합니다.
        UI 프로그레스 바와 연동되어 진행 상황을 표시합니다.
        """
        self.lbl_status.setText("상태: 파일 목록 조회 중...")
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        QCoreApplication.processEvents()

        try:
            # 1. 파일 목록 및 전체 크기 조회
            api = HfApi()
            model_info = api.model_info(repo_id=repo_id, files_metadata=True)
            siblings = model_info.siblings
            
            self.bytes_total = sum((f.size if f.size is not None else 0) for f in siblings)
            self.bytes_received = 0
            
            target_dir = os.path.join(self.save_dir, folder_name)
            os.makedirs(target_dir, exist_ok=True)

            self.start_time = time.monotonic()
            self.last_time = self.start_time
            self.timer.start()

            # 2. 각 파일 순차 다운로드
            for i, file_info in enumerate(siblings):
                if self._is_cancelled:
                    raise InterruptedError("사용자에 의해 취소됨")

                filename = file_info.rfilename
                self.lbl_status.setText(f"상태: [{i+1}/{len(siblings)}] {os.path.basename(filename)} 다운로드 중...")
                QCoreApplication.processEvents()

                file_url = hf_hub_url(repo_id=repo_id, filename=filename)
                local_path = os.path.join(target_dir, filename)
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # 이미 다운로드된 파일 스킵 (크기 비교)
                if os.path.exists(local_path) and file_info.size is not None and os.path.getsize(local_path) == file_info.size:
                    self.bytes_received += file_info.size
                    self._on_progress(self.bytes_received, self.bytes_total)
                    continue

                # requests를 사용한 스트리밍 다운로드
                response = requests.get(file_url, stream=True)
                response.raise_for_status()

                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if self._is_cancelled:
                            raise InterruptedError("사용자에 의해 취소됨")
                        if chunk:
                            f.write(chunk)
                            self.bytes_received += len(chunk)
                            self._on_progress(self.bytes_received, self.bytes_total)
                            QCoreApplication.processEvents()
            
            # 완료 처리
            self.timer.stop()
            self.lbl_status.setText("상태: 완료")
            self.bar.setValue(100)
            self.lbl_percent.setText("100%")

        except InterruptedError:
            self.timer.stop()
            self.lbl_status.setText("상태: 취소됨")

        except Exception as e:
            self.timer.stop()
            self.lbl_status.setText(f"상태: 실패 ({type(e).__name__})")
        
        finally:
            self.btn_start.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self._is_running = False

    def download_google_drive(self, url):
        """Google Drive 파일을 다운로드합니다."""
        self.lbl_status.setText("상태: 준비 중...")
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)

        try:
            # 파일명 및 경로 설정
            use_mirror = False
            if callable(self.mirror_state_getter):
                use_mirror = bool(self.mirror_state_getter())
            
            fname = self.custom_name
            if use_mirror:
                root, ext = os.path.splitext(fname)
                fname = f"{root}_uncensored{ext}"

            self.final_path = os.path.join(self.save_dir, fname)
            os.makedirs(self.save_dir, exist_ok=True)
            if os.path.exists(self.final_path):
                os.remove(self.final_path)

            self.lbl_status.setText("상태: 다운로드 중...")
            QCoreApplication.processEvents()

            # 파일 ID 추출
            file_id = None
            match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
            if match:
                file_id = match.group(1)
            else:
                match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
                if match:
                    file_id = match.group(1)
            
            if not file_id:
                raise ValueError("URL에서 파일 ID를 찾을 수 없습니다.")

            # 다운로드 세션 시작
            session = requests.Session()
            # User-Agent 설정 (브라우저처럼 보이게 하여 차단 가능성 낮춤)
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })

            initial_url = "https://docs.google.com/uc?export=download"
            # Note: 이 요청은 파일 다운로드를 위해 1회(경고 페이지가 있는 경우 2회)만 수행됩니다.
            response = session.get(initial_url, params={'id': file_id}, stream=True)
            
            # 대용량 파일 경고 페이지 처리
            if 'text/html' in response.headers.get('Content-Type', ''):
                html_content = response.text
                
                # 쿼터 초과 에러 메시지 확인
                if "지금은 이 파일을 보거나 다운로드할 수 없습니다" in html_content:
                    raise Exception("구글 드라이브 허용량 초과 (24시간 후 재시도)")

                action_match = re.search(r'<form id="download-form" action="([^"]+)"', html_content)
                if not action_match:
                    raise ValueError("다운로드 확인 폼을 찾을 수 없습니다.")
                final_url = action_match.group(1).replace("&amp;", "&")
                inputs = re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]+)">', html_content)
                params = {name: value for name, value in inputs}
                response = session.get(final_url, params=params, stream=True)

            self.bytes_total = int(response.headers.get('content-length', 0))
            self.bytes_received = 0
            self.hasher = hashlib.sha256()
            self.start_time = time.monotonic()
            self.last_time = self.start_time
            self.timer.start()

            # 파일 쓰기
            with open(self.final_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if self._is_cancelled:
                        raise InterruptedError("사용자에 의해 취소됨")
                    if chunk:
                        f.write(chunk)
                        self.hasher.update(chunk)
                        self.bytes_received += len(chunk)
                        self._on_progress(self.bytes_received, self.bytes_total)
                        QCoreApplication.processEvents()

            self.timer.stop()
            self.lbl_status.setText("상태: 검사 중...")
            QCoreApplication.processEvents()
            
            # 체크섬 검증
            passed = False
            calc_hex = self.hasher.hexdigest()
            if not self.expected_sha256:
                passed = True
            else:
                passed = (calc_hex.lower() == self.expected_sha256.lower())

            if passed:
                # 압축 해제 처리
                if self.final_path.lower().endswith(".zip"):
                    self.lbl_status.setText("상태: 압축 해제 중...")
                    QCoreApplication.processEvents()
                    extract_to = os.path.splitext(self.final_path)[0]
                    os.makedirs(extract_to, exist_ok=True)
                    try:
                        with zipfile.ZipFile(self.final_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_to)
                        msg = "상태: 완료"
                        # 압축 해제 후 원본 zip 삭제
                        try: os.remove(self.final_path)
                        except: pass
                    except Exception:
                        msg = "상태: 실패 (압축 해제 오류)"
                else:
                    msg = "상태: 완료"
                self.lbl_status.setText(msg)
                self.bar.setValue(100)
            else:
                self.lbl_status.setText("상태: 실패 (체크섬 불일치)")
                os.remove(self.final_path)
                self.bar.setValue(0)

        except InterruptedError:
            self.timer.stop()
            self.lbl_status.setText("상태: 취소됨")
            if self.final_path and os.path.exists(self.final_path):
                try: os.remove(self.final_path)
                except: pass
        except Exception as e:
            self.timer.stop()
            self.lbl_status.setText(f"상태: 실패 ({type(e).__name__})")
            if self.final_path and os.path.exists(self.final_path):
                try: os.remove(self.final_path)
                except: pass
        finally:
            self.btn_start.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self._is_running = False

    def cancel(self):
        """다운로드 작업을 취소합니다."""
        self._is_cancelled = True
        if self.reply:
            self.lbl_status.setText("상태: 취소 중...")
            self.timer.stop()
            self.reply.abort()
        else:
            self.lbl_status.setText("상태: 취소 중...")

    # ─────────────────────────────────────────────────────────────
    # QNetworkReply 슬롯 (HTTP 다운로드용)
    # ─────────────────────────────────────────────────────────────
    def _on_ready_read(self):
        """데이터가 도착할 때마다 파일에 쓰고 해시를 업데이트합니다."""
        if not self.reply or not self.file:
            return
        data = self.reply.readAll()
        if not data.isEmpty():
            b = bytes(data)
            self.file.write(b)
            if self.hasher is not None:
                self.hasher.update(b)

    def _on_progress(self, recvd, total):
        """다운로드 진행률을 UI에 업데이트합니다."""
        self.bytes_received = int(recvd)
        self.bytes_total = int(total)
        if total > 0:
            pct = int((recvd / total) * 100)
            self.bar.setRange(0, 100)
            self.bar.setValue(pct)
            self.lbl_percent.setText(f"{pct:d}% ({human_bytes(recvd)} / {human_bytes(total)})")
        else:
            self.bar.setRange(0, 0)
            self.lbl_percent.setText(f"{human_bytes(recvd)}")

    def _update_speed_eta(self):
        """다운로드 속도와 남은 시간을 계산하여 UI에 표시합니다."""
        now = time.monotonic()
        if self.last_time is None or self.start_time is None:
            self.last_time = now
            self.start_time = now
            self.last_bytes = self.bytes_received
            return
        
        dt = now - self.last_time
        if dt > 0.1: # 최소 0.1초 간격으로 업데이트
            delta = self.bytes_received - self.last_bytes
            speed_bps = max(0, delta / dt)
            self.lbl_speed.setText(f"{human_bytes(speed_bps)}/s")
            
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

    def _on_error(self, code):
        """네트워크 오류 발생 시 처리합니다."""
        self.timer.stop()
        self.lbl_status.setText("상태: 실패 (네트워크 오류)")
        self._cleanup(success=False)

    def _on_finished(self):
        """HTTP 다운로드 완료 시 처리합니다."""
        try:
            self.timer.stop()
            ok = (self.reply and self.reply.error() == QNetworkReply.NoError)
            calc_hex = None
            if ok and self.hasher is not None:
                calc_hex = self.hasher.hexdigest()

            if not ok:
                if not self._is_cancelled:
                    self.lbl_status.setText("상태: 실패")
                return

            # 체크섬 검증
            passed = True
            if self.expected_sha256 and calc_hex:
                passed = (calc_hex.lower() == self.expected_sha256.lower())

            if passed:
                try:
                    if self.file: self.file.close()
                    self.file = None
                    
                    # 기존 파일 삭제 후 이름 변경
                    if os.path.exists(self.final_path):
                        os.remove(self.final_path)
                    os.rename(self.part_path, self.final_path)
                    
                    # 압축 해제 처리
                    if self.final_path.lower().endswith(".zip"):
                        self.lbl_status.setText("상태: 압축 해제 중...")
                        extract_to = os.path.splitext(self.final_path)[0]
                        os.makedirs(extract_to, exist_ok=True)
                        try:
                            with zipfile.ZipFile(self.final_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_to)
                            msg = "상태: 완료"
                            try: os.remove(self.final_path)
                            except: pass
                        except Exception:
                            msg = "상태: 실패 (압축 해제 오류)"
                    else:
                        msg = "상태: 완료"
                    self.lbl_status.setText(msg)
                except Exception:
                    self.lbl_status.setText("상태: 실패 (파일 처리 오류)")
                    return
            else:
                self.lbl_status.setText("상태: 실패 (체크섬 불일치)")
        finally:
            self._cleanup(success=(self.reply is not None and self.reply.error() == QNetworkReply.NoError and not self._is_cancelled))

    def _cleanup(self, success: bool):
        """자원을 정리하고 UI 상태를 복구합니다."""
        try:
            if self.file: self.file.close()
        except Exception: pass
        self.file = None
        
        try:
            if not success and self.part_path and os.path.exists(self.part_path):
                os.remove(self.part_path)
        except Exception: pass
        
        if self.reply:
            self.reply.deleteLater()
        self.reply = None
        self.hasher = None
        
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._is_running = False

        if not success:
            self.bar.setRange(0, 100)
            self.bar.setValue(0)
