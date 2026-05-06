# ──────────────────────────────────────────────────────────────────────────────
# Ui/controllers.py
# 네비게이션, 시스템 제어, 모델 다운로드 등을 담당하는 컨트롤러 모음입니다.
# ──────────────────────────────────────────────────────────────────────────────
from functools import partial
import os
import sys
import json
import logging
import platform

from PyQt5.QtCore import pyqtSlot, QObject, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QFileDialog, QComboBox, QGraphicsColorizeEffect
from PyQt5.QtNetwork import QNetworkAccessManager

# 프로젝트 내부 모듈
from Ui.server_control import ServerControl
import Core.server_config as server_config
from Core.sql import MySQLManager
import Ui.ws_server as ws_server
import Core.rag_search as rag_search

# 선택적 모듈 임포트 (다운로드 관련)
try:
    from Ui.model_registrys import build_default_registry
    _HAS_REG = True
except ImportError:
    _HAS_REG = False

try:
    from Ui.download_task import DownloadTask
    _HAS_DL = True
except ImportError:
    _HAS_DL = False

# 모듈 경로 보정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ──────────────────────────────────────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────────────────────────────────────
MODEL_LLM = 0
MODEL_EMOTION = 1
MODEL_TTS_GPT = 2   # GPT 모델 파일 (.ckpt)
MODEL_TTS_CKPT = 3  # SoVITS 모델 파일 (.pth)
CONFIG_PATH = "config.json"

# 로거 설정
logger = logging.getLogger("system.controllers")
logger_llm = logging.getLogger("llm")


# ──────────────────────────────────────────────────────────────────────────────
# Navigation Controller
# UI의 페이지 전환(StackedWidget)을 담당합니다.
# ──────────────────────────────────────────────────────────────────────────────
class NavController:
    def __init__(self, root: QDialog):
        self.w = root
        
        # 필수 위젯 존재 여부 확인 (Phi4, Gemma44 잔재 제거 및 Gemma4로 통일)
        required_widgets = (
            "Gemma3PrevButton", "Gemma3NextButton",
            "Gemma4PrevButton", "Gemma4NextButton",
            "Gemma3StackedWidget", "Gemma4StackedWidget"
        )
        
        for attr in required_widgets:
            if not hasattr(self.w, attr):
                logger.debug(f"[NavController] '{attr}' 위젯을 찾을 수 없어 네비게이션 연결을 건너뜁니다.")
                return

        # 버튼 시그널 연결
        self._connect_signals()

    def _connect_signals(self):
        """버튼 클릭 시그널을 슬롯에 연결합니다."""
        self.w.Gemma3PrevButton.clicked.connect(self.on_gemma3_prev)
        self.w.Gemma3NextButton.clicked.connect(self.on_gemma3_next)
        self.w.Gemma4PrevButton.clicked.connect(self.on_gemma4_prev)
        self.w.Gemma4NextButton.clicked.connect(self.on_gemma4_next)
        
        self.w.LLMAdvancedButton.clicked.connect(self.on_llm_advanced)
        self.w.LLMGeneralButton.clicked.connect(self.on_llm_general)
        
        self.w.TTSAdvancedButton.clicked.connect(self.on_tts_advanced)
        self.w.TTSGeneralButton.clicked.connect(self.on_tts_general)
        self.w.TTSNextButton_2.clicked.connect(self.on_tts_next)
        self.w.TTSPrevButton_2.clicked.connect(self.on_tts_prev)

    def _shift(self, stacked, delta: int, name: str):
        """
        QStackedWidget의 페이지를 이동시킵니다.
        :param stacked: 대상 QStackedWidget
        :param delta: 이동할 인덱스 변화량 (+1: 다음, -1: 이전)
        :param name: 로그용 이름
        """
        cur = stacked.currentIndex()
        cnt = stacked.count()
        new_idx = max(0, min(cnt - 1, cur + delta))
        stacked.setCurrentIndex(new_idx)
        logger.info(f"[{name}] 페이지 변경: {cur + 1} -> {new_idx + 1} (총 {cnt}페이지)")

    # ── 슬롯 메서드 ──
    def on_gemma3_prev(self): self._shift(self.w.Gemma3StackedWidget, -1, "Gemma3")
    def on_gemma3_next(self): self._shift(self.w.Gemma3StackedWidget, +1, "Gemma3")
    def on_gemma4_prev(self): self._shift(self.w.Gemma4StackedWidget, -1, "Gemma4")
    def on_gemma4_next(self): self._shift(self.w.Gemma4StackedWidget, +1, "Gemma4")

    def on_llm_advanced(self): self._shift(self.w.LLMstackedWidget, +1, "LLM")
    def on_llm_general(self): self._shift(self.w.LLMstackedWidget, -1, "LLM")
    
    def on_tts_advanced(self): self._shift(self.w.TTSstackedWidget, +1, "TTS")
    def on_tts_general(self): self._shift(self.w.TTSstackedWidget, -1, "TTS")
    def on_tts_next(self): self._shift(self.w.Gemma4StackedWidget_2, +1, "TTS")
    def on_tts_prev(self): self._shift(self.w.Gemma4StackedWidget_2, -1, "TTS")


# ──────────────────────────────────────────────────────────────────────────────
# System Controller
# 서버 프로세스 관리, 설정 로드/저장, 하드웨어 모니터링 등을 담당합니다.
# ──────────────────────────────────────────────────────────────────────────────
class SystemController:
    def __init__(self, root: QDialog):
        self.w = root
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. ServerControl 초기화 (백그라운드 서버 프로세스 관리)
        self.ctrl = ServerControl(
            workdir=script_dir,
            module_spec="ws_server:app",
            host="0.0.0.0",
            port=3545,
            venv_dir=None,
            python_exe=None,
            parent=self.w,
        )

        # 2. UI 위젯 바인딩 (서버 상태 표시 등)
        self._bind_server_controls()

        # 3. 버튼 이벤트 연결
        self._connect_buttons()

        # 4. 초기 설정 (이미지, GPU 정보 등)
        self._init_ui_components()

        # 5. 설정 파일 로드
        self.load_config()
        
        # 6. 다운로드 컨트롤러 초기화 (선택적)
        self._init_download_controller()

        logger.info("SystemController Initialized")

    def _bind_server_controls(self):
        """서버 제어 및 상태 표시 위젯을 ServerControl에 연결합니다."""
        startstop_btn = getattr(self.w, "StartStopButton", None)
        restart_btn   = getattr(self.w, "RestartButton", None)

        # 상태 표시 위젯 (TextBrowser 우선, 없으면 Label)
        status_in = getattr(self.w, "StatusTextBrowserIn", None) or getattr(self.w, "StatusLabelIn", None)
        status_out = getattr(self.w, "StatusTextBrowserOut", None) or getattr(self.w, "StatusLabelOut", None)
        status_llm = getattr(self.w, "StatusTextBrowserLLM", None) or getattr(self.w, "StatusLabelLLM", None)
        status_tts = getattr(self.w, "StatusTextBrowserTTS", None) or getattr(self.w, "StatusLabelTTS", None)
        status_audio = getattr(self.w, "StatusTextBrowserAudio", None) or getattr(self.w, "StatusLabelAudio", None)

        self.ctrl.bind_controls(
            startstop_btn=startstop_btn,
            restart_btn=restart_btn,
            status_in_widget=status_in,
            status_out_widget=status_out,
            status_llm_widget=status_llm,
            status_tts_widget=status_tts,
            status_audio_widget=status_audio
        )

    def _connect_buttons(self):
        """각종 버튼의 클릭 이벤트를 연결합니다."""
        # 로그 삭제 버튼
        if hasattr(self.w, "LLMTextDeleteButton"):
            self.w.LLMTextDeleteButton.clicked.connect(self.on_llm_delete)
        if hasattr(self.w, "SystemLogDeleteButton"):
            self.w.SystemLogDeleteButton.clicked.connect(self.on_system_delete)

        # 시스템 제어 버튼
        if hasattr(self.w, "ExitButton"):
            self.w.ExitButton.clicked.connect(self.on_exit)
        if hasattr(self.w, "ApplyButton"):
            self.w.ApplyButton.clicked.connect(self.on_apply)
        
        # DB 설정 불러오기 버튼
        if hasattr(self.w, "LoadCommandButton"):
            self.w.LoadCommandButton.clicked.connect(self.on_command_fetch)
        if hasattr(self.w, "LoadCommandButtonLLM"):
            self.w.LoadCommandButtonLLM.clicked.connect(self.on_command_fetch_llm)
        if hasattr(self.w, "LoadCommandButtonTTS"):
            self.w.LoadCommandButtonTTS.clicked.connect(self.on_command_fetch_tts)

        # 모델 파일 찾기 버튼
        if hasattr(self.w, "LLMFindButton"):
            self.w.LLMFindButton.clicked.connect(partial(self.on_model_find_button, MODEL_LLM))
        if hasattr(self.w, "EmotionFindButton"):
            self.w.EmotionFindButton.clicked.connect(partial(self.on_model_find_button, MODEL_EMOTION))
        if hasattr(self.w, "TTSGPTFindButton"):
            self.w.TTSGPTFindButton.clicked.connect(partial(self.on_model_find_button, MODEL_TTS_GPT))
        if hasattr(self.w, "TTSCKPTFindButton"):
            self.w.TTSCKPTFindButton.clicked.connect(partial(self.on_model_find_button, MODEL_TTS_CKPT))

        # 테스트 버튼
        if hasattr(self.w, "SqlTestButton"):
            self.w.SqlTestButton.clicked.connect(self.on_sql_test_button)
        if hasattr(self.w, "RAGSqlTestButton"):
            self.w.RAGSqlTestButton.clicked.connect(self.on_rag_test_button)

        # TTS 활성화 토글
        if hasattr(self.w, "TTSEnableButtonYes"):
            self.w.TTSEnableButtonYes.toggled.connect(self.on_tts_enable_toggled)

    def on_tts_enable_toggled(self):
        """TTS 활성화 여부에 따라 관련 UI를 활성화/비활성화합니다."""
        enabled = self.w.TTSEnableButtonYes.isChecked()
        self._update_tts_ui_state(enabled)

    def _update_tts_ui_state(self, enabled):
        """TTS 관련 위젯들의 활성화 상태를 일괄 변경합니다."""
        widgets = [
            "LLMFindGroupBox_5",    # TTS 모델 경로 그룹
            "LoadCommandButtonTTS", # DB 불러오기
            "GPUGroupBox_3",        # GPU 설정
            "groupBox_60",          # 언어 설정
            "TTSAdvancedButton",    # 고급 설정 진입
        ]
        for name in widgets:
            w = getattr(self.w, name, None)
            if w:
                w.setEnabled(enabled)

    def _init_ui_components(self):
        """UI 컴포넌트 초기화 (이미지, GPU 정보, CPU 스레드 등)"""
        # 샘플 이미지 로드
        sample_img = os.path.join(os.path.dirname(__file__), "sample.png")
        if hasattr(self.w, "ProfileLabelImage") and os.path.exists(sample_img):
            self.w.ProfileLabelImage.setPixmap(QPixmap(sample_img))
            self.w.ProfileLabelImage.setScaledContents(True)

        # GPU 정보 조회 및 UI 업데이트
        self._update_gpu_info()

        # CPU 스레드 콤보박스 설정
        self.update_cpu_list()
        
        # Windows 환경 예외 처리
        if platform.system() == "Windows":
            if hasattr(self.w, "ThreadComboBox"):
                self.w.ThreadComboBox.setEnabled(False)
                self.w.ThreadComboBox.setCurrentText("1")
                self.w.ThreadComboBox.setToolTip("Windows에서는 멀티 워커를 지원하지 않아 1로 고정됩니다.")
                logger.info("Windows 환경 감지: ThreadComboBox 비활성화 (1로 고정)")
        else:
            if hasattr(self.w, "ThreadComboBox"):
                self.w.ThreadComboBox.setEnabled(True)
                self.w.ThreadComboBox.setToolTip("")

        # [NEW] 실시간 GPU 모니터링 타이머
        self.gpu_timer = QTimer(self.w)
        self.gpu_timer.timeout.connect(self._update_gpu_usage)
        self.gpu_timer.start(1000) # 1초마다 갱신

    def _update_gpu_info(self):
        """GPU 정보를 조회하여 UI에 반영합니다."""
        gpu_infos = ws_server.get_all_vram_info()
        logger.info(f"GPU 정보 조회 결과: {gpu_infos}")

        gpu_img_path = os.path.join(os.path.dirname(__file__), "gpu.png")
        pixmap = QPixmap(gpu_img_path) if os.path.exists(gpu_img_path) else None
        
        first_valid_tts_gpu_radio_set = False

        for i in range(5): # 최대 5개 GPU 슬롯 가정
            gpu_exists = i < len(gpu_infos)
            gpu_info = gpu_infos[i] if gpu_exists else None

            # 1. LLM GPU 위젯 설정
            self._setup_gpu_widget(i, gpu_exists, gpu_info, pixmap)

            # 2. TTS GPU 위젯 설정
            self._setup_tts_gpu_widget(i, gpu_exists, gpu_info, pixmap, first_valid_tts_gpu_radio_set)
            if gpu_exists and not first_valid_tts_gpu_radio_set:
                first_valid_tts_gpu_radio_set = True

    def _setup_gpu_widget(self, i, exists, info, pixmap):
        """LLM 탭의 개별 GPU 위젯 설정"""
        # 이미지
        img_widget = getattr(self.w, f"GpuImage{i}", None)
        if img_widget:
            if pixmap:
                img_widget.setPixmap(pixmap)
                img_widget.setScaledContents(True)
            
            if not exists:
                effect = QGraphicsColorizeEffect()
                effect.setColor(Qt.black)
                effect.setStrength(1.0)
                img_widget.setGraphicsEffect(effect)
            else:
                img_widget.setGraphicsEffect(None)

        # 라벨
        lbl_widget = getattr(self.w, f"GpuLabel{i}", None)
        if lbl_widget:
            if exists and info:
                lbl_widget.setText(f"{info['gpu_name']} ({info['gpu_total']}MB)")
                lbl_widget.setEnabled(True)
            else:
                lbl_widget.setText("No GPU")
                lbl_widget.setEnabled(False)

        # 체크박스
        chk_widget = getattr(self.w, f"GpuCheckBox{i}", None)
        if chk_widget:
            if not exists:
                chk_widget.setChecked(False)
                chk_widget.setEnabled(False)
            else:
                chk_widget.setEnabled(True)
                chk_widget.setChecked(True) # 기본값 ON
                # 최소 1개 선택 강제 로직 연결
                chk_widget.clicked.connect(partial(self.on_gpu_checkbox_clicked, chk_widget))

        # 스핀박스 (VRAM 할당량)
        spin_widget = getattr(self.w, f"GpuSpinBox{i}", None)
        if spin_widget:
            if not exists:
                spin_widget.setValue(0)
                spin_widget.setEnabled(False)
            else:
                spin_widget.setEnabled(True)
                if info:
                    val_gb = info['gpu_total'] / 1024
                    spin_widget.setMaximum(val_gb)
                    spin_widget.setValue(val_gb)

    def _setup_tts_gpu_widget(self, i, exists, info, pixmap, first_set):
        """TTS 탭의 개별 GPU 위젯 설정"""
        # 이미지
        img_widget = getattr(self.w, f"TTSGpuImage{i}", None)
        if img_widget:
            if pixmap:
                img_widget.setPixmap(pixmap)
                img_widget.setScaledContents(True)
            
            if not exists:
                effect = QGraphicsColorizeEffect()
                effect.setColor(Qt.black)
                effect.setStrength(1.0)
                img_widget.setGraphicsEffect(effect)
            else:
                img_widget.setGraphicsEffect(None)

        # 라디오 버튼
        radio_widget = getattr(self.w, f"TTSradioButton{i}", None)
        if radio_widget:
            if not exists:
                radio_widget.setChecked(False)
                radio_widget.setEnabled(False)
            else:
                radio_widget.setEnabled(True)
                # 첫 번째 유효한 GPU 자동 선택
                if not first_set:
                    radio_widget.setChecked(True)

        # 라벨
        lbl_widget = getattr(self.w, f"TTSGpuLabel{i}", None)
        if lbl_widget:
            if exists and info:
                lbl_widget.setText(f"{info['gpu_name']}\n({info['gpu_total']}MB)")
                lbl_widget.setEnabled(True)
            else:
                lbl_widget.setText("No GPU")
                lbl_widget.setEnabled(False)

    def _update_gpu_usage(self):
        """실시간 GPU 사용량을 조회하여 라벨(GpuPercentLabel, TTSPercentLabel)에 업데이트합니다."""
        try:
            gpu_infos = ws_server.get_all_vram_info()
            
            # 최대 6개 슬롯 (GpuPercentLabel0~5, TTSPercentLabel0~4)
            for i in range(6):
                # GPU 정보 가져오기
                text = "N/A"
                if i < len(gpu_infos):
                    info = gpu_infos[i]
                    used = info.get('gpu_used', 0)
                    total = info.get('gpu_total', 1)
                    if total == 0: total = 1
                    percent = (used / total) * 100
                    text = f"{percent:.1f}% ({used}MB / {total}MB)"

                # 1. LLM 탭 라벨 (GpuPercentLabel0 ~ 5)
                label_gpu = getattr(self.w, f"GpuPercentLabel{i}", None)
                if label_gpu:
                    label_gpu.setText(text)

                # 2. TTS 탭 라벨 (TTSPercentLabel0 ~ 4)
                if i < 5:
                    label_tts = getattr(self.w, f"TTSPercentLabel{i}", None)
                    if label_tts:
                        label_tts.setText(text)

        except Exception as e:
            logger.warning(f"GPU 모니터링 오류: {e}")

    def _init_download_controller(self):
        """다운로드 컨트롤러 초기화 (ModelRegistry 필요)"""
        self._dl = None
        
        if _HAS_REG and _HAS_DL:
            try:
                if getattr(self.w, "_dl_controller_attached", False):
                    logger.info("[DownloadController] 이미 연결되어 있습니다.")
                    return

                registry = build_default_registry()

                # 저장 경로 설정 (Models 폴더 강제)
                save_dir = getattr(server_config, "DOWNLOAD_DIR", None) or getattr(self.w, "save_dir", None)
                if not save_dir:
                    project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
                    save_dir = os.path.join(project_root, "Models")

                save_dir = os.path.normpath(save_dir)
                os.makedirs(save_dir, exist_ok=True)

                self._dl = DownloadController(root=self.w, save_dir=save_dir, registry=registry)
                logger.info(f"[DownloadController] 허깅페이스 다운로더 초기화 완료 (경로: {save_dir})")
                
            except Exception:
                logger.exception("[DownloadController] 초기화 실패")

    # ─────────────────────────────────────────────────────────────
    # 이벤트 핸들러 (Slots)
    # ─────────────────────────────────────────────────────────────
    def on_gpu_checkbox_clicked(self, clicked_checkbox):
        """GPU 체크박스 클릭 시 최소 1개 선택 유지 및 병렬 처리 옵션 갱신"""
        checked_count = 0
        for i in range(5):
            cb = getattr(self.w, f"GpuCheckBox{i}", None)
            if cb and cb.isEnabled() and cb.isChecked():
                checked_count += 1
        
        if checked_count == 0:
            clicked_checkbox.setChecked(True)
            logger.info("최소 하나의 GPU는 선택되어야 합니다.")
        
        self._update_parallel_checkbox_state()

    def _update_parallel_checkbox_state(self):
        """활성화된 GPU 개수에 따라 ParallelCheckBox 활성화 여부 결정"""
        checked_count = 0
        for i in range(5):
            cb = getattr(self.w, f"GpuCheckBox{i}", None)
            if cb and cb.isEnabled() and cb.isChecked():
                checked_count += 1
        
        parallel_cb = getattr(self.w, "ParallelCheckBox", None)
        if parallel_cb:
            parallel_cb.setEnabled(checked_count >= 2)

    def update_cpu_list(self):
        """CPU 코어 수에 맞춰 스레드 콤보박스 갱신"""
        self.w.ThreadComboBox.clear()
        for i in range(os.cpu_count()):
            self.w.ThreadComboBox.addItem(str(i+1))

    def on_llm_delete(self):
        self.w.LLMTextBrowser.clear()

    def on_system_delete(self):
        self.w.SystemTextBrowser.clear()

    def on_exit(self):
        logger.info("종료 버튼 클릭됨")
        self.w.close()

    def on_apply(self):
        self.save_config()

    def on_model_find_button(self, model_type: int, *_, **__):
        """모델 파일/폴더 선택 다이얼로그"""
        if model_type == MODEL_LLM:
            path = QFileDialog.getExistingDirectory(self.w, "LLM 모델 폴더 선택", "")
            if path:
                self._set_text("LLMLocationTextBrowser", os.path.dirname(path))
                self._set_text("LLMModelTextBrowser", os.path.basename(path))
                
        elif model_type == MODEL_EMOTION:
            path, _ = QFileDialog.getOpenFileName(self.w, "Emotion 모델 파일 선택", "", "Emotion Model Files (*.safetensors)")
            if path:
                self._set_text("EmotionLocationTextBrowser", os.path.dirname(path))
                self._set_text("EmotionModelTextBrowser", os.path.basename(path))
                
        elif model_type == MODEL_TTS_GPT:
            path, _ = QFileDialog.getOpenFileName(self.w, "GPT 모델 파일 선택", "", "GPT Model Files (*.ckpt)")
            if path:
                self._set_text("TTSGPTLocationTextBrowser", os.path.dirname(path))
                self._set_text("TTSGPTModelTextBrowser", os.path.basename(path))
                
        elif model_type == MODEL_TTS_CKPT:
            path, _ = QFileDialog.getOpenFileName(self.w, "SoVITS 모델 파일 선택", "", "SoVITS Model Files (*.pth)")
            if path:
                self._set_text("TTSCKPTLocationTextBrowser", os.path.dirname(path))
                self._set_text("TTSCKPTModelTextBrowser", os.path.basename(path))

    def _set_text(self, widget_name, text):
            """TextBrowser/Edit/LineEdit 텍스트 설정 헬퍼"""
            w = getattr(self.w, widget_name, None)
            if w:
                if hasattr(w, "setPlainText"):
                    w.setPlainText(text)
                elif hasattr(w, "setText"):
                    w.setText(text)

    def on_sql_test_button(self):
        """MySQL 연결 테스트"""
        db = MySQLManager()
        MySQLManager.test_mysql_connection(db, self.w)

    def on_rag_test_button(self):
        """RAG 서버 연결 테스트"""
        self.w.RAGTestResultLabel.setText("결과 : N/A (Google MCP 사용)")

    # ─────────────────────────────────────────────────────────────
    # 설정 로드 / 저장 (Config & DB)
    # ─────────────────────────────────────────────────────────────
    def load_config(self):
        """config.json 파일에서 설정을 로드하여 UI에 반영합니다."""
        if not os.path.exists(CONFIG_PATH):
            logger.debug(f"{CONFIG_PATH} 파일이 없어 로드를 건너뜁니다.")
            return

        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.exception(f"{CONFIG_PATH} 로드 실패")
            return

        # GPU 구성 검증
        saved_gpus = data.get("GPU_DEVICES", [])
        current_gpus = ws_server.get_all_vram_info()
        gpu_config_valid = self._validate_gpu_config(saved_gpus, current_gpus)

        # 기본 UI 설정 반영 (텍스트 필드 등)
        self._apply_basic_config(data)
        
        # GPU 관련 설정 반영
        self._apply_gpu_config(data, saved_gpus, gpu_config_valid)

        # LLM/TTS 상세 설정 반영
        try:
            self._apply_llm_tts_ui(data)
        except Exception:
            logger.exception("LLM/TTS 설정 반영 중 오류")

        # 전역 설정 리로드
        server_config.reload()
        logger.info(f"{CONFIG_PATH} 로드 완료")

    def _validate_gpu_config(self, saved_gpus, current_gpus):
        """저장된 GPU 정보와 현재 시스템 정보가 일치하는지 확인"""
        if not saved_gpus or len(saved_gpus) != len(current_gpus):
            return False
            
        for i, saved in enumerate(saved_gpus):
            current = current_gpus[i]
            if saved.get("name") != current["gpu_name"] or saved.get("total_mb") != current["gpu_total"]:
                logger.warning(f"GPU 변경 감지: 저장됨={saved.get('name')}, 현재={current['gpu_name']}")
                return False
        return True

    def _apply_basic_config(self, data):
        """기본적인 텍스트/숫자 설정 UI 반영"""
        # TTS 모델 경로
        self._set_text("TTSGPTLocationTextBrowser", data.get("LOCATION_GPT", ""))
        self._set_text("TTSGPTModelTextBrowser",    data.get("GPT_NAME", ""))
        self._set_text("TTSCKPTLocationTextBrowser", data.get("LOCATION_CKPT", ""))
        self._set_text("TTSCKPTModelTextBrowser",    data.get("CKPT_NAME", ""))

        # Emotion / DB / Concept
        self._set_text("EmotionLocationTextBrowser", data.get("LOCATION_EMOTION", ""))
        self._set_text("EmotionModelTextBrowser",    data.get("EMOTION_NAME", ""))
        
        if hasattr(self.w, "IPLineEdit"): self.w.IPLineEdit.setText(data.get("MYSQL_HOST", ""))
        if hasattr(self.w, "DBLineEdit"): self.w.DBLineEdit.setText(data.get("MYSQL_DATABASE", ""))
        if hasattr(self.w, "IDLineEdit"): self.w.IDLineEdit.setText(data.get("MYSQL_USER", ""))
        if hasattr(self.w, "PWLineEdit"): self.w.PWLineEdit.setText(data.get("MYSQL_PASSWORD", ""))
        if hasattr(self.w, "PortLineEdit"): self.w.PortLineEdit.setText(str(data.get("MYSQL_PORT", 3306)))
        
        self._set_text("CharacterLineEdit",  data.get("CHARACTER_NAME", ""))
        self._set_text("CharacterConceptTextEdit",  data.get("CHARACTER_CONCEPT", ""))
        self._set_text("CharacterFeedbackTextEdit", data.get("COMMAND_FEEDBACK", ""))
        self._set_text("CharacterSearchTextEdit",   data.get("COMMAND_SEARCH", ""))

    def _apply_gpu_config(self, data, saved_gpus, is_valid):
        """GPU 관련 설정(체크박스, 스핀박스, TTS 라디오버튼) 반영"""
        # TTS GPU 선택
        saved_tts_gpu = int(data.get("GPU_TTS", 0)) if is_valid else 0
        rb = getattr(self.w, f"TTSradioButton{saved_tts_gpu}", None)
        if rb and rb.isEnabled():
            rb.setChecked(True)
        else:
            # 기본값 0번 시도
            rb0 = getattr(self.w, "TTSradioButton0", None)
            if rb0 and rb0.isEnabled(): rb0.setChecked(True)

        # LLM GPU 선택 및 할당량
        if is_valid:
            gpu_index_str = str(data.get("GPU_INDEX", "0"))
            try:
                active_indices = [int(x.strip()) for x in gpu_index_str.split(",") if x.strip().isdigit()]
            except Exception:
                active_indices = [0]

            for i in range(5):
                chk = getattr(self.w, f"GpuCheckBox{i}", None)
                if chk: chk.setChecked(i in active_indices)
            
            for saved in saved_gpus:
                idx = saved.get("index")
                if idx is None: continue
                spin = getattr(self.w, f"GpuSpinBox{idx}", None)
                if spin:
                    try:
                        spin.setValue(float(saved.get("limit_gb", 0)))
                    except Exception:
                        pass
        
        self._update_parallel_checkbox_state()

    def _apply_llm_tts_ui(self, data: dict):
        """LLM 및 TTS 상세 파라미터를 UI에 반영 (JSON/DB 공용)"""
        
        # 헬퍼 함수들
        def set_spin(name, val):
            w = getattr(self.w, name, None)
            if w and val is not None:
                try: w.setValue(float(val))
                except: w.setValue(val)

        def set_combo(name, val):
            w = getattr(self.w, name, None)
            if w and val is not None:
                idx = w.findText(str(val))
                if idx >= 0: w.setCurrentIndex(idx)
                else: w.setCurrentText(str(val))

        def set_radio(names, val):
            if val is None: return
            yes, no = names
            w_yes = getattr(self.w, yes, None)
            w_no = getattr(self.w, no, None)
            if bool(val) and w_yes: w_yes.setChecked(True)
            elif not bool(val) and w_no: w_no.setChecked(True)

        # LLM 설정
        self._set_text("LLMLocationTextBrowser", data.get("LOCATION"))
        self._set_text("LLMModelTextBrowser", data.get("LOCATION_MODEL"))
        set_spin("LLMOutputSpinBox", data.get("CONTEXT", data.get("context")))
        set_combo("ThreadComboBox", data.get("CPU_THREADS", data.get("cpu_threads")))
        set_combo("LLMCacheQuantComboBox", data.get("CACHE_QUANT", data.get("llm_cache_quant")))
        
        tp = data.get("TENSOR_PARALLEL", data.get("llm_tensor_parallel"))
        if tp is not None:
            cb = getattr(self.w, "ParallelCheckBox", None)
            if cb: cb.setChecked(bool(tp))

        set_spin("LLMTemperatureSpinBox", data.get("LLM_TEMPERATURE", data.get("llm_temperature")))
        set_spin("LLMTopKSpinBox", data.get("LLM_TOP_K", data.get("llm_top_k")))
        set_spin("LLMTopPSpinBox", data.get("LLM_TOP_P", data.get("llm_top_p")))
        set_spin("LLMMinPSpinBox", data.get("LLM_MIN_P", data.get("llm_min_p")))
        set_spin("LLMRepetitionPenaltySpinBox", data.get("LLM_REPETITION_PENALTY", data.get("llm_repetition_penalty")))
        set_spin("LLMPresencePenaltySpinBox", data.get("LLM_PRESENCE_PENALTY", data.get("llm_presence_penalty")))
        set_spin("LLMFrequencyPenaltySpinBox", data.get("LLM_FREQUENCY_PENALTY", data.get("llm_frequency_penalty")))

        # TTS 설정
        set_combo("TextSplitMethodComboBox", data.get("TEXT_SPLIT_METHOD", data.get("tts_text_split_method", "cut5")))
        set_spin("BatchSizeSpinBox", data.get("BATCH_SIZE", data.get("tts_batch_size")))
        set_radio(("ParallelInferRadioButtonYes", "ParallelInferRadioButtonNo"), data.get("PARALLEL_INFER", data.get("tts_parallel_infer")))
        set_radio(("SplitBucketRadioButtonYes", "SplitBucketRadioButtonNo"), data.get("SPLIT_BUCKET", data.get("tts_split_bucket")))
        set_spin("SeedSpinBox", data.get("SEED", data.get("tts_seed")))
        set_spin("TopKSpinBox", data.get("TOP_K", data.get("tts_top_k", 5)))
        set_spin("TopPSpinBox", data.get("TOP_P", data.get("tts_top_p", 0.85)))
        set_spin("TemperatureSpinBox", data.get("TEMPERATURE", data.get("tts_temperature")))
        set_spin("RepetitionPenaltySpinBox", data.get("REPETITION_PENALTY", data.get("tts_repetition_penalty")))
        set_spin("SpeedFactorSpinBox", data.get("SPEED_FACTOR", data.get("tts_speed_factor")))
        set_combo("LanguageComboBox", data.get("TTS_LANGUAGE", data.get("tts_language")))
        set_spin("CommuLogSpinBox", data.get("COMMU_LOG_INTERVAL", data.get("commu_log_interval")))
        
        set_radio(("TTSEnableButtonYes", "TTSEnableNo"), data.get("TTS_ENABLE", data.get("tts_enable")))

    def save_config(self):
        """현재 UI 상태를 config.json 및 DB에 저장합니다."""
        # 헬퍼 함수들
        def get_text(name, default=""):
            w = getattr(self.w, name, None)
            return w.text() if w else default
        
        def get_plain(name, default=""):
            w = getattr(self.w, name, None)
            return w.toPlainText() if w else default
            
        def get_spin(name, default=0):
            w = getattr(self.w, name, None)
            return w.value() if w else default
            
        def get_combo_text(name, default=""):
            w = getattr(self.w, name, None)
            return w.currentText() if w else default
            
        def get_combo_int(name, default=0):
            try: return int(get_combo_text(name))
            except: return default
            
        def get_radio(yes_name, no_name, default=False):
            yes = getattr(self.w, yes_name, None)
            if yes and yes.isChecked(): return True
            no = getattr(self.w, no_name, None)
            if no and no.isChecked(): return False
            return default
            
        def get_check(name, default=False):
            w = getattr(self.w, name, None)
            return w.isChecked() if w else default

        # GPU 설정 수집
        tts_gpu_val = 0
        for i in range(5):
            rb = getattr(self.w, f"TTSradioButton{i}", None)
            if rb and rb.isChecked():
                tts_gpu_val = i
                break
        
        active_gpu_indices = []
        gpu_split_values = []
        gpu_list = []
        current_gpus = ws_server.get_all_vram_info()
        
        for i, gpu in enumerate(current_gpus):
            chk = getattr(self.w, f"GpuCheckBox{i}", None)
            spin = getattr(self.w, f"GpuSpinBox{i}", None)
            
            enabled = chk.isChecked() if chk else False
            limit_gb = spin.value() if spin else 0.0
            
            if enabled:
                active_gpu_indices.append(str(i))
                gpu_split_values.append(str(limit_gb))
            else:
                gpu_split_values.append("0")
                
            gpu_list.append({
                "index": i,
                "name": gpu["gpu_name"],
                "total_mb": gpu["gpu_total"],
                "enabled": enabled,
                "limit_gb": limit_gb
            })

        gpu_index_str = ",".join(active_gpu_indices) if active_gpu_indices else "0"
        gpu_split_str = ",".join(gpu_split_values) if gpu_split_values else ""

        # 데이터 구성
        data = {
            # LLM
            "LOCATION": get_plain("LLMLocationTextBrowser"),
            "LOCATION_MODEL": get_plain("LLMModelTextBrowser"),
            "CONTEXT": get_spin("LLMOutputSpinBox"),
            "CPU_THREADS": get_combo_int("ThreadComboBox"),
            "GPU_TTS": tts_gpu_val,
            "GPU_INDEX": gpu_index_str,
            "GPU_SPLIT": gpu_split_str,
            "CACHE_QUANT": get_combo_text("LLMCacheQuantComboBox"),
            "TENSOR_PARALLEL": get_check("ParallelCheckBox"),
            "LLM_TEMPERATURE": get_spin("LLMTemperatureSpinBox"),
            "LLM_TOP_K": int(get_spin("LLMTopKSpinBox")),
            "LLM_TOP_P": get_spin("LLMTopPSpinBox"),
            "LLM_MIN_P": get_spin("LLMMinPSpinBox"),
            "LLM_REPETITION_PENALTY": get_spin("LLMRepetitionPenaltySpinBox"),
            "LLM_PRESENCE_PENALTY": get_spin("LLMPresencePenaltySpinBox"),
            "LLM_FREQUENCY_PENALTY": get_spin("LLMFrequencyPenaltySpinBox"),

            # TTS
            "LOCATION_GPT":  get_plain("TTSGPTLocationTextBrowser"),
            "GPT_NAME":      get_plain("TTSGPTModelTextBrowser"),
            "LOCATION_CKPT": get_plain("TTSCKPTLocationTextBrowser"),
            "CKPT_NAME":     get_plain("TTSCKPTModelTextBrowser"),
            "TEXT_SPLIT_METHOD":  get_combo_text("TextSplitMethodComboBox"),
            "BATCH_SIZE":   get_spin("BatchSizeSpinBox"),
            "PARALLEL_INFER": get_radio("ParallelInferRadioButtonYes", "ParallelInferRadioButtonNo"),
            "SPLIT_BUCKET": get_radio("SplitBucketRadioButtonYes", "SplitBucketRadioButtonNo"),
            "SEED": get_spin("SeedSpinBox"),
            "TOP_K": get_spin("TopKSpinBox"),
            "TOP_P": get_spin("TopPSpinBox"),
            "TEMPERATURE": get_spin("TemperatureSpinBox"),
            "REPETITION_PENALTY": get_spin("RepetitionPenaltySpinBox"),
            "SPEED_FACTOR": get_spin("SpeedFactorSpinBox"),
            "TTS_LANGUAGE": get_combo_text("LanguageComboBox"),
            "COMMU_LOG_INTERVAL": get_spin("CommuLogSpinBox"),
            "TTS_ENABLE": get_radio("TTSEnableButtonYes", "TTSEnableNo"),

            # EMOTION & DB
            "LOCATION_EMOTION": get_plain("EmotionLocationTextBrowser"),
            "EMOTION_NAME":     get_plain("EmotionModelTextBrowser"),
            "MYSQL_HOST": get_text("IPLineEdit"),
            "MYSQL_DATABASE": get_text("DBLineEdit"),
            "MYSQL_USER": get_text("IDLineEdit"),
            "MYSQL_PASSWORD": get_text("PWLineEdit"),
            "MYSQL_PORT": int(get_text("PortLineEdit") or "3306"),
            
            # CONCEPT
            "CHARACTER_NAME": get_text("CharacterLineEdit"),
            "CHARACTER_CONCEPT": get_plain("CharacterConceptTextEdit"),
            "COMMAND_FEEDBACK":  get_plain("CharacterFeedbackTextEdit"),
            "COMMAND_SEARCH":    get_plain("CharacterSearchTextEdit"),
            
            # GPU List
            "GPU_DEVICES": gpu_list
        }

        # 파일 저장
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"설정 저장 완료: {CONFIG_PATH}")
            
            # 설정 리로드 및 서버 재시작
            self.load_config()
            if self.ctrl.status() != "stopped":
                self.ctrl.restart()
        except Exception:
            logger.exception("설정 파일 저장 실패")

        # DB 저장
        try:
            db = MySQLManager()
            if db.save_command():
                logger.info("DB 설정 저장 완료")
            else:
                logger.warning("DB 설정 저장 실패 또는 변경사항 없음")
        except Exception:
            logger.exception("DB 저장 중 오류")

        # DOWNLOAD_DIR 재고정
        try:
            save_dir = getattr(self.w, "save_dir", None)
            if save_dir and getattr(server_config, "DOWNLOAD_DIR", None) != save_dir:
                server_config.DOWNLOAD_DIR = save_dir
                logger.info(f"DOWNLOAD_DIR 재설정: {server_config.DOWNLOAD_DIR}")
        except Exception:
            logger.exception("DOWNLOAD_DIR 재설정 실패")

    # ── DB Fetch 메서드 ──
    def on_command_fetch(self):
        """DB에서 캐릭터/프롬프트 설정을 불러옵니다."""
        self._fetch_and_apply(MySQLManager().fetch_server_command_settings, "캐릭터 설정")

    def on_command_fetch_llm(self):
        """DB에서 LLM 설정을 불러옵니다."""
        self._fetch_and_apply(MySQLManager().fetch_llm_settings, "LLM 설정", self._apply_llm_tts_ui)

    def on_command_fetch_tts(self):
        """DB에서 TTS 설정을 불러옵니다."""
        self._fetch_and_apply(MySQLManager().fetch_tts_settings, "TTS 설정", self._apply_llm_tts_ui)

    def _fetch_and_apply(self, fetch_func, log_name, apply_func=None):
        try:
            data = fetch_func()
            if not data:
                logger.warning(f"{log_name} 조회 결과 없음")
                return
            
            if apply_func:
                apply_func(data)
            else:
                # 기본 캐릭터 설정 적용
                self._set_text("CharacterLineEdit", data.get('character_name', ""))
                self._set_text("CharacterConceptTextEdit", data.get('character_concept', ""))
                self._set_text("CharacterFeedbackTextEdit", data.get('command_feedback', ""))
                self._set_text("CharacterSearchTextEdit", data.get('command_search', ""))
                
            logger.info(f"{log_name} 불러오기 완료")
        except Exception:
            logger.exception(f"{log_name} 불러오기 실패")


# ──────────────────────────────────────────────────────────────────────────────
# Download Controller
# ──────────────────────────────────────────────────────────────────────────────
class DownloadController:
    def __init__(self, root: QDialog, save_dir: str, registry):
        self.w = root
        
        # 중복 바인딩 방지
        if getattr(self.w, "_dl_bound_once", False):
            logger.warning("[DownloadController] 이미 바인딩되어 있습니다.")
            return
        setattr(self.w, "_dl_bound_once", True)

        self._save_dir = save_dir
        self._reg = registry
        
        os.makedirs(self._save_dir, exist_ok=True)
        self.tasks = {}
        
        self._bind_download_tasks()

    def _bind_download_tasks(self):
        """레지스트리에 등록된 모델들에 대해 다운로드 UI를 연결합니다."""
        for key in getattr(self._reg, "items", []):
            # 필수 UI 위젯 확인
            widgets = self._get_download_widgets(key)
            if not widgets:
                continue

            btn_start, btn_cancel, bar, lbl_pct, lbl_spd, lbl_eta, lbl_stat = widgets

            # 이전 태스크 정리
            self._unbind_previous_task(key, btn_start, btn_cancel)

            # 새 태스크 생성 및 연결
            try:
                task = DownloadTask(
                    parent=self.w,
                    save_dir=self._save_dir,
                    key=key,
                    repo_id=self._reg.url_map.get(key),
                    btn_start=btn_start,
                    btn_cancel=btn_cancel,
                    bar=bar,
                    lbl_percent=lbl_pct,
                    lbl_speed=lbl_spd,
                    lbl_eta=lbl_eta,
                    lbl_status=lbl_stat
                )
                
                # 버튼에 태스크 참조 저장 (나중에 해제하기 위해)
                setattr(btn_start, "_dl_task", task)
                setattr(btn_cancel, "_dl_task", task)

                self.tasks[key] = task
                logger.debug(f"[DownloadController] '{key}' 바인딩 완료")
            except Exception as e:
                logger.exception(f"[DownloadController] '{key}' 바인딩 실패: {e}")

    def _get_download_widgets(self, key):
        """해당 키에 대한 다운로드 관련 위젯들을 찾아서 반환합니다."""
        names = [
            f"{key}DownloadButton", f"{key}CancelButton", f"{key}ProgressBar",
            f"{key}PercentLabel", f"{key}SpeedLabel", f"{key}EtaLabel", f"{key}StatusLabel"
        ]
        if not all(hasattr(self.w, n) for n in names):
            return None
        return [getattr(self.w, n) for n in names]

    def _unbind_previous_task(self, key: str, btn_start, btn_cancel):
        """기존에 연결된 다운로드 태스크를 안전하게 제거합니다."""
        old_task = getattr(btn_start, "_dl_task", None)
        if not old_task:
            return

        try:
            if hasattr(old_task, "dispose"):
                old_task.dispose()
            else:
                try: btn_start.clicked.disconnect()
                except: pass
                try: btn_cancel.clicked.disconnect()
                except: pass
        except Exception:
            logger.exception(f"[DownloadController] '{key}' 태스크 해제 중 오류")
        finally:
            if hasattr(btn_start, "_dl_task"): delattr(btn_start, "_dl_task")
            if hasattr(btn_cancel, "_dl_task"): delattr(btn_cancel, "_dl_task")