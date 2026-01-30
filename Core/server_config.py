# ──────────────────────────────────────────────────────────────────────────────
# Core/server_config.py
# 서버 전역 설정 관리 모듈
# config.json 파일에서 설정을 로드하여 각 클래스(LLM, TTS, SQL 등)에 매핑합니다.
# ──────────────────────────────────────────────────────────────────────────────
import os
import json

# 설정 파일 경로 (Core 기준 상위 폴더의 config.json)
DEFAULT_CFG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.json")
)

# 모델 다운로드 경로 (프로젝트 루트/Models)
DOWNLOAD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Models")
)

class PORTS:
    """서버 포트 및 호스트 설정"""
    LLAMA_HOST = "127.0.0.1"
    LLAMA_PORT = 8000
    TTS_HOST = "127.0.0.1"
    TTS_PORT = 9880
    AUDIO_SERVER_HOST = "127.0.0.1"
    AUDIO_SERVER_PORT = 3546
    MAIN_SERVER_PORT = 3545

class LLM:
    """LLM (Large Language Model) 관련 설정"""
    LOCATION: str = ""          # 모델 폴더 경로
    LOCATION_MODEL: str = ""    # 모델 파일/폴더명
    MODEL_TYPE: str = "gemma"   # 모델 타입 (gemma, phi 등)
    
    # 모델 로드 옵션
    CACHE_QUANT = ""            # 캐시 양자화 (4bit, 8bit 등)
    TENSOR_PARALLEL = False     # 텐서 병렬화 사용 여부
    GPU_INDEX = "0"             # 사용할 GPU 인덱스 (콤마로 구분)
    GPU_SPLIT = ""              # GPU 메모리 분할 설정
    
    # 생성 파라미터 (Generation Config)
    TEMPERATURE = "0.8"
    TOP_K = "40"
    TOP_P = "0.9"
    MIN_P = "0.05"
    REPETITION_PENALTY = "1.05"
    PRESENCE_PENALTY = "0.0"
    FREQUENCY_PENALTY = "0.0"
    
    # 시스템 리소스
    CPU_THREADS = 1             # CPU 스레드 수
    CONTEXT = 1024              # 컨텍스트 길이 (cache_size)

    # 프롬프트 템플릿 (DB 또는 config.json에서 로드)
    LLM_CHAT_FORMAT: str = ""
    LLM_RAG_SEARCH_FORMAT: str = ""
    LLM_FEEDBACK_FORMAT: str = ""
    
    # 대화 로그 설정
    COMMU_LOG_TIME = False      # 대화 로그 시간 기록 여부
    COMMU_LOG_INTERVAL = 10     # 대화 로그 저장 간격

class TTS:
    """TTS (Text-to-Speech) 관련 설정 (GPT-SoVITS)"""
    LOCATION_GPT : str = ""     # GPT 모델 경로
    GPT_NAME : str = ""         # GPT 모델명
    LOCATION_CKPT : str = ""    # SoVITS 체크포인트 경로
    CKPT_NAME : str = ""        # SoVITS 체크포인트명
    
    # 추론 옵션
    TEXT_SPLIT_METHOD : str = "cut5"
    BATCH_SIZE : int = 1
    PARALLEL_INFER : bool = True
    SPLIT_BUCKET : bool = True
    SEED : int = -1
    TOP_K : float = 64
    TOP_P : float = 0.95
    TEMPERATURE : float = 1.0
    REPETITION_PENALTY : float = 1.35
    SPEED_FACTOR : float = 1.0
    TTS_LANGUAGE : str = "auto"
    GPU_TTS = 0                 # TTS 전용 GPU 인덱스
    TTS_ENABLE : bool = False   # TTS 활성화 여부


class EMOTION:
    """감정 분석 모델 설정"""
    LOCATION_EMOTION : str = ""
    EMOTION_NAME : str = ""

class RAG:
    """RAG (Retrieval-Augmented Generation) 서버 설정"""
    RAG_IP: str = ""
    RAG_PORT: int = 8080

class SQL:
    """MySQL 데이터베이스 연결 설정"""
    MYSQL_HOST: str = ""
    MYSQL_DATABASE: str = ""
    MYSQL_USER: str = ""
    MYSQL_PASSWORD: str = ""
    MYSQL_PORT: int = 3306


def _apply(data: dict):
    """딕셔너리 데이터를 각 설정 클래스에 적용합니다."""
    
    # PORTS
    PORTS.LLAMA_HOST = data.get("LLAMA_HOST", "127.0.0.1")
    PORTS.LLAMA_PORT = int(data.get("LLAMA_PORT", 8000))
    PORTS.TTS_HOST = data.get("TTS_HOST", "127.0.0.1")
    PORTS.TTS_PORT = int(data.get("TTS_PORT", 9880))
    PORTS.AUDIO_SERVER_HOST = data.get("AUDIO_SERVER_HOST", "127.0.0.1")
    PORTS.AUDIO_SERVER_PORT = int(data.get("AUDIO_SERVER_PORT", 3546))
    PORTS.MAIN_SERVER_PORT = int(data.get("MAIN_SERVER_PORT", 3545))

    # LLM
    LLM.LOCATION = data.get("LOCATION", "")
    LLM.LOCATION_MODEL = data.get("LOCATION_MODEL", "")
    LLM.MODEL_TYPE = data.get("LLM_MODEL_TYPE", "gemma")

    LLM.CACHE_QUANT = data.get("CACHE_QUANT", "")
    LLM.TENSOR_PARALLEL = data.get("TENSOR_PARALLEL", False)
    LLM.GPU_INDEX = data.get("GPU_INDEX", "0")
    LLM.GPU_SPLIT = data.get("GPU_SPLIT", "")
    
    LLM.TEMPERATURE = data.get("LLM_TEMPERATURE", "0.8")
    LLM.TOP_K = data.get("LLM_TOP_K", "40")
    LLM.TOP_P = data.get("LLM_TOP_P", "0.9")
    LLM.MIN_P = data.get("LLM_MIN_P", "0.05")
    LLM.REPETITION_PENALTY = data.get("LLM_REPETITION_PENALTY", "1.05")
    LLM.PRESENCE_PENALTY = data.get("LLM_PRESENCE_PENALTY", "0.0")
    LLM.FREQUENCY_PENALTY = data.get("LLM_FREQUENCY_PENALTY", "0.0")
    
    LLM.CPU_THREADS = int(data.get("CPU_THREADS", 1))
    LLM.CONTEXT = data.get("CONTEXT", 1024)

    LLM.LLM_CHAT_FORMAT = data.get("CHARACTER_CONCEPT", "")
    LLM.LLM_FEEDBACK_FORMAT = data.get("COMMAND_FEEDBACK", "")
    LLM.LLM_RAG_SEARCH_FORMAT = data.get("COMMAND_SEARCH", "")
    
    LLM.COMMU_LOG_TIME = data.get("COMMU_LOG_TIME", False)
    LLM.COMMU_LOG_INTERVAL = int(data.get("COMMU_LOG_INTERVAL", 10))

    # EMOTION
    EMOTION.EMOTION_NAME = data.get("EMOTION_NAME", "")
    EMOTION.LOCATION_EMOTION = data.get("LOCATION_EMOTION", "")

    # RAG
    RAG.RAG_IP = data.get("RAG_IP", "")
    RAG.RAG_PORT = int(data.get("RAG_PORT", 8080))

    # SQL
    SQL.MYSQL_HOST = data.get("MYSQL_HOST", "")
    SQL.MYSQL_DATABASE = data.get("MYSQL_DATABASE", "")
    SQL.MYSQL_USER = data.get("MYSQL_USER", "")
    SQL.MYSQL_PASSWORD = data.get("MYSQL_PASSWORD", "")
    SQL.MYSQL_PORT = int(data.get("MYSQL_PORT", 3306))

    # TTS
    TTS.LOCATION_GPT = data.get("LOCATION_GPT", "")
    TTS.GPT_NAME = data.get("GPT_NAME", "")
    TTS.LOCATION_CKPT = data.get("LOCATION_CKPT", "")
    TTS.CKPT_NAME = data.get("CKPT_NAME", "")
    TTS.TEXT_SPLIT_METHOD = data.get("TEXT_SPLIT_METHOD", "cut5")
    TTS.BATCH_SIZE = int(data.get("BATCH_SIZE", 1))
    TTS.PARALLEL_INFER = data.get("PARALLEL_INFER", True)
    TTS.SPLIT_BUCKET = data.get("SPLIT_BUCKET", True)
    TTS.SEED = int(data.get("SEED", -1))
    TTS.TOP_K = float(data.get("TOP_K", 64))
    TTS.TOP_P = float(data.get("TOP_P", 0.95))
    TTS.TEMPERATURE = float(data.get("TEMPERATURE", 1.0))
    TTS.REPETITION_PENALTY = float(data.get("REPETITION_PENALTY", 1.35))
    TTS.SPEED_FACTOR = float(data.get("SPEED_FACTOR", 1.0))
    TTS.TTS_LANGUAGE = data.get("TTS_LANGUAGE", "auto")
    TTS.GPU_TTS = data.get("GPU_TTS", 0)
    TTS.TTS_ENABLE = data.get("TTS_ENABLE", False)


def reload(path: str | None = None):
    """
    설정 파일을 다시 로드하여 전역 설정에 반영합니다.
    
    Args:
        path (str, optional): 설정 파일 경로. 지정하지 않으면 기본 경로 사용.
        
    Returns:
        bool: 로드 성공 여부
    """
    cfg = path or DEFAULT_CFG
    if not os.path.isfile(cfg):
        return False
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        _apply(data)
        return True
    except Exception:
        return False


# 모듈 임포트 시 자동으로 설정 로드
reload()
