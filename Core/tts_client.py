# ──────────────────────────────────────────────────────────────────────────────
# Core/tts_client.py
# GPT-SoVITS 서버와 통신하여 텍스트를 음성으로 변환(TTS)하는 클라이언트 모듈입니다.
# ──────────────────────────────────────────────────────────────────────────────
import os
import threading
import requests
from typing import Optional

# 프로젝트 설정
from Core.server_config import TTS

# ──────────────────────────────────────────────────────────────────────────────
# 설정 및 상수
# ──────────────────────────────────────────────────────────────────────────────

# GPT-SoVITS 서버 URL (환경변수 또는 기본값)
TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "http://127.0.0.1:9880")

# TTS 요청 동시성 제어용 락 (Lock)
# GPU 부하를 방지하기 위해 한 번에 하나의 TTS 변환만 수행하도록 제한합니다.
tts_lock = threading.Lock()

# 감정별 참조 오디오 및 프롬프트 설정
# 경로는 Core/GptSoVits 폴더 기준 상대 경로입니다.
REF_AUDIO_MAP = {
    # 1. 보통 (Neutral) - 기본값
    "neutral": {
        "path": "Core/GptSoVits/Emotions/neutral.wav",
        "text": "ここから先は結果に応じてボイスを選んでいただきます。",
        "lang": "ja"
    },
    # 2. 기쁨 (Happy)
    "happy": {
        "path": "Core/GptSoVits/Emotions/happy.wav",
        "text": "ふわふわしっぽの5番生あなたの心の一番星",
        "lang": "ja"
    },
    # 3. 슬픔/상처 (Sad)
    "sad": {
        "path": "Core/GptSoVits/Emotions/sad.wav",
        "text": "今回は残念でしたね。",
        "lang": "ja"
    },
    # 4. 분노/불안 (Angry)
    "angry": {
        "path": "Core/GptSoVits/Emotions/angry.wav",
        "text": "言ったじゃないですか",
        "lang": "ja"
    },
    # 5. 당황 (Surprise)
    "surprise": {
        "path": "Core/GptSoVits/Emotions/surprise.wav",
        "text": "あーでも無理はダメです",
        "lang": "ja"
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# TTS 변환 함수
# ──────────────────────────────────────────────────────────────────────────────

def text_to_speech(text: str, emotion: str = "neutral") -> Optional[bytes]:
    """
    GPT-SoVITS 서버에 텍스트를 보내 음성 데이터(WAV bytes)를 받아옵니다.
    감정(emotion)에 따라 적절한 참조 오디오를 자동으로 선택합니다.

    Args:
        text (str): 음성으로 변환할 텍스트.
        emotion (str, optional): 감정 키워드 (neutral, happy, sad, angry, surprise). 
                                 기본값은 "neutral".

    Returns:
        Optional[bytes]: 생성된 WAV 오디오 바이너리 데이터. 실패 시 None 반환.
    """
    api_url = f"{TTS_SERVER_URL}/tts"
    
    # 감정 키워드 보정 (한글 -> 영어 매핑이 안 된 경우 대비)
    # emotion_analyzer.py에서 이미 영어로 변환해서 넘겨주지만, 안전장치로 한 번 더 확인
    emotion_key = emotion.lower()
    if emotion_key not in REF_AUDIO_MAP:
        # 한글 키워드가 들어왔을 경우를 위한 매핑 (필요 시)
        ko_to_en = {"보통": "neutral", "기쁨": "happy", "슬픔": "sad", "분노": "angry", "당황": "surprise"}
        emotion_key = ko_to_en.get(emotion, "neutral")

    print(f"🗣️ TTS 요청: '{text}' (감정: {emotion} -> {emotion_key})")

    # 감정에 맞는 참조 오디오 설정 가져오기
    ref_config = REF_AUDIO_MAP.get(emotion_key, REF_AUDIO_MAP["neutral"])

    ref_path = ref_config["path"]
    prompt_text = ref_config["text"]
    prompt_lang = ref_config["lang"]

    # 참조 오디오 경로를 절대 경로로 변환
    if not os.path.isabs(ref_path):
        # 현재 파일(Core/tts_client.py)의 상위 폴더(ProjectLucia_Server)를 기준으로 경로 설정
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ref_path = os.path.join(project_root, ref_path)

    # 파일 존재 여부 확인 (디버깅용)
    if not os.path.exists(ref_path):
        print(f"⚠️ 경고: 참조 오디오 파일을 찾을 수 없습니다: {ref_path}")
        # 파일이 없으면 neutral로 한 번 더 시도 (안전장치)
        if emotion_key != "neutral":
            print("⚠️ 기본(neutral) 참조 오디오로 대체합니다.")
            ref_config = REF_AUDIO_MAP["neutral"]
            ref_path = os.path.join(project_root, ref_config["path"])
            prompt_text = ref_config["text"]
            prompt_lang = ref_config["lang"]

    # 요청 페이로드 구성
    payload = {
        "text": text,
        "text_lang": TTS.TTS_LANGUAGE,
        "ref_audio_path": ref_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "batch_size" : TTS.BATCH_SIZE,
        "parallel_infer" : TTS.PARALLEL_INFER,
        "split_bucket" : TTS.SPLIT_BUCKET,
        "seed" : TTS.SEED,
        "repetition_penalty" : TTS.REPETITION_PENALTY,
        "speed_factor" : TTS.SPEED_FACTOR, # 대소문자 주의 (서버 API에 맞춤)
        "top_k": TTS.TOP_K,
        "top_p": TTS.TOP_P,
        "temperature": TTS.TEMPERATURE,
        "text_split_method": TTS.TEXT_SPLIT_METHOD,
        "is_half" : True,
        "streaming_mode": False,
        "media_type": "wav"
    }

    try:
        # TTS 요청 시 락을 걸어 동시 실행 방지 (GPU 보호)
        with tts_lock:
            response = requests.post(api_url, json=payload, timeout=60)

        if response.status_code == 200 and response.headers.get('Content-Type') == 'audio/wav':
            # 성공 시 오디오 데이터 반환
            return response.content
        else:
            print(f"❌ TTS 서버 오류 ({response.status_code}): {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ TTS 통신 실패: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 테스트 코드 (직접 실행 시)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 테스트용 텍스트 및 감정
    test_text = "안녕하세요? GPT-SoVITS 테스트입니다. 목소리가 잘 들리시나요?"
    test_emotion = "happy"

    print(f"🧪 TTS 테스트 시작: '{test_text}' ({test_emotion})")
    wav_bytes = text_to_speech(test_text, test_emotion)

    if wav_bytes:
        output_filename = "tts_test_output.wav"
        print(f"✅ 음성 생성 완료! ({len(wav_bytes)} bytes)")
        with open(output_filename, "wb") as f:
            f.write(wav_bytes)
        print(f"💾 파일 저장됨: {output_filename}")
    else:
        print("❌ 음성 생성 실패.")
