# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# Core/emotion_analyzer.py
# KoELECTRA 기반의 감정 분석 모델을 사용하여 텍스트의 감정을 분석하는 모듈입니다.
# ──────────────────────────────────────────────────────────────────────────────
import os
# ONNX Runtime 로그 레벨 설정 (3: ERROR, 경고 메시지 숨김)
os.environ["ORT_LOGGING_LEVEL"] = "3"

import logging
import threading
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 로거 설정
logger = logging.getLogger("system.emotion_analyzer")

# ──────────────────────────────────────────────────────────────────────────────
# 전역 변수 및 설정
# ──────────────────────────────────────────────────────────────────────────────
emotion_model = None
emotion_tokenizer = None
model_lock = threading.Lock()

# [최적화] GPU 사용 가능 시 GPU 우선 사용, 아니면 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face 모델 ID (KoELECTRA 기반 감정 분석 모델)
MODEL_ID = "MelissaJ/koelectra-emotion-6-emotion-base"

# 감정 라벨 매핑 (한국어 -> 영어)
# TTS 엔진 등에서 영어 라벨을 사용하는 경우를 대비
EMOTION_MAP_KO_EN = {
    "보통": "neutral",
    "기쁨": "happy",
    "슬픔": "sad",
    "상처": "sad",
    "분노": "angry",
    "불안": "angry",
    "당황": "surprise",
}


# ──────────────────────────────────────────────────────────────────────────────
# 모델 로드 및 유틸리티 함수
# ──────────────────────────────────────────────────────────────────────────────

def load_emotion_model():
    """
    KoELECTRA 감정 분석 모델과 토크나이저를 로드합니다.
    멀티스레드 환경에서 안전하게 한 번만 로드되도록 락(Lock)을 사용합니다.
    """
    global emotion_model, emotion_tokenizer

    if emotion_model is None or emotion_tokenizer is None:
        with model_lock:
            # 락 획득 후 다시 확인 (Double-checked locking)
            if emotion_model is None or emotion_tokenizer is None:
                logger.info(f"🔹 감정 분석 모델 로드 시작 (Device: {device})")
                try:
                    emotion_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                    emotion_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
                    emotion_model.to(device)
                    emotion_model.eval() # 평가 모드로 설정
                    logger.info("✅ 감정 분석 모델 로드 완료!")
                except Exception as e:
                    logger.error(f"❌ 감정 분석 모델 로드 실패: {e}")
                    raise


def _get_id2label() -> Dict[int, str]:
    """
    모델 설정(config)에서 id2label 매핑을 가져옵니다.
    키가 문자열인 경우 정수로 변환하여 반환합니다.
    """
    id2label_raw = getattr(emotion_model.config, "id2label", None) or {}
    id2label = {}
    for k, v in id2label_raw.items():
        try:
            id2label[int(k)] = v
        except ValueError:
            id2label[k] = v
    return id2label


# ──────────────────────────────────────────────────────────────────────────────
# 감정 분석 메인 함수
# ──────────────────────────────────────────────────────────────────────────────

def analyze_emotion(
    text: str,
    return_all_scores: bool = False,
    top_k: int = None,
    max_length: int = 128,
    neutral_threshold: float = 0.6
) -> Union[str, List[Dict]]:
    """
    입력된 텍스트의 감정을 분석합니다.

    Args:
        text (str): 분석할 텍스트
        return_all_scores (bool): True일 경우 모든 감정의 점수를 반환
        top_k (int): 상위 k개의 감정만 반환 (return_all_scores=True일 때 유효)
        max_length (int): 토큰화 최대 길이
        neutral_threshold (float): 최고 점수가 이 값보다 낮으면 'neutral(보통)'으로 분류

    Returns:
        Union[str, List[Dict]]: 
            - 기본: 가장 높은 확률의 감정 라벨 (영어 문자열)
            - return_all_scores=True: 감정별 점수 리스트 [{"label": str, "score": float}, ...]
    """
    global emotion_model, emotion_tokenizer
    
    # 모델이 로드되지 않았다면 로드
    if emotion_model is None or emotion_tokenizer is None:
        load_emotion_model()

    # 1. 텍스트 토큰화
    inputs = emotion_tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    # 2. 모델 추론
    with torch.no_grad():
        logits = emotion_model(**inputs).logits  # [1, num_labels]
        probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_labels]

    id2label = _get_id2label()

    # 3. 결과 반환 (모든 점수 요청 시)
    if return_all_scores:
        results = [
            {"label": id2label.get(i, f"LABEL_{i}"), "score": float(p)} 
            for i, p in enumerate(probs.tolist())
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        return results

    # 4. 결과 반환 (Top-K 요청 시)
    if top_k is not None:
        topk = torch.topk(probs, k=min(top_k, probs.numel()))
        results = [
            {"label": id2label.get(idx, f"LABEL_{idx}"), "score": float(score)}
            for score, idx in zip(topk.values.tolist(), topk.indices.tolist())
        ]
        return results

    # 5. 결과 반환 (기본: 최고 점수 1개)
    max_prob = torch.max(probs).item()
    pred_idx = int(torch.argmax(probs).item())

    # 중립(보통) 감정 처리: 최고 점수가 임계값보다 낮으면 '보통'으로 간주
    if max_prob < neutral_threshold:
        pred_label_ko = "보통"
        logger.info(f"입력: '{text}' -> 예측: {pred_label_ko} (신뢰도: {max_prob:.4f} < {neutral_threshold})")
    else:
        pred_label_ko = id2label.get(pred_idx, f"LABEL_{pred_idx}")
        logger.info(f"입력: '{text}' -> 예측: {pred_label_ko} (신뢰도: {max_prob:.4f})")

    # 최종 결과를 영어로 변환하여 반환
    pred_label_en = EMOTION_MAP_KO_EN.get(pred_label_ko, "neutral")
    return pred_label_en


# ──────────────────────────────────────────────────────────────────────────────
# 테스트 코드 (직접 실행 시)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 로깅 설정 (콘솔 출력)
    logging.basicConfig(level=logging.INFO)
    
    load_emotion_model()

    test_texts = [
        "오늘 날씨가 정말 좋네요! 기분이 상쾌해요.",  # 기쁨
        "안녕하세요? 반갑습니다.",                  # 보통
        "너무 슬픈 영화를 봐서 눈물이 났어요.",      # 슬픔
        "이게 뭐하는 짓이야! 화가 난다!",            # 분노
        "내일 발표인데 너무 떨리고 걱정돼.",          # 불안
    ]

    print("\n===== 감정 분석 테스트 (기본) =====")
    for t in test_texts:
        emo = analyze_emotion(t)
        print(f"'{t}' -> {emo}")

    print("\n===== 상세 점수 확인 =====")
    text_detail = "오늘 저한테 신기한 질문을 많이하셨네요"
    print(f"입력: '{text_detail}'")
    scores = analyze_emotion(text_detail, return_all_scores=True)
    for s in scores:
        print(f"  - {s['label']}: {s['score']:.4f}")

    print("\n===== 임계값 테스트 =====")
    text_ambiguous = "조금 놀라운 소식이네요."
    print(f"입력: '{text_ambiguous}'")
    print(f"  - Threshold 0.5: {analyze_emotion(text_ambiguous, neutral_threshold=0.5)}")
    print(f"  - Threshold 0.8: {analyze_emotion(text_ambiguous, neutral_threshold=0.8)}")
