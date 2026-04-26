# ──────────────────────────────────────────────────────────────────────────────
# Ui/model_registrys.py
# 모델 다운로드에 필요한 정보(Hugging Face 리포지토리 ID)를 관리하는 레지스트리입니다.
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelRegistry:
    items: List[str]  # 모델 식별자 목록
    url_map: Dict[str, str]  # 허깅페이스 원본 리포지토리 ID
    mirror_url_map: Optional[Dict[str, str]] = None  # 허깅페이스 Uncensored 리포지토리 ID


def build_default_registry() -> ModelRegistry:
    items = [
        "Gemma3_27b_2_0", "Gemma3_27b_4_0", "Gemma3_27b_6_0", "Gemma3_27b_8_0",
        "Gemma3_12b_2_0", "Gemma3_12b_4_0", "Gemma3_12b_6_0", "Gemma3_12b_8_0",
        "Gemma3_4b_2_0", "Gemma3_4b_4_0", "Gemma3_4b_6_0", "Gemma3_4b_8_0",
        "Gemma3_270m_8_0",
        "Phi4_2_0", "Phi4_4_0", "Phi4_6_0", "Phi4_8_0",
        "TTS1", "Emotion",
    ]

    # 1. 원본 모델 리포지토리 ID (Hugging Face)
    url_map = {
        "Gemma3_27b_2_0": "LuciaValentine/Gemma3-hiyori-27B-2_0",
        "Gemma3_27b_4_0": "LuciaValentine/Gemma3-hiyori-27B-4_0",
        "Gemma3_27b_6_0": "LuciaValentine/Gemma3-hiyori-27B-6_0",
        "Gemma3_27b_8_0": "LuciaValentine/Gemma3-hiyori-27B-8_0",

        "Gemma3_12b_2_0": "LuciaValentine/Gemma3-hiyori-12B-2_0",
        "Gemma3_12b_4_0": "LuciaValentine/Gemma3-hiyori-12B-4_0",
        "Gemma3_12b_6_0": "LuciaValentine/Gemma3-hiyori-12B-6_0",
        "Gemma3_12b_8_0": "LuciaValentine/Gemma3-hiyori-12B-8_0",

        "Gemma3_4b_2_0": "LuciaValentine/Gemma3-hiyori-4B-2_0",
        "Gemma3_4b_4_0": "LuciaValentine/Gemma3-hiyori-4B-4_0",
        "Gemma3_4b_6_0": "LuciaValentine/Gemma3-hiyori-4B-6_0",
        "Gemma3_4b_8_0": "LuciaValentine/Gemma3-hiyori-4B-8_0",

        "Gemma3_270m_8_0": "LuciaValentine/Gemma3-hiyori-270M-8_0",

        "Phi4_2_0": "LuciaValentine/Phi4-2_0",
        "Phi4_4_0": "LuciaValentine/Phi4-4_0",
        "Phi4_6_0": "LuciaValentine/Phi4-6_0",
        "Phi4_8_0": "LuciaValentine/Phi4-8_0",

        "TTS1": "LuciaValentine/TTS1-hiyori-Model",
        "Emotion": "MelissaJ/koelectra-emotion-6-emotion-base",
    }

    # 2. 검열 해제(Uncensored) 모델 리포지토리 ID (Hugging Face)
    mirror_url_map = {
        "Gemma3_27b_2_0": "LuciaValentine/Gemma3-hiyori-27B-2_0-Uncensored",
        "Gemma3_27b_4_0": "LuciaValentine/Gemma3-hiyori-27B-4_0-Uncensored",
        "Gemma3_27b_6_0": "LuciaValentine/Gemma3-hiyori-27B-6_0-Uncensored",
        "Gemma3_27b_8_0": "LuciaValentine/Gemma3-hiyori-27B-8_0-Uncensored",

        "Gemma3_12b_2_0": "LuciaValentine/Gemma3-hiyori-12B-2_0-Uncensored",
        "Gemma3_12b_4_0": "LuciaValentine/Gemma3-hiyori-12B-4_0-Uncensored",
        "Gemma3_12b_6_0": "LuciaValentine/Gemma3-hiyori-12B-6_0-Uncensored",
        "Gemma3_12b_8_0": "LuciaValentine/Gemma3-hiyori-12B-8_0-Uncensored",

        "Gemma3_4b_2_0": "LuciaValentine/Gemma3-hiyori-4B-2_0-Uncensored",
        "Gemma3_4b_4_0": "LuciaValentine/Gemma3-hiyori-4B-4_0-Uncensored",
        "Gemma3_4b_6_0": "LuciaValentine/Gemma3-hiyori-4B-6_0-Uncensored",
        "Gemma3_4b_8_0": "LuciaValentine/Gemma3-hiyori-4B-8_0-Uncensored",

        "Gemma3_270m_8_0": "",  # Uncensored 미지원

        "Phi4_2_0": "LuciaValentine/Phi4-2_0-Uncensored",
        "Phi4_4_0": "LuciaValentine/Phi4-4_0-Uncensored",
        "Phi4_6_0": "LuciaValentine/Phi4-6_0-Uncensored",
        "Phi4_8_0": "LuciaValentine/Phi4-8_0-Uncensored",

        "TTS1": "",
        "Emotion": "",
    }

    return ModelRegistry(
        items=items,
        url_map=url_map,
        mirror_url_map=mirror_url_map
    )