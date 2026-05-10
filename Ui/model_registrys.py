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


def build_default_registry() -> ModelRegistry:
    items = [
        "Gemma3_27b_2_0", "Gemma3_27b_4_0", "Gemma3_27b_6_0", "Gemma3_27b_8_0",
        "Gemma3_12b_2_0", "Gemma3_12b_4_0", "Gemma3_12b_6_0", "Gemma3_12b_8_0",
        "Gemma3_4b_2_0", "Gemma3_4b_4_0", "Gemma3_4b_6_0", "Gemma3_4b_8_0",
        "Gemma3_1b_2_0", "Gemma3_1b_4_0", "Gemma3_1b_6_0", "Gemma3_1b_8_0",
        "Gemma3_270m_2_0", "Gemma3_270m_4_0", "Gemma3_270m_6_0", "Gemma3_270m_8_0",
        "Gemma4_31b_2_0", "Gemma4_31b_4_0", "Gemma4_31b_6_0", "Gemma4_31b_8_0",
        "Gemma4_26b_a4b_2_0", "Gemma4_26b_a4b_4_0", "Gemma4_26b_a4b_6_0", "Gemma4_26b_a4b_8_0",

        "Gemma3_27b_2_0_Origin", "Gemma3_27b_4_0_Origin", "Gemma3_27b_6_0_Origin", "Gemma3_27b_8_0_Origin",
        "Gemma3_12b_2_0_Origin", "Gemma3_12b_4_0_Origin", "Gemma3_12b_6_0_Origin", "Gemma3_12b_8_0_Origin",
        "Gemma3_4b_2_0_Origin", "Gemma3_4b_4_0_Origin", "Gemma3_4b_6_0_Origin", "Gemma3_4b_8_0_Origin",
        "Gemma3_1b_2_0_Origin", "Gemma3_1b_4_0_Origin", "Gemma3_1b_6_0_Origin", "Gemma3_1b_8_0_Origin",
        "Gemma3_270m_2_0_Origin", "Gemma3_270m_4_0_Origin", "Gemma3_270m_6_0_Origin", "Gemma3_270m_8_0_Origin",
        "Gemma4_31b_2_0_Origin", "Gemma4_31b_4_0_Origin", "Gemma4_31b_6_0_Origin", "Gemma4_31b_8_0_Origin",
        "Gemma4_26b_a4b_2_0_Origin", "Gemma4_26b_a4b_4_0_Origin", "Gemma4_26b_a4b_6_0_Origin",
        "Gemma4_26b_a4b_8_0_Origin",

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

        # 미완성
        "Gemma3_1b_2_0": "LuciaValentine/hiyori_gemma3_1b_exllama3_2.0bpw",
        "Gemma3_1b_4_0": "LuciaValentine/hiyori_gemma3_1b_exllama3_4.0bpw",
        "Gemma3_1b_6_0": "LuciaValentine/hiyori_gemma3_1b_exllama3_6.0bpw",
        "Gemma3_1b_8_0": "LuciaValentine/hiyori_gemma3_1b_exllama3_8.0bpw",

        # 미완성
        "Gemma3_270m_2_0": "LuciaValentine/hiyori_gemma3_270m_exllama3_2.0bpw",
        "Gemma3_270m_4_0": "LuciaValentine/hiyori_gemma3_270m_exllama3_4.0bpw",
        "Gemma3_270m_6_0": "LuciaValentine/hiyori_gemma3_270m_exllama3_6.0bpw",
        "Gemma3_270m_8_0": "LuciaValentine/hiyori_gemma3_270m_exllama3_8.0bpw",

        "Gemma4_31b_2_0": "LuciaValentine/hiyori_gemma4_31b_exllama3_gemma-4_exllama3_2.0bpw",
        "Gemma4_31b_4_0": "LuciaValentine/hiyori_gemma4_31b_exllama3_gemma-4_exllama3_4.0bpw",
        "Gemma4_31b_6_0": "LuciaValentine/hiyori_gemma4_31b_exllama3_gemma-4_exllama3_6.0bpw",
        "Gemma4_31b_8_0": "LuciaValentine/hiyori_gemma4_31b_exllama3_gemma-4_exllama3_8.0bpw",

        "Gemma4_26b_a4b_2_0": "LuciaValentine/hiyori_gemma4_26b_a4b_exllama3_gemma-4_exllama3_2.0bpw",
        "Gemma4_26b_a4b_4_0": "LuciaValentine/hiyori_gemma4_26b_a4b_exllama3_gemma-4_exllama3_4.0bpw",
        "Gemma4_26b_a4b_6_0": "LuciaValentine/hiyori_gemma4_26b_a4b_exllama3_gemma-4_exllama3_6.0bpw",
        "Gemma4_26b_a4b_8_0": "LuciaValentine/hiyori_gemma4_26b_a4b_exllama3_gemma-4_exllama3_8.0bpw",


        #오리지널 모델 리스트
        "Gemma3_27b_2_0_Origin": "LuciaValentine/origin_gemma3_27b_exllama3_2.0bpw",
        "Gemma3_27b_4_0_Origin": "LuciaValentine/origin_gemma3_27b_exllama3_4.0bpw",
        "Gemma3_27b_6_0_Origin": "LuciaValentine/origin_gemma3_27b_exllama3_6.0bpw",
        "Gemma3_27b_8_0_Origin": "LuciaValentine/origin_gemma3_27b_exllama3_8.0bpw",

        "Gemma3_12b_2_0_Origin": "LuciaValentine/origin_gemma3_12b_exllama3_2.0bpw",
        "Gemma3_12b_4_0_Origin": "LuciaValentine/origin_gemma3_12b_exllama3_4.0bpw",
        "Gemma3_12b_6_0_Origin": "LuciaValentine/origin_gemma3_12b_exllama3_6.0bpw",
        "Gemma3_12b_8_0_Origin": "LuciaValentine/origin_gemma3_12b_exllama3_8.0bpw",

        "Gemma3_4b_2_0_Origin": "LuciaValentine/origin_gemma3_4b_exllama3_2.0bpw",
        "Gemma3_4b_4_0_Origin": "LuciaValentine/origin_gemma3_4b_exllama3_4.0bpw",
        "Gemma3_4b_6_0_Origin": "LuciaValentine/origin_gemma3_4b_exllama3_6.0bpw",
        "Gemma3_4b_8_0_Origin": "LuciaValentine/origin_gemma3_4b_exllama3_8.0bpw",

        "Gemma3_1b_2_0_Origin": "LuciaValentine/origin_gemma3_1b_exllama3_2.0bpw",
        "Gemma3_1b_4_0_Origin": "LuciaValentine/origin_gemma3_1b_exllama3_4.0bpw",
        "Gemma3_1b_6_0_Origin": "LuciaValentine/origin_gemma3_1b_exllama3_6.0bpw",
        "Gemma3_1b_8_0_Origin": "LuciaValentine/origin_gemma3_1b_exllama3_8.0bpw",

        "Gemma3_270m_2_0_Origin": "LuciaValentine/origin_gemma3_270m_exllama3_2.0bpw",
        "Gemma3_270m_4_0_Origin": "LuciaValentine/origin_gemma3_270m_exllama3_4.0bpw",
        "Gemma3_270m_6_0_Origin": "LuciaValentine/origin_gemma3_270m_exllama3_6.0bpw",
        "Gemma3_270m_8_0_Origin": "LuciaValentine/origin_gemma3_270m_exllama3_8.0bpw",

        "Gemma4_31b_2_0_Origin": "LuciaValentine/origin_gemma4_31b_exllama3_2.0bpw",
        "Gemma4_31b_4_0_Origin": "LuciaValentine/origin_gemma4_31b_exllama3_4.0bpw",
        "Gemma4_31b_6_0_Origin": "LuciaValentine/origin_gemma4_31b_exllama3_6.0bpw",
        "Gemma4_31b_8_0_Origin": "LuciaValentine/origin_gemma4_31b_exllama3_8.0bpw",

        "Gemma4_26b_a4b_2_0_Origin": "LuciaValentine/origin_gemma4_26b_a4b_exllama3_2.0bpw",
        "Gemma4_26b_a4b_4_0_Origin": "LuciaValentine/origin_gemma4_26b_a4b_exllama3_4.0bpw",
        "Gemma4_26b_a4b_6_0_Origin": "LuciaValentine/origin_gemma4_26b_a4b_exllama3_6.0bpw",
        "Gemma4_26b_a4b_8_0_Origin": "LuciaValentine/origin_gemma4_26b_a4b_exllama3_8.0bpw",


        "TTS1": "LuciaValentine/TTS1-hiyori-Model",
        "Emotion": "MelissaJ/koelectra-emotion-6-emotion-base",
    }

    return ModelRegistry(
        items=items,
        url_map=url_map,
    )