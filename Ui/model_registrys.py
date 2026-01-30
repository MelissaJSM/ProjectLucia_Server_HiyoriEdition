# ──────────────────────────────────────────────────────────────────────────────
# Ui/model_registrys.py
# 모델 다운로드에 필요한 정보(URL, 체크섬, 파일명 등)를 관리하는 레지스트리입니다.
# UI 코드와 분리되어 순수 데이터만을 다룹니다.
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelRegistry:
    """
    모델 레지스트리 데이터 클래스.
    불변 객체로 설계되어 데이터 무결성을 보장합니다.
    """
    items: List[str]                        # 모델 식별자 목록
    url_map: Dict[str, str]                 # 모델별 다운로드 URL (기본)
    checksum_map: Dict[str, str]            # 모델별 SHA256 체크섬 (기본)
    filename_map: Dict[str, str]            # 저장될 파일명
    mirror_url_map: Optional[Dict[str, str]] = None       # 미러(Uncensored) URL
    mirror_checksum_map: Optional[Dict[str, str]] = None  # 미러(Uncensored) 체크섬


def build_default_registry() -> ModelRegistry:
    """
    기본 모델 레지스트리를 생성하여 반환합니다.
    포함된 모델: Gemma3, Phi4, TTS, Emotion 등
    """
    
    # 1. 모델 식별자 목록 정의
    items = [
        # ── Gemma3 시리즈 ──
        "Gemma3_27b_2_0", "Gemma3_27b_4_0", "Gemma3_27b_6_0", "Gemma3_27b_8_0",
        "Gemma3_12b_2_0", "Gemma3_12b_4_0", "Gemma3_12b_6_0", "Gemma3_12b_8_0",
        "Gemma3_4b_2_0",  "Gemma3_4b_4_0",  "Gemma3_4b_6_0",  "Gemma3_4b_8_0",
        "Gemma3_270m_8_0",
        
        # ── Phi4 시리즈 ──
        "Phi4_2_0", "Phi4_4_0", "Phi4_6_0", "Phi4_8_0",
        
        # ── 기타 모델 ──
        "TTS1",
        "Emotion",
    ]

    # ────────────────────────────────────────────────
    # 2. URL 맵 (기본 버전)
    # ────────────────────────────────────────────────
    url_map = {
        # Gemma3 27B
        "Gemma3_27b_2_0": "https://drive.google.com/file/d/1KHVSC3wQTMv--z7RR-NMEMufkMF0OPmY/view?usp=sharing",
        "Gemma3_27b_4_0": "https://drive.google.com/file/d/15D2Cqm6MbX7lNQlpzLB-CrMXASJCINCk/view?usp=sharing",
        "Gemma3_27b_6_0": "https://drive.google.com/file/d/1SM4lfyLon4gfE425acmCFfPxWKFKcAWw/view?usp=sharing",
        "Gemma3_27b_8_0": "https://drive.google.com/file/d/1mYjg-bJuTQWi39bKaW4fFEbSiLZ1JIyU/view?usp=sharing",
        
        # Gemma3 12B
        "Gemma3_12b_2_0": "https://drive.google.com/file/d/1DsKT4tYCDMwbVj_VeCpv-YCUjoRiAiE1/view?usp=sharing",
        "Gemma3_12b_4_0": "https://drive.google.com/file/d/1avp55oi-9wnBK6PXG4BYghrbqYSikM5o/view?usp=sharing",
        "Gemma3_12b_6_0": "https://drive.google.com/file/d/1tTJka3oUBHHxTLDpY-RrZxLDLT1sk1WS/view?usp=sharing",
        "Gemma3_12b_8_0": "https://drive.google.com/file/d/1V5uT3uveK5fKKeKvhVW0mUGxZoQqBWDX/view?usp=sharing",
        
        # Gemma3 4B
        "Gemma3_4b_2_0": "https://drive.google.com/file/d/1102ZJxcJOxS2UFMB4nOa6Wh92rxWh5pq/view?usp=sharing",
        "Gemma3_4b_4_0": "https://drive.google.com/file/d/1Sb61RG1Nsp2wDa4tb4A9N3-gpn-0u8IE/view?usp=sharing",
        "Gemma3_4b_6_0": "https://drive.google.com/file/d/1ZPcARbgIuCF5GJlhqzboqa6aJoLKethp/view?usp=sharing",
        "Gemma3_4b_8_0": "https://drive.google.com/file/d/1-l3ShqvhuqPKNhnYylBQQzQV_t6Fzf5j/view?usp=sharing",
        
        # Gemma3 270M
        "Gemma3_270m_8_0": "https://drive.google.com/file/d/1tdWfurzNy4dq87x_oW0H806CBuW38FcB/view?usp=sharing",
        
        # Phi4
        "Phi4_2_0": "https://drive.google.com/file/d/1VN7qjc6dMvE4OgGOV9iLA8gaY8FqxFQw/view?usp=sharing",
        "Phi4_4_0": "https://drive.google.com/file/d/1MM-BQAloZ2Dq0yIyBDEDI80kN5iNDS4s/view?usp=sharing",
        "Phi4_6_0": "https://drive.google.com/file/d/1b0IV5GLg_kFu7wUnhyujbxQAhKa0CwEG/view?usp=sharing",
        "Phi4_8_0": "https://drive.google.com/file/d/1WzOlQEoE2qL0Gj1bDtV3BCI3Nq4cib2B/view?usp=sharing",
        
        # Others
        "TTS1": "https://drive.google.com/file/d/179yiCLUt-HJ0S7FNBvNMgSHWYUbqYP8v/view?usp=sharing",
        "Emotion": "MelissaJ/koelectra-emotion-6-emotion-base", # Hugging Face Repo ID
    }

    # ────────────────────────────────────────────────
    # 3. URL 맵 (미러/Uncensored 버전)
    # ────────────────────────────────────────────────
    mirror_url_map = {
        # Gemma3 27B (Uncensored)
        "Gemma3_27b_2_0": "https://drive.google.com/file/d/1xq2aSZ_IevCMC2K2g1PFfyav5I61PNPy/view?usp=sharing",
        "Gemma3_27b_4_0": "https://drive.google.com/file/d/1HP0r5W2rJPiugjx-JSnduyhWKTR9FNGP/view?usp=sharing",
        "Gemma3_27b_6_0": "https://drive.google.com/file/d/1PzUNn4PeJDEuJPgxENdLKpqOqgf_igJ7/view?usp=sharing",
        "Gemma3_27b_8_0": "https://drive.google.com/file/d/17JerrW5H6lGE3agSwLGS4_fvY-gMj52Z/view?usp=sharing",
        
        # Gemma3 12B (Uncensored)
        "Gemma3_12b_2_0": "https://drive.google.com/file/d/1PwAdGiDuh8cHrSZ12q-pYkuHHhr-FjVq/view?usp=sharing",
        "Gemma3_12b_4_0": "https://drive.google.com/file/d/1WrNXz9e-J4lMXYLWNskb7BOrAac75gxn/view?usp=sharing",
        "Gemma3_12b_6_0": "https://drive.google.com/file/d/1zTHne9sV7e0raSvsfMFpy2xYf8efJmOk/view?usp=sharing",
        "Gemma3_12b_8_0": "https://drive.google.com/file/d/10TOHIZkqMidHe5j2snlKkElPSGU8xMwe/view?usp=sharing",
        
        # Gemma3 4B (Uncensored)
        "Gemma3_4b_2_0": "https://drive.google.com/file/d/1RB6fvhwLS-7-TI_VrsUG30CCrty-UD9Y/view?usp=sharing",
        "Gemma3_4b_4_0": "https://drive.google.com/file/d/1p3S0O4kvr2DhX9w-ppeRLb7cfZykCa4b/view?usp=sharing",
        "Gemma3_4b_6_0": "https://drive.google.com/file/d/1B3Iq5BuwWbZo4Cuxu1vJa9M9wN2XLugZ/view?usp=sharing",
        "Gemma3_4b_8_0": "https://drive.google.com/file/d/19MjQdtLuVDt1UNhO4QAKIwa-LyNh-S7n/view?usp=sharing",
        
        # Gemma3 270M (Uncensored 미지원)
        "Gemma3_270m_8_0": "",
        
        # Phi4 (Uncensored)
        "Phi4_2_0": "https://drive.google.com/file/d/1ul7mmru7P7d3P9JMTHXNpD0n-oDUsm0V/view?usp=sharing",
        "Phi4_4_0": "https://drive.google.com/file/d/16PR7hzcQXOHpJDlCu8uoorikU-jHa8Fv/view?usp=sharing",
        "Phi4_6_0": "https://drive.google.com/file/d/1G-J39D5oPick5_dHJcer50Cp1thInB7Z/view?usp=sharing",
        "Phi4_8_0": "https://drive.google.com/file/d/1yvKN9jTTaice3wdsVzRzq2-6-LIvuOHi/view?usp=sharing",
        
        # Others
        "TTS1": "",
        "Emotion": "",
    }

    # ────────────────────────────────────────────────
    # 4. 체크섬 맵 (SHA256) - 원본
    # ────────────────────────────────────────────────
    checksum_map = {
        # Gemma3 27B
        "Gemma3_27b_2_0": "e3ac3b29485f31eedb9de9b5396884e75d7b5e5b226c2adfb5d6c25fb8bbdfe9",
        "Gemma3_27b_4_0": "c204f40df02fbaedd64852d612716c50b90d15c5492316a2194065ad8c42ec03",
        "Gemma3_27b_6_0": "ac63afcf4a75f6bb0f633272fd797146e37ec5dcf6cc7d29d7ce9745f1beee84",
        "Gemma3_27b_8_0": "88cf2455c0686be8028fa71bd6489a38f18c7f2514b0e024d2eadc2713bc82bf",
        
        # Gemma3 12B
        "Gemma3_12b_2_0": "5f8546b2132d4646d16cec92dee6f907229b090b9bbcf0a232f71a6b511622ee",
        "Gemma3_12b_4_0": "f1246309d03dffc126a43e220d3c29ccf6fcf08e888cb1fcfe125d55cb388f38",
        "Gemma3_12b_6_0": "7823b04b971d725b450b578766ada9d6089674cba23909d923b54e45ba922040",
        "Gemma3_12b_8_0": "e11d4424cd0da53ad3ec211d24e090bc60905e378356eb724caf23e521f90263",
        
        # Gemma3 4B
        "Gemma3_4b_2_0": "052a59bdaaff9be958f9b8e460d19958c38cb5c8b4fbfd3b0d179da720cf23f4",
        "Gemma3_4b_4_0": "9b69ae85063d224366f04d2276efa67d0bc4aa3ffa12882e44cd229a63157bb9",
        "Gemma3_4b_6_0": "6fc698edaff517a6fd33cccb13bb59088b7ae443e7f544d6585b08736ec36ea4",
        "Gemma3_4b_8_0": "0de98ad5bfa21c85faf47b49f03d2bd5386f7d29613f0ccef6d1c5f7a6c3bd91",
        
        # Gemma3 270M
        "Gemma3_270m_8_0": "ab50de3ebe0133d523298b4bf1eba5b4ab95ee244202c130195afb21b006d6b6",
        
        # Phi4
        "Phi4_2_0": "f5ea673afdc6ad19f508b9f71397a05517da36dec1721fa67d227de8f9d31f7d",
        "Phi4_4_0": "a3a64dc8c8afafb52d24e53673e3cd14c0567dda5f0698ff179c6f7be75c8ab2",
        "Phi4_6_0": "3fa79d4060f123b24dc22bbf9b5afbcd5ffe1f242c84c12d5d14210bfe233c82",
        "Phi4_8_0": "59719473006b50091fc5f0d9ff11796b5e14926a1124770c5f00c6788208cb97",
        
        # Others
        "TTS1": "ec651b10c051f02423ebc72a4fe5c716ee4ee8f65320a54c34f269643b02474d",
        "Emotion": "",
    }

    # ────────────────────────────────────────────────
    # 5. 체크섬 맵 (SHA256) - 미러/Uncensored
    # ────────────────────────────────────────────────
    mirror_checksum_map = {
        # Gemma3 27B
        "Gemma3_27b_2_0": "7f5671394cc7ffe44d7ebc238e78fc73670212ecbd375bc88570f48438024fce",
        "Gemma3_27b_4_0": "8cced1d169c14cb506e6e91bfcb274b92a97aa0dbcaa8c0d959970d96ff84f08",
        "Gemma3_27b_6_0": "b1a5028d0ccd8aea32da9c791dbf8115e127286b2bc4f49b8ac91d6685810c81",
        "Gemma3_27b_8_0": "2355ec33c6fe294d0a4fc56771ebdc6b40b88693a16bb9ec1c71034d4b52f6be",
        
        # Gemma3 12B
        "Gemma3_12b_2_0": "bc18dfe1a863180b71ad4fad1c251a49b5c305a48653dfe3c892d93795db580c",
        "Gemma3_12b_4_0": "37079609255b8f74c612fe03359f9466f55fc3d3d87741152361fabd3ef69411",
        "Gemma3_12b_6_0": "31cd1c2a29e5a595f5626e7cb474ac15d0b0385bef951c7747c986cb65be2d37",
        "Gemma3_12b_8_0": "c30d38464788061363be365e158b751a57bb099999f3e2aede04fa17c92ca202",
        
        # Gemma3 4B
        "Gemma3_4b_2_0": "a6f43e808d3d3747de99f0cf54d0e5af57dbe379d137f311d8e16d2da05edee2",
        "Gemma3_4b_4_0": "19b417a6c0568f08c1a68863f174d8162005e011f3746a0ca10d6daec3f0b0c7",
        "Gemma3_4b_6_0": "b3c06dc9c8d91620aae9b85a09f08532e0ed39f33639850092d2adea096da5a0",
        "Gemma3_4b_8_0": "8ad94dadb14553940bd27a735df779d7af3d7fdef4900c9331e052f397433882",
        
        # Gemma3 270M
        "Gemma3_270m_8_0": "",
        
        # Phi4
        "Phi4_2_0": "01b1aa5b9595c8e18246cd3cb2305216ca1a82991c785c1a4309243e12a80194",
        "Phi4_4_0": "645bd6b1da6b2242e4e7f0baf2735a95d05eefab9a5bf9006fd88256b5f0a724",
        "Phi4_6_0": "becaf9cbc10cde182643447aff68f59e98300e0acf454f2523ad8cc9aa229916",
        "Phi4_8_0": "49c026443dcf51bcc527e865a9fa9393cf56f9e38a5e7292c6f0a3644104cbe0",
        
        # Others
        "TTS1": "",
        "Emotion": "",
    }

    # ────────────────────────────────────────────────
    # 6. 파일명 맵 (저장될 파일 이름)
    # ────────────────────────────────────────────────
    filename_map = {
        # Gemma3
        "Gemma3_27b_2_0":  "Gemma3-27B-2_0.zip",
        "Gemma3_27b_4_0":  "Gemma3-27B-4_0.zip",
        "Gemma3_27b_6_0":  "Gemma3-27B-6_0.zip",
        "Gemma3_27b_8_0":  "Gemma3-27B-8_0.zip",
        "Gemma3_12b_2_0":  "Gemma3-12B-2_0.zip",
        "Gemma3_12b_4_0":  "Gemma3-12B-4_0.zip",
        "Gemma3_12b_6_0":  "Gemma3-12B-6_0.zip",
        "Gemma3_12b_8_0":  "Gemma3-12B-8_0.zip",
        "Gemma3_4b_2_0":   "Gemma3-4B-2_0.zip",
        "Gemma3_4b_4_0":   "Gemma3-4B-4_0.zip",
        "Gemma3_4b_6_0":   "Gemma3-4B-6_0.zip",
        "Gemma3_4b_8_0":   "Gemma3-4B-8_0.zip",
        "Gemma3_270m_8_0": "Gemma3-270M-8_0.zip",
        
        # Phi4
        "Phi4_2_0":        "Phi4-2_0.zip",
        "Phi4_4_0":        "Phi4-4_0.zip",
        "Phi4_6_0":        "Phi4-6_0.zip",
        "Phi4_8_0":        "Phi4-8_0.zip",
        
        # Others
        "TTS1": "TTS1.zip",
        "Emotion": "Emotion.safetensors",
    }

    return ModelRegistry(
        items=items,
        url_map=url_map,
        checksum_map=checksum_map,
        filename_map=filename_map,
        mirror_url_map=mirror_url_map,
        mirror_checksum_map=mirror_checksum_map,
    )
