# Project Lucia Server

**Project Lucia Server**는 실시간 멀티모달 AI 비서 'Lucia'를 위한 고성능 로컬 백엔드 서버 및 관리 도구입니다.  
LLM(대규모 언어 모델), TTS(음성 합성), Vision(화면 인식), 감정 분석 기능을 통합하여 사용자에게 자연스럽고 생동감 있는 상호작용을 제공합니다.

모든 AI 추론은 로컬 GPU 자원을 활용하여 수행되며, PyQt5 기반의 GUI를 통해 서버 상태를 모니터링하고 제어할 수 있습니다.

---

## 📚 목차
1. [소개](#-소개)
2. [주요 기능](#-주요-기능)
3. [기술 스택](#-기술-스택)
4. [설치 방법](#-설치-방법)
5. [사용 방법](#-사용-방법)
6. [프로젝트 구조](#-프로젝트-구조)

---

## 📝 소개

이 프로젝트는 외부 API 의존도를 최소화하고, 개인정보 보호와 빠른 응답 속도를 위해 **On-Premise(로컬)** 환경에서 동작하도록 설계되었습니다.  
사용자의 화면을 실시간으로 인식하여 상황에 맞는 대화를 주도하거나, 사용자의 질문에 대해 웹 검색(RAG)을 통해 최신 정보를 답변합니다.

---

## ✨ 주요 기능

*   **통합 제어 패널 (GUI)**: PyQt5로 제작된 대시보드에서 서버 시작/중지, 하드웨어(GPU/CPU) 모니터링, 로그 확인, 설정 변경이 가능합니다.
*   **고성능 로컬 LLM**: `ExLlamaV3` 백엔드를 사용하여 Gemma 3, Phi-4 등의 최신 모델을 고속으로 추론합니다. (멀티 GPU 및 텐서 병렬화 지원)
*   **감성적인 TTS**: `GPT-SoVITS`를 활용하여 텍스트의 감정(기쁨, 슬픔, 분노 등)을 분석하고 그에 맞는 톤으로 음성을 생성합니다.
*   **실시간 화면 인식 (Vision)**: 사용자의 화면을 캡처하여 상황을 분석하고, AI가 먼저 말을 거는 능동적인 상호작용을 지원합니다.
*   **하이브리드 RAG 검색**: `SearXNG`(Docker) 또는 `DuckDuckGo`를 통해 최신 웹 정보를 검색하여 답변의 정확도를 높입니다.
*   **모델 관리자**: GUI 내에서 필요한 AI 모델 파일을 손쉽게 다운로드하고 관리할 수 있습니다.

---

## 🛠 기술 스택

### Core
*   **Language**: Python 3.10+
*   **Framework**: FastAPI (WebSocket & REST API)
*   **Database**: MySQL (대화 로그 및 설정 저장)

### AI & ML
*   **LLM Backend**: ExLlamaV3 (Gemma 3, Phi-4 지원)
*   **TTS Engine**: GPT-SoVITS
*   **Emotion Analysis**: KoELECTRA (Hugging Face Transformers)
*   **Vision**: Multimodal LLM Integration

### GUI & Tools
*   **Interface**: PyQt5 (Qt Designer)
*   **Process Management**: QProcess, psutil
*   **Search**: SearXNG, DuckDuckGo (ddgs)

---

## 📥 설치 방법

### 1. 필수 요구 사항
*   **OS**: Ubuntu 22.04 (Windows 호환 가능)
*   **GPU**: NVIDIA GPU (CUDA 12.8+ 권장, VRAM 12GB 이상 권장)
*   **Database**: MySQL Server 설치 및 실행 필요

### 2. 클론 및 패키지 설치
```bash
git clone https://github.com/MelissaJSM/ProjectLucia_Server_HiyoriEdition.git
cd ProjectLucia_Server_HiyoriEdition

#가상 환경 생성 (아나콘다 혹은 직접)

# 의존성 설치
pip install uv

uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

uv pip install notebook ipywidgets hf_xet wordsegment python-multipart PyQt5 pytz flask ddgs nvidia-ml-py trafilatura mysql-connector-python fastapi transformers soundfile "uvicorn[standard]" ffmpeg-python librosa pytorch_lightning matplotlib x_transformers peft jieba fast_langdetect g2p_en split_lang cn2an pypinyin jieba_fast pyopenjtalk jamo ko_pron g2pk2 python-mecab-ko onnxruntime-gpu opencc

windows 의 경우
uv pip install https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
uv pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.18/exllamav3-0.0.18+cu128.torch2.8.0-cp310-cp310-win_amd64.whl

linux 의 경우
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
uv pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.18/exllamav3-0.0.18+cu128.torch2.8.0-cp310-cp310-linux_x86_64.whl
```

### 3. 추가 라이브러리 설치
[(라이브러리)](https://drive.google.com/file/d/1KUQ5REXQrKPENT60B25J7EOHpDOrtT3L/view?usp=sharing) 를 다운로드하여 프로젝트 최상단 폴더에 Core 폴더째로 압축을 해제합니다.
- 혹은 Release 에서 모든 파일을 다운받으시면 해당과정을 스킵해도됩니다.

### 3. 데이터베이스 설정
MySQL을 설치 한 후 DB.SQL 파일을 임포트하여 구성합니다.

### 4. SURXNG 설치
[(SURXNG)](https://docs.searxng.org/admin/installation.html) 에서 다운로드 이후 연결을 하면 surxng 기반 검색을, 미설치시 DuckDuckGo 기반으로 검색을 시도합니다.

### 0. Docker 환경 설치
[(Finetuning)](https://github.com/MelissaJSM/ProjectLucia_Finetuning.git) 의 링크에서 Docker 설치환경을 그대로 따라하시면 됩니다. 
서버와 파인튜닝 Docker 환경을 공유합니다.

---

## 🚀 사용 방법

### 1. 관리 도구 실행
프로젝트 루트에서 다음 명령어로 GUI를 실행합니다.
```bash
python ui_main.py
```

### 2. 초기 설정
1.  **설정 탭**: MySQL 접속 정보를 입력하고 테스트를 눌러서 접속 확인 후 `적용`을 누릅니다.
2.  **모델 다운로드**: `Download` 탭에서 필요한 LLM(Gemma3 등)과 TTS 모델, 감정 분석 모델을 다운로드합니다.
3.  **하드웨어 설정**: 사용할 GPU 인덱스와 VRAM 할당량을 설정합니다.

### 3. 서버 시작
*   메인 화면의 **Start** 버튼을 클릭합니다.
*   `Main Server`, `LLM Server`, `TTS Server`가 순차적으로 부팅됩니다.
*   상태 표시등이 모두 **초록색(Running)**으로 바뀌면 준비 완료입니다.

### 4. 클라이언트 연결
*   Unity에서 자동으로 연결을 받습니다.

---

## 📂 프로젝트 구조

```
ProjectLucia_Server/
├── ui_main.py             # 프로그램 진입점 (GUI 실행)
├── config.json            # 서버 설정 파일 (자동 생성됨)
├── Core/                  # 핵심 로직 (AI, DB, 통신)
│   ├── emotion_analyzer.py  # 감정 분석 (KoELECTRA)
│   ├── llm_handler.py       # LLM 추론 및 프롬프트 관리
│   ├── rag_search.py        # 웹 검색 (RAG) 로직
│   ├── server_config.py     # 전역 설정 클래스
│   ├── sql.py               # MySQL 데이터베이스 관리
│   ├── tts_client.py        # GPT-SoVITS 클라이언트
│   ├── runtime_control.py   # 프로세스 재시작 제어
│   ├── ExLlamaV3/           # (Submodule) LLM 백엔드
│   └── GptSoVits/           # (Submodule) TTS 백엔드
└── Ui/                    # UI 및 서버 제어 로직
    ├── controllers.py       # UI 이벤트 핸들러 및 로직
    ├── server_control.py    # 백그라운드 프로세스 관리
    ├── ws_server.py         # FastAPI WebSocket 서버
    ├── download_task.py     # 모델 다운로드 스레드
    ├── model_registrys.py   # 지원 모델 목록 및 URL
    └── untitled.ui          # Qt Designer UI 파일
```

---

## ⚠️ 주의 사항
*   **포트 충돌**: 기본 포트(3545, 3546, 8000, 9880)가 사용 중인지 확인하세요.
*   **GPU 메모리**: LLM과 TTS 모델을 동시에 로드하므로 충분한 VRAM이 필요합니다.

---
# 📥 모델 수동 다운로드 가이드 (Model Manual Download)

자동 다운로드 실패 시, 아래 링크를 통해 모델을 직접 다운로드해 주세요.
다운로드한 파일은 반드시 **파일명 (Filename)** 과 동일하게 이름을 변경한 뒤 모델 폴더에 넣어주셔야 합니다.

> **주의:** 동일한 모델에 대해 **Standard(일반)** 버전과 **Uncensored(검열 해제)** 버전 중 하나만 선택하여 다운로드하세요. (파일명이 동일하므로 덮어씌워집니다.)

---

## 1. Gemma3 Series (Standard)
구글(Google)의 공식 Gemma3 모델 기반입니다.

| 모델 (Model) | 퀀텀 (Quant) | 파일명 (Filename) | 다운로드 (Link) | SHA256 Checksum |
|:---:|:---:|:---|:---:|:---|
| **27B** | 2.0 | `Gemma3-27B-2_0.zip` | [Download](https://drive.google.com/file/d/1KHVSC3wQTMv--z7RR-NMEMufkMF0OPmY/view?usp=sharing) | `e3ac3b29485f31eedb9de9b5396884e75d7b5e5b226c2adfb5d6c25fb8bbdfe9` |
| **27B** | 4.0 | `Gemma3-27B-4_0.zip` | [Download](https://drive.google.com/file/d/15D2Cqm6MbX7lNQlpzLB-CrMXASJCINCk/view?usp=sharing) | `c204f40df02fbaedd64852d612716c50b90d15c5492316a2194065ad8c42ec03` |
| **27B** | 6.0 | `Gemma3-27B-6_0.zip` | [Download](https://drive.google.com/file/d/1SM4lfyLon4gfE425acmCFfPxWKFKcAWw/view?usp=sharing) | `ac63afcf4a75f6bb0f633272fd797146e37ec5dcf6cc7d29d7ce9745f1beee84` |
| **27B** | 8.0 | `Gemma3-27B-8_0.zip` | [Download](https://drive.google.com/file/d/1mYjg-bJuTQWi39bKaW4fFEbSiLZ1JIyU/view?usp=sharing) | `88cf2455c0686be8028fa71bd6489a38f18c7f2514b0e024d2eadc2713bc82bf` |
| **12B** | 2.0 | `Gemma3-12B-2_0.zip` | [Download](https://drive.google.com/file/d/1DsKT4tYCDMwbVj_VeCpv-YCUjoRiAiE1/view?usp=sharing) | `5f8546b2132d4646d16cec92dee6f907229b090b9bbcf0a232f71a6b511622ee` |
| **12B** | 4.0 | `Gemma3-12B-4_0.zip` | [Download](https://drive.google.com/file/d/1avp55oi-9wnBK6PXG4BYghrbqYSikM5o/view?usp=sharing) | `f1246309d03dffc126a43e220d3c29ccf6fcf08e888cb1fcfe125d55cb388f38` |
| **12B** | 6.0 | `Gemma3-12B-6_0.zip` | [Download](https://drive.google.com/file/d/1tTJka3oUBHHxTLDpY-RrZxLDLT1sk1WS/view?usp=sharing) | `7823b04b971d725b450b578766ada9d6089674cba23909d923b54e45ba922040` |
| **12B** | 8.0 | `Gemma3-12B-8_0.zip` | [Download](https://drive.google.com/file/d/1V5uT3uveK5fKKeKvhVW0mUGxZoQqBWDX/view?usp=sharing) | `e11d4424cd0da53ad3ec211d24e090bc60905e378356eb724caf23e521f90263` |
| **4B** | 2.0 | `Gemma3-4B-2_0.zip` | [Download](https://drive.google.com/file/d/1102ZJxcJOxS2UFMB4nOa6Wh92rxWh5pq/view?usp=sharing) | `052a59bdaaff9be958f9b8e460d19958c38cb5c8b4fbfd3b0d179da720cf23f4` |
| **4B** | 4.0 | `Gemma3-4B-4_0.zip` | [Download](https://drive.google.com/file/d/1Sb61RG1Nsp2wDa4tb4A9N3-gpn-0u8IE/view?usp=sharing) | `9b69ae85063d224366f04d2276efa67d0bc4aa3ffa12882e44cd229a63157bb9` |
| **4B** | 6.0 | `Gemma3-4B-6_0.zip` | [Download](https://drive.google.com/file/d/1ZPcARbgIuCF5GJlhqzboqa6aJoLKethp/view?usp=sharing) | `6fc698edaff517a6fd33cccb13bb59088b7ae443e7f544d6585b08736ec36ea4` |
| **4B** | 8.0 | `Gemma3-4B-8_0.zip` | [Download](https://drive.google.com/file/d/1-l3ShqvhuqPKNhnYylBQQzQV_t6Fzf5j/view?usp=sharing) | `0de98ad5bfa21c85faf47b49f03d2bd5386f7d29613f0ccef6d1c5f7a6c3bd91` |
| **270M** | 8.0 | `Gemma3-270M-8_0.zip`| [Download](https://drive.google.com/file/d/1tdWfurzNy4dq87x_oW0H806CBuW38FcB/view?usp=sharing) | `ab50de3ebe0133d523298b4bf1eba5b4ab95ee244202c130195afb21b006d6b6` |

---

## 2. Gemma3 Series (Uncensored / Mirror)
검열이 해제된 튜닝 버전입니다. **파일명은 Standard와 동일합니다.**

| 모델 (Model) | 퀀텀 (Quant) | 파일명 (Filename) | 다운로드 (Link) | SHA256 Checksum |
|:---:|:---:|:---|:---:|:---|
| **27B** | 2.0 | `Gemma3-27B-2_0.zip` | [Download](https://drive.google.com/file/d/1xq2aSZ_IevCMC2K2g1PFfyav5I61PNPy/view?usp=sharing) | `7f5671394cc7ffe44d7ebc238e78fc73670212ecbd375bc88570f48438024fce` |
| **27B** | 4.0 | `Gemma3-27B-4_0.zip` | [Download](https://drive.google.com/file/d/1HP0r5W2rJPiugjx-JSnduyhWKTR9FNGP/view?usp=sharing) | `8cced1d169c14cb506e6e91bfcb274b92a97aa0dbcaa8c0d959970d96ff84f08` |
| **27B** | 6.0 | `Gemma3-27B-6_0.zip` | [Download](https://drive.google.com/file/d/1PzUNn4PeJDEuJPgxENdLKpqOqgf_igJ7/view?usp=sharing) | `b1a5028d0ccd8aea32da9c791dbf8115e127286b2bc4f49b8ac91d6685810c81` |
| **27B** | 8.0 | `Gemma3-27B-8_0.zip` | [Download](https://drive.google.com/file/d/17JerrW5H6lGE3agSwLGS4_fvY-gMj52Z/view?usp=sharing) | `2355ec33c6fe294d0a4fc56771ebdc6b40b88693a16bb9ec1c71034d4b52f6be` |
| **12B** | 2.0 | `Gemma3-12B-2_0.zip` | [Download](https://drive.google.com/file/d/1PwAdGiDuh8cHrSZ12q-pYkuHHhr-FjVq/view?usp=sharing) | `bc18dfe1a863180b71ad4fad1c251a49b5c305a48653dfe3c892d93795db580c` |
| **12B** | 4.0 | `Gemma3-12B-4_0.zip` | [Download](https://drive.google.com/file/d/1WrNXz9e-J4lMXYLWNskb7BOrAac75gxn/view?usp=sharing) | `37079609255b8f74c612fe03359f9466f55fc3d3d87741152361fabd3ef69411` |
| **12B** | 6.0 | `Gemma3-12B-6_0.zip` | [Download](https://drive.google.com/file/d/1zTHne9sV7e0raSvsfMFpy2xYf8efJmOk/view?usp=sharing) | `31cd1c2a29e5a595f5626e7cb474ac15d0b0385bef951c7747c986cb65be2d37` |
| **12B** | 8.0 | `Gemma3-12B-8_0.zip` | [Download](https://drive.google.com/file/d/10TOHIZkqMidHe5j2snlKkElPSGU8xMwe/view?usp=sharing) | `c30d38464788061363be365e158b751a57bb099999f3e2aede04fa17c92ca202` |
| **4B** | 2.0 | `Gemma3-4B-2_0.zip` | [Download](https://drive.google.com/file/d/1RB6fvhwLS-7-TI_VrsUG30CCrty-UD9Y/view?usp=sharing) | `a6f43e808d3d3747de99f0cf54d0e5af57dbe379d137f311d8e16d2da05edee2` |
| **4B** | 4.0 | `Gemma3-4B-4_0.zip` | [Download](https://drive.google.com/file/d/1p3S0O4kvr2DhX9w-ppeRLb7cfZykCa4b/view?usp=sharing) | `19b417a6c0568f08c1a68863f174d8162005e011f3746a0ca10d6daec3f0b0c7` |
| **4B** | 6.0 | `Gemma3-4B-6_0.zip` | [Download](https://drive.google.com/file/d/1B3Iq5BuwWbZo4Cuxu1vJa9M9wN2XLugZ/view?usp=sharing) | `b3c06dc9c8d91620aae9b85a09f08532e0ed39f33639850092d2adea096da5a0` |
| **4B** | 8.0 | `Gemma3-4B-8_0.zip` | [Download](https://drive.google.com/file/d/19MjQdtLuVDt1UNhO4QAKIwa-LyNh-S7n/view?usp=sharing) | `8ad94dadb14553940bd27a735df779d7af3d7fdef4900c9331e052f397433882` |

---

## 3. Phi-4 Series (Standard)
Microsoft의 Phi-4 공식 모델 기반입니다.

| 퀀텀 (Quant) | 파일명 (Filename) | 다운로드 (Link) | SHA256 Checksum |
|:---:|:---|:---:|:---|
| 2.0 | `Phi4-2_0.zip` | [Download](https://drive.google.com/file/d/1VN7qjc6dMvE4OgGOV9iLA8gaY8FqxFQw/view?usp=sharing) | `f5ea673afdc6ad19f508b9f71397a05517da36dec1721fa67d227de8f9d31f7d` |
| 4.0 | `Phi4-4_0.zip` | [Download](https://drive.google.com/file/d/1MM-BQAloZ2Dq0yIyBDEDI80kN5iNDS4s/view?usp=sharing) | `a3a64dc8c8afafb52d24e53673e3cd14c0567dda5f0698ff179c6f7be75c8ab2` |
| 6.0 | `Phi4-6_0.zip` | [Download](https://drive.google.com/file/d/1b0IV5GLg_kFu7wUnhyujbxQAhKa0CwEG/view?usp=sharing) | `3fa79d4060f123b24dc22bbf9b5afbcd5ffe1f242c84c12d5d14210bfe233c82` |
| 8.0 | `Phi4-8_0.zip` | [Download](https://drive.google.com/file/d/1WzOlQEoE2qL0Gj1bDtV3BCI3Nq4cib2B/view?usp=sharing) | `59719473006b50091fc5f0d9ff11796b5e14926a1124770c5f00c6788208cb97` |

---

## 4. Phi-4 Series (Uncensored / Mirror)
검열이 해제된 튜닝 버전입니다. **파일명은 Standard와 동일합니다.**

| 퀀텀 (Quant) | 파일명 (Filename) | 다운로드 (Link) | SHA256 Checksum |
|:---:|:---|:---:|:---|
| 2.0 | `Phi4-2_0.zip` | [Download](https://drive.google.com/file/d/1ul7mmru7P7d3P9JMTHXNpD0n-oDUsm0V/view?usp=sharing) | `01b1aa5b9595c8e18246cd3cb2305216ca1a82991c785c1a4309243e12a80194` |
| 4.0 | `Phi4-4_0.zip` | [Download](https://drive.google.com/file/d/16PR7hzcQXOHpJDlCu8uoorikU-jHa8Fv/view?usp=sharing) | `645bd6b1da6b2242e4e7f0baf2735a95d05eefab9a5bf9006fd88256b5f0a724` |
| 6.0 | `Phi4-6_0.zip` | [Download](https://drive.google.com/file/d/1G-J39D5oPick5_dHJcer50Cp1thInB7Z/view?usp=sharing) | `becaf9cbc10cde182643447aff68f59e98300e0acf454f2523ad8cc9aa229916` |
| 8.0 | `Phi4-8_0.zip` | [Download](https://drive.google.com/file/d/1yvKN9jTTaice3wdsVzRzq2-6-LIvuOHi/view?usp=sharing) | `49c026443dcf51bcc527e865a9fa9393cf56f9e38a5e7292c6f0a3644104cbe0` |

---

## 5. Other Models (TTS & Emotion)

| 모델명 (Model) | 파일명 (Filename) | 다운로드 (Link) | SHA256 / Note |
|:---:|:---|:---:|:---|
| **TTS1** | `TTS1.zip` | [Download](https://drive.google.com/file/d/179yiCLUt-HJ0S7FNBvNMgSHWYUbqYP8v/view?usp=sharing) | `ec651b10c051f02423ebc72a4fe5c716ee4ee8f65320a54c34f269643b02474d` |
| **Emotion** | `Emotion.safetensors` | [HuggingFace Repo](https://huggingface.co/MelissaJ/koelectra-emotion-6-emotion-base/tree/main) | 해당 리포지토리에서 `model.safetensors` (또는 `pytorch_model.bin`)을 다운로드 후 `Emotion.safetensors`로 변경 |

---
*Developed by MelissaJ*
