# Project Lucia Server

[![YouTube Playlist](https://img.shields.io/badge/YouTube-재생목록_보기-red?style=for-the-badge&logo=youtube)](https://youtu.be/SkylnPtca3g?list=PLraK8WBiwejO1oeHq9y3W5iO0dZl5q2k8)

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
*   **하이브리드 RAG 검색**: `DuckDuckGo`를 통해 최신 웹 정보를 검색하여 답변의 정확도를 높입니다.
*   **모델 관리자**: GUI 내에서 필요한 AI 모델 파일을 손쉽게 다운로드하고 관리할 수 있습니다.

---

## 🛠 기술 스택

### Core
*   **Language**: Python 3.10+
*   **Framework**: FastAPI (WebSocket & REST API)
*   **Database**: MySQL (대화 로그 및 설정 저장)

### AI & ML
*   **LLM Backend**: ExLlamaV3 (Gemma-3, Gemma-4  지원)
*   **TTS Engine**: GPT-SoVITS
*   **Emotion Analysis**: KoELECTRA (Hugging Face Transformers)
*   **Vision**: Multimodal LLM Integration

### GUI & Tools
*   **Interface**: PyQt5 (Qt Designer)
*   **Process Management**: QProcess, psutil
*   **Search**: DuckDuckGo (ddgs)

---

## 📥 설치 방법

### 1. 필수 요구 사항
*   **OS**: Ubuntu 22.04 (Windows 호환 가능)
*   **GPU**: NVIDIA GPU (CUDA 12.8+ 권장, VRAM 12GB 이상 권장)
*   **Database**: MySQL Server 설치 및 실행 필요
  
#### 🖥️ 시스템 요구 사양 (GPT-SoVITS 병렬 구동 기준)
TTS 모델(약 3GB)이 상시 VRAM을 점유하므로 이를 고려한 사양입니다.

*   **1. 최소 사양 (Minimum Spec)**
    *   **목표:** 초경량 모델(Gemma-3 270M 또는 1B) 구동 및 TTS 음성 출력
    *   **필요 VRAM:** 약 4.5GB ~ 5.3GB (모델 1.5~2.3GB + TTS 3GB)
    *   **권장 GPU:** **NVIDIA GTX 1660 Super (6GB) 또는 RTX 3050 / 4060 (8GB)**
    *   **시스템 RAM:** 16GB 이상
    > **💡 참고:** 가장 가벼운 270M이나 1B 모델을 사용할 경우, VRAM 6~8GB 수준의 보급형 게이밍 노트북이나 데스크탑에서도 무리 없이 구동 가능합니다.

*   **2. 표준 사양 (Standard Spec)**
    *   **목표:** 경량 모델(Gemma-3 4B) 구동 및 TTS 병행
    *   **필요 VRAM:** 약 8.1GB ~ 9.6GB (모델 5~6.6GB + TTS 3GB)
    *   **권장 GPU:** **NVIDIA RTX 3060 (12GB) 또는 RTX 4060 Ti (16GB)**
    *   **시스템 RAM:** 32GB 이상
    > **💡 참고:** 4B 모델부터는 VRAM 8GB GPU 사용 시 메모리 부족(OOM)이 발생하거나 속도가 저하될 수 있으므로 12GB 이상의 메인스트림 GPU를 권장합니다.

*   **3. 권장 사양 (Recommended Spec)**
    *   **목표:** 가장 효율이 좋은 Sweet Spot 구간(Gemma-3 12B) 구동 및 TTS 병행
    *   **필요 VRAM:** 약 12.8GB ~ 17.8GB (모델 9~14GB + TTS 3GB)
    *   **권장 GPU:** **NVIDIA RTX 4070 Ti SUPER (16GB) 또는 RTX 4080 (16GB)**
    *   **시스템 RAM:** 64GB 이상
    > **💡 참고:** 기존 12B 4bit는 12GB GPU로 가능했으나, TTS(3GB) 병행 시 총 12.8GB가 필요하므로 16GB VRAM 탑재 모델이 안정적입니다.

*   **4. 하이엔드 사양 (High-End Spec)**
    *   **목표:** 대형 모델(Gemma-3 27B / Gemma-4 31B) 구동 및 TTS 병행
    *   **필요 VRAM:** 약 21.6GB ~ 27.5GB (모델 18~24GB + TTS 3GB)
    *   **권장 GPU:** **NVIDIA RTX 3090 (24GB) 또는 RTX 4090 (24GB)**
    *   **시스템 RAM:** 64GB ~ 128GB
    > **💡 참고:** 27B 4bit 모델은 단일 24GB GPU에서 원활히 동작합니다. 단, 27B 6bit 이상이나 31B 8bit 구동 시에는 3090 2way(NVLink) 또는 Mac Studio(통합 메모리 64GB 이상)가 필요합니다.
    
### 2. 클론 및 패키지 설치

# 가상 환경 생성 (아나콘다 혹은 직접)
[(설치 환경 가이드)](https://github.com/MelissaJSM/ProjectLucia_Finetuning_Server#%EF%B8%8F-project-lucia-server-guide) 를 따라하시면 환경 생성이 가능합니다.

# 깃 클론
```bash
git clone https://github.com/MelissaJSM/ProjectLucia_Server_HiyoriEdition.git
cd ProjectLucia_Server_HiyoriEdition
```

### 3. 추가 라이브러리 설치
[(라이브러리)](https://drive.google.com/file/d/1_uipZcaVuKtW5CodxHCbQVHErQ51e2YS/view?usp=sharing) 를 다운로드하여 프로젝트 최상단 폴더에 있는 Core폴더에 압축 내부 파일을 전부 복사합니다.
- 혹은 Release 에서 모든 파일을 다운받으시면 해당과정을 스킵해도됩니다.

### 4. 데이터베이스 설정
MySQL을 설치 한 후 DB.SQL 파일을 임포트하여 구성합니다.

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
*   **포트 충돌**: 기본 포트(3545, 3546, 8000, 9880)가 사용 중인지 확인하세요. 만약 충돌상황이면 cmd(관리자권한) -> net stop winnat -> net start winnat 을 입력해보세요. 인터넷이 잠시 끊길수도 있습니다.
*   **GPU 메모리**: LLM과 TTS 모델을 동시에 로드하므로 충분한 VRAM이 필요합니다.

---
*Developed by MelissaJ*
