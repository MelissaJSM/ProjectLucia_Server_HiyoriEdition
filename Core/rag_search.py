# ──────────────────────────────────────────────────────────────────────────────
# Core/rag_search.py
# 하이브리드 검색 모듈 (SearXNG 우선 시도 -> 실패 시 DuckDuckGo 전환)
# Docker 기반의 SearXNG가 있으면 고품질 검색을, 없으면 간편한 DDG 검색을 사용합니다.
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import sys
import time
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple

# DuckDuckGo 검색 라이브러리
from ddgs import DDGS

# 프로젝트 설정
import Core.server_config as server_config

# ──────────────────────────────────────────────────────────────────────────────
# 로거 설정
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("llm.rag_search")
logger.setLevel(logging.INFO)

# 서버 환경에서 로그가 출력되지 않는 문제를 해결하기 위해 핸들러 강제 추가
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

# ──────────────────────────────────────────────────────────────────────────────
# 설정 및 상수
# ──────────────────────────────────────────────────────────────────────────────
SEARXNG_TIMEOUT = 5.0

# 검색 지역 및 안전 검색 설정
REGION_SEARX = "ko-KR"
REGION_DDG = "kr-ko"
SAFESEARCH = "moderate"


def now_utc_iso() -> str:
    """현재 UTC 시간을 ISO 8601 형식으로 반환합니다."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ──────────────────────────────────────────────────────────────────────────────
# [신규] RAG 서버 상태 확인
# ──────────────────────────────────────────────────────────────────────────────
def check_rag_status(ip: str, port: int) -> Tuple[bool, str]:
    """
    SearXNG 서버의 상태를 확인합니다.
    
    Args:
        ip (str): 서버 IP 주소
        port (int): 서버 포트 번호
        
    Returns:
        Tuple[bool, str]: (성공 여부, 메시지)
    """
    if not ip:
        return False, "IP 주소가 설정되지 않았습니다."
    
    url = f"http://{ip}:{port}/"
    try:
        response = requests.get(url, timeout=SEARXNG_TIMEOUT)
        # SearXNG는 보통 200 OK와 함께 HTML 페이지를 반환합니다.
        if response.status_code == 200:
            return True, "연결 성공"
        else:
            return False, f"연결 실패 (HTTP {response.status_code})"
    except requests.exceptions.RequestException as e:
        return False, f"연결 실패 ({type(e).__name__})"


# ──────────────────────────────────────────────────────────────────────────────
# 검색 엔진 1: SearXNG (Docker 기반)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_searxng(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    SearXNG 인스턴스를 통해 검색을 수행합니다.
    """
    # [중요] 브라우저인 척 위장 (403 에러 방지)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # 설정에서 IP/PORT 가져와서 URL 구성
    rag_ip = getattr(server_config.RAG, "RAG_IP", "")
    rag_port = getattr(server_config.RAG, "RAG_PORT", 8080)
    
    # IP가 설정되지 않았으면 기본값 사용 (예: 오드로이드 IP)
    if not rag_ip:
        rag_ip = "192.168.35.97"
        
    searxng_url = f"http://{rag_ip}:{rag_port}/search"

    try:
        params = {
            "q": query,
            "format": "json",
            "language": REGION_SEARX,
            "categories": "general",
            "safesearch": 1
        }

        resp = requests.get(searxng_url, params=params, headers=headers, timeout=SEARXNG_TIMEOUT)

        if resp.status_code != 200:
            logger.warning(f"⚠️ [SearXNG] Status {resp.status_code}: {resp.text[:50]}...")
            return []

        data = resp.json()
        raw_results = data.get("results", [])

        docs = []
        for i, r in enumerate(raw_results[:max_results], 1):
            content = r.get("content", "")
            if not content: continue

            docs.append({
                "index": i,
                "title": r.get("title", ""),
                "snippet": content,
                "url": r.get("url", ""),
                "source": "SearXNG"
            })
        return docs

    except Exception as e:
        logger.warning(f"⚠️ [Hybrid-RAG] SearXNG unavailable ({searxng_url}): {e}")

    return []


# ──────────────────────────────────────────────────────────────────────────────
# 검색 엔진 2: DuckDuckGo (Fallback)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_duckduckgo(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    DuckDuckGo를 통해 검색을 수행합니다. (SearXNG 실패 시 사용)
    """
    docs = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query=query,
                region=REGION_DDG,
                safesearch=SAFESEARCH,
                backend="auto",  # [수정] lite -> auto (에러 방지)
                max_results=max_results
            )

            for i, r in enumerate(results, 1):
                body = r.get("body", "")
                if not body: continue

                docs.append({
                    "index": i,
                    "title": r.get("title", ""),
                    "snippet": body,
                    "url": r.get("href", ""),
                    "source": "DuckDuckGo"
                })
    except Exception as e:
        logger.error(f"❌ [Hybrid-RAG] DuckDuckGo fallback failed: {e}")

    return docs


# ──────────────────────────────────────────────────────────────────────────────
# 메인 오케스트레이터 (검색 수행 및 결과 조합)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_webrag(
        question: str,
        search_query: Optional[str] = None,
        *,
        max_results_search: int = 10,
        use_embedding: bool = False,
        select_top: bool = False
) -> Dict[str, Any]:
    """
    RAG 검색을 수행하고 결과를 전처리하여 반환합니다.
    
    1. SearXNG 검색 시도
    2. 실패 시 DuckDuckGo 검색 시도
    3. 검색 결과를 텍스트로 포맷팅
    """
    q_for_search = (search_query or question).strip()
    logger.info(f"🔹 [Hybrid-RAG] Searching for: {q_for_search}")
    
    used_engine = "None"

    # 1. SearXNG 시도
    docs = fetch_searxng(q_for_search, max_results_search)

    if docs:
        used_engine = "SearXNG (Odroid)"
    else:
        # 2. 실패 시 DuckDuckGo
        logger.info("🔸 [Hybrid-RAG] Switching to DuckDuckGo...")
        docs = fetch_duckduckgo(q_for_search, max_results_search)
        used_engine = "DuckDuckGo (Fallback)"

    # 3. 텍스트 조립
    formatted_texts = []
    if docs:
        for d in docs:
            formatted_texts.append(
                f"[문서 {d['index']} | {d['source']}]\n"
                f"제목: {d['title']}\n"
                f"출처: {d['url']}\n"
                f"내용: {d['snippet']}"
            )
        final_context = "\n\n".join(formatted_texts)
    else:
        final_context = "검색 결과가 없습니다."

    # ─────────────────────────────────────────────────────────────
    # [로그] 검색 결과 요약 출력
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 40)
    logger.info(f"🔎 [RAG Status] Engine: {used_engine} | Docs: {len(docs)}")
    if docs:
        # 첫 번째 검색 결과의 제목만 미리보기로 출력
        logger.info(f"📄 [Preview] Top 1: {docs[0]['title']}")
    else:
        logger.warning("❌ [Result] No documents found.")
    logger.info("=" * 40)

    return {
        "query": question,
        "search_query": q_for_search,
        "generated_at": now_utc_iso(),
        "docs": docs,
        "best_text": final_context,
        "debug_info": {
            "engine_used": used_engine,
            "doc_count": len(docs)
        }
    }


# (테스트용) 직접 실행 시 동작
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    preprocess_webrag(question="체인소맨 극장판 개봉일", max_results_search=5)
