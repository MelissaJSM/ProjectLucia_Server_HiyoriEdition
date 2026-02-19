# ──────────────────────────────────────────────────────────────────────────────
# Core/rag_search.py
# 하이브리드 검색 모듈 (Google MCP Router -> DuckDuckGo)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import sys
import time
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple

# DuckDuckGo 검색 라이브러리
from ddgs import DDGS

# MCP 관련 라이브러리 (pip install mcp noapi-google-search-mcp)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ──────────────────────────────────────────────────────────────────────────────
# 로거 설정
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("llm.rag_search")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

# ──────────────────────────────────────────────────────────────────────────────
# 설정 및 상수
# ──────────────────────────────────────────────────────────────────────────────
REGION_DDG = "kr-ko"
SAFESEARCH = "moderate"

# 검색 트리거 키워드 정의
KEYWORDS_WEATHER = ["날씨", "기온", "비 오나", "눈 오나", "weather"]
KEYWORDS_MAP = ["위치", "지도", "어디", "맛집", "가는 길", "거리"]
KEYWORDS_NEWS = ["뉴스", "속보", "사건", "news"]
KEYWORDS_FINANCE = ["주식", "주가", "코인", "환율", "삼성전자", "bitcoin", "stock"]
KEYWORDS_SHOPPING = ["가격", "얼마", "최저가", "싸게 사는"]
KEYWORDS_GENERAL = ["검색", "찾아", "search", "find", "알려줘", "뭐야", "누구야"]

# 전체 키워드 통합 (검색 여부 판단용)
ALL_SEARCH_KEYWORDS = KEYWORDS_WEATHER + KEYWORDS_MAP + KEYWORDS_NEWS + KEYWORDS_FINANCE + KEYWORDS_SHOPPING + KEYWORDS_GENERAL


def now_utc_iso() -> str:
    """현재 UTC 시간을 ISO 8601 형식으로 반환합니다."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def is_search_needed(query: str) -> bool:
    """쿼리에 검색 키워드가 포함되어 있는지 확인합니다."""
    q_lower = query.lower()
    return any(k in q_lower for k in ALL_SEARCH_KEYWORDS)


# ──────────────────────────────────────────────────────────────────────────────
# [신규] 검색 엔진 0: Google MCP (Router + Fallback 적용)
# ──────────────────────────────────────────────────────────────────────────────
async def _run_mcp_google_router(query: str, max_results: int) -> Tuple[str, str]:
    """
    비동기 함수: 질문 내용을 분석하여 적절한 Google 도구를 선택해 실행합니다.
    Returns: (결과 텍스트, 사용된 도구 이름)
    """
    # MCP 서버 실행 설정
    server_params = StdioServerParameters(
        command="noapi-google-search-mcp",
        args=[],
        env=None
    )

    # 1. 도구 라우팅 (Router) 로직
    tool_name = None
    tool_args = {}

    q_lower = query.lower()

    # [날씨] - location 인자 사용
    if any(k in q_lower for k in KEYWORDS_WEATHER):
        tool_name = "google_weather"
        tool_args = {"location": query}

    # [지도/위치/맛집]
    elif any(k in q_lower for k in KEYWORDS_MAP):
        tool_name = "google_maps"
        tool_args = {"query": query, "num_results": max_results}

    # [뉴스/속보]
    elif any(k in q_lower for k in KEYWORDS_NEWS):
        tool_name = "google_news"
        tool_args = {"query": query, "num_results": max_results}

    # [주식/금융]
    elif any(k in q_lower for k in KEYWORDS_FINANCE):
        tool_name = "google_finance"
        tool_args = {"query": query}

    # [쇼핑/가격]
    elif any(k in q_lower for k in KEYWORDS_SHOPPING):
        tool_name = "google_shopping"
        tool_args = {"query": query, "num_results": max_results}

    # [일반 검색] - 최후순위
    elif any(k in q_lower for k in KEYWORDS_GENERAL):
        tool_name = "google_search"
        tool_args = {
            "query": query,
            "num_results": max_results,
            "language": "ko",
            "region": "kr"
        }

    # 키워드 매칭 실패 시 검색하지 않음
    if tool_name is None:
        return "", "None"

    logger.info(f"🛠️ [MCP Router] '{query}' -> 선택된 도구: {tool_name}")

    # 2. 실제 실행
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            try:
                # 선택된 도구 실행
                result = await session.call_tool(tool_name, arguments=tool_args)

                # 결과가 있으면 반환
                if result.content:
                    # 텍스트 추출 로직 강화
                    extracted_text = ""
                    for content in result.content:
                        if hasattr(content, "text") and content.text:
                            extracted_text += content.text + "\n"

                    if extracted_text.strip():
                        return extracted_text.strip(), tool_name
                    else:
                        logger.warning(f"⚠️ [MCP Warning] {tool_name} 실행 완료되었으나 텍스트 내용이 비어있습니다.")

            except Exception as e:
                logger.warning(f"⚠️ [MCP Error] {tool_name} 실행 실패 ({e}). 일반 검색(google_search)으로 전환합니다.")

                # 실패 시 Fallback: 일반 검색 재시도
                if tool_name != "google_search":
                    try:
                        fallback_args = {
                            "query": query,
                            "num_results": max_results,
                            "language": "ko",
                            "region": "kr"
                        }
                        result = await session.call_tool("google_search", arguments=fallback_args)

                        extracted_text = ""
                        if result.content:
                            for content in result.content:
                                if hasattr(content, "text") and content.text:
                                    extracted_text += content.text + "\n"

                        if extracted_text.strip():
                            return extracted_text.strip(), "google_search (Fallback)"

                    except Exception as fallback_e:
                        logger.error(f"❌ [MCP Error] Fallback 검색도 실패: {fallback_e}")

            return "", "None"


def fetch_google_mcp(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Google MCP를 통해 검색을 수행하고 결과를 반환합니다. (동기 래퍼)
    """
    logger.info(f"🚀 [MCP] Google Search 시작: {query}")
    docs = []

    try:
        # 비동기 함수 실행
        raw_text, tool_used = asyncio.run(_run_mcp_google_router(query, max_results))

        if not raw_text:
            return []

        # 결과를 하나의 문서로 포맷팅
        docs.append({
            "index": 1,
            "title": f"Google Result ({tool_used})",
            "snippet": raw_text,
            "url": "google.com",
            "source": f"Google MCP [{tool_used}]"
        })

    except Exception as e:
        logger.error(f"❌ [MCP] 실행 중 치명적 오류: {e}")
        return []

    return docs


# ──────────────────────────────────────────────────────────────────────────────
# 검색 엔진 1: DuckDuckGo (Fallback)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_duckduckgo(query: str, max_results: int) -> List[Dict[str, Any]]:
    docs = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query=query, region=REGION_DDG, safesearch=SAFESEARCH, backend="auto",
                                max_results=max_results)
            for i, r in enumerate(results, 1):
                body = r.get("body", "")
                if not body: continue
                docs.append({
                    "index": i, "title": r.get("title", ""), "snippet": body,
                    "url": r.get("href", ""), "source": "DuckDuckGo"
                })
    except Exception as e:
        logger.error(f"❌ [Hybrid-RAG] DuckDuckGo fallback failed: {e}")
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# 메인 오케스트레이터
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_webrag(
        question: str,
        search_query: Optional[str] = None,
        *,
        max_results_search: int = 10,
        use_embedding: bool = False,
        select_top: bool = False
) -> Dict[str, Any]:
    q_for_search = (search_query or question).strip()

    # [검색 의도 파악] 키워드가 없으면 검색을 수행하지 않음
    if not is_search_needed(q_for_search):
        # logger.info(f"🚫 [Hybrid-RAG] 검색 키워드 미발견. 검색을 건너뜁니다. ('{q_for_search}')")
        return {
            "query": question,
            "search_query": q_for_search,
            "generated_at": now_utc_iso(),
            "docs": [],
            "best_text": "검색 결과가 없습니다. (검색어 미감지)",
            "debug_info": {"engine_used": "None", "doc_count": 0}
        }

    logger.info(f"🔹 [Hybrid-RAG] Searching for: {q_for_search}")

    used_engine = "None"
    docs = []

    # 1순위: Google MCP (Router 적용됨)
    if not docs:
        docs = fetch_google_mcp(q_for_search, max_results_search)
        if docs:
            # docs[0]['source']에 사용된 도구 이름이 들어있음 (예: Google MCP [google_weather])
            used_engine = docs[0]['source']

    # 2순위: DuckDuckGo (Google MCP 실패 시 Fallback)
    # 키워드가 있어서 검색을 시도했으나 Google MCP가 실패한 경우에만 실행
    if not docs:
        logger.info("🔸 [Hybrid-RAG] Switching to DuckDuckGo...")
        docs = fetch_duckduckgo(q_for_search, max_results_search)
        if docs: used_engine = "DuckDuckGo (Fallback)"

    # 결과 텍스트 조립
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

    # 로그 출력
    logger.info("=" * 40)
    logger.info(f"🔎 [RAG Status] Engine: {used_engine} | Docs: {len(docs)}")
    if docs:
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
