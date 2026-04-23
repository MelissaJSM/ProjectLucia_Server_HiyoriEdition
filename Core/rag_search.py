# ──────────────────────────────────────────────────────────────────────────────
# Core/rag_search.py
# 통합 검색 모듈 (DDGS - Dux Distributed Global Search 활용판)
# ──────────────────────────────────────────────────────────────────────────────
import sys
import time
import logging
from typing import List, Dict, Any, Optional

from ddgs import DDGS

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
# 의도 분류 및 검색 필터링 키워드
# ──────────────────────────────────────────────────────────────────────────────
KEYWORDS_IMAGE = ["사진", "이미지", "짤", "그림", "image", "picture", "photo"]
KEYWORDS_VIDEO = ["영상", "동영상", "유튜브", "비디오", "video", "youtube"]
KEYWORDS_NEWS = ["뉴스", "속보", "사건", "보도", "기사", "news", "최근 소식"]
KEYWORDS_BOOK = ["책", "도서", "작가", "출판", "소설", "book", "author"]
KEYWORDS_WEATHER = ["날씨", "기온", "비 오나", "눈 오나", "weather"]
KEYWORDS_MAP = ["위치", "지도", "어디", "맛집", "가는 길", "거리"]
KEYWORDS_FINANCE = ["주식", "주가", "코인", "환율", "삼성전자", "bitcoin", "stock"]
KEYWORDS_SHOPPING = ["가격", "얼마", "최저가", "싸게 사는"]
KEYWORDS_GENERAL = ["검색", "찾아", "search", "find", "알려줘", "뭐야", "누구야"]

ALL_SEARCH_KEYWORDS = (
        KEYWORDS_IMAGE + KEYWORDS_VIDEO + KEYWORDS_NEWS + KEYWORDS_BOOK +
        KEYWORDS_WEATHER + KEYWORDS_MAP + KEYWORDS_FINANCE + KEYWORDS_SHOPPING + KEYWORDS_GENERAL
)

def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def is_search_needed(query: str) -> bool:
    q_lower = query.lower()
    return any(k in q_lower for k in ALL_SEARCH_KEYWORDS)

def detect_search_intent(query: str) -> str:
    q_lower = query.lower()
    if any(k in q_lower for k in KEYWORDS_IMAGE): return "image"
    if any(k in q_lower for k in KEYWORDS_VIDEO): return "video"
    if any(k in q_lower for k in KEYWORDS_NEWS): return "news"
    if any(k in q_lower for k in KEYWORDS_BOOK): return "book"
    return "general"

def fetch_full_content(url: str) -> str:
    try:
        logger.info(f"📄 [Extract] 본문 추출 중: {url}")
        result = DDGS().extract(url, fmt="text_markdown")
        return result.get("content", "")
    except Exception as e:
        logger.warning(f"⚠️ [Extract] 추출 실패 ({url}): {e}")
        return ""

def fetch_ddgs_smart(query: str, max_results: int) -> List[Dict[str, Any]]:
    docs = []
    intent = detect_search_intent(query)
    ddgs = DDGS()

    logger.info(f"🚀 [DDGS Search] 의도: {intent} | 검색어: {query}")

    try:
        if intent == "image":
            results = ddgs.images(query, region="kr-kr", safesearch="moderate", max_results=max_results)
        elif intent == "video":
            results = ddgs.videos(query, region="kr-kr", safesearch="moderate", max_results=max_results)
        elif intent == "news":
            results = ddgs.news(query, region="kr-kr", safesearch="moderate", max_results=max_results)
        elif intent == "book":
            results = ddgs.books(query, max_results=max_results)
        else:
            results = ddgs.text(query, region="kr-kr", safesearch="moderate", max_results=max_results)

        if not results:
            return docs

        for i, r in enumerate(results, 1):
            if intent == "image":
                url = r.get("image") or r.get("url", "")
                title = r.get("title", "이미지")
                snippet = f"이미지 출처: {r.get('source', '알 수 없음')} | 썸네일: {r.get('thumbnail', '')}"
            elif intent == "video":
                url = r.get("content") or r.get("embed_url", "")
                title = r.get("title", "동영상")
                snippet = r.get("description", "설명 없음")
            else:
                url = r.get("href") or r.get("url", "")
                title = r.get("title", "제목 없음")
                snippet = r.get("body") or r.get("description") or r.get("info", "")

            doc = {
                "index": i,
                "title": title,
                "snippet": snippet,
                "url": url,
                "source": f"DDGS {intent.capitalize()}",
                "full_content": ""
            }

            if intent in ["general", "news"] and i <= 2 and url:
                full_text = fetch_full_content(url)
                if full_text:
                    doc["full_content"] = full_text[:1500]

            docs.append(doc)

    except Exception as e:
        logger.error(f"❌ [DDGS Search] 검색 실패 ({intent}): {e}")

    return docs

def preprocess_webrag(
        question: str,
        search_query: Optional[str] = None,
        *,
        max_results_search: int = 3,
        use_embedding: bool = False,
        select_top: bool = False,
        **kwargs
) -> Dict[str, Any]:
    q_for_search = (search_query or question).strip()

    if not is_search_needed(q_for_search):
        logger.info(f"⏭️ [Web-RAG] 검색 불필요 (일반 대화 감지) -> 패스: {q_for_search}")
        return {
            "query": question,
            "search_query": q_for_search,
            "generated_at": now_utc_iso(),
            "docs": [],
            "best_text": "검색 결과가 없습니다. (일반 대화 모드)",
            "debug_info": {"engine_used": "None (Bypassed)", "doc_count": 0}
        }

    docs = fetch_ddgs_smart(q_for_search, max_results_search)
    used_engine = "DDGS" if docs else "None"

    formatted_texts = []
    if docs:
        for d in docs:
            content = d['full_content'] if d['full_content'] else d['snippet']
            formatted_texts.append(
                f"### [문서 {d['index']} | {d['source']}]\n"
                f"- 제목: {d['title']}\n"
                f"- 링크: {d['url']}\n"
                f"- 내용: {content}\n"
            )
        final_context = "\n".join(formatted_texts)
    else:
        final_context = "검색 결과를 찾지 못했습니다."

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