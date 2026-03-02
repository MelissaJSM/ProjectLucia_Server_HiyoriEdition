# ──────────────────────────────────────────────────────────────────────────────
# Core/rag_search.py
# 단일 검색 모듈 (DDGS - Dux Distributed Global Search)
# ──────────────────────────────────────────────────────────────────────────────
import sys
import time
import logging
from typing import List, Dict, Any, Optional

# 최신 DDGS 메타검색 라이브러리 (pip install -U ddgs)
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
# 설정 및 상수
# ──────────────────────────────────────────────────────────────────────────────
REGION_DDG = "kr-kr"
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
# 단일 검색 엔진: DDGS
# ──────────────────────────────────────────────────────────────────────────────
def fetch_ddgs(query: str, max_results: int) -> List[Dict[str, Any]]:
    """새로운 DDGS 라이브러리를 통해 웹 검색을 수행합니다."""
    docs = []
    logger.info(f"🚀 [DDGS Search] 검색 시작: {query}")

    try:
        # 공식 문서에 따른 새로운 초기화 및 호출 방식 적용
        results = DDGS().text(
            query=query,  # 구버전의 'keywords' 파라미터가 'query'로 변경됨
            region=REGION_DDG,
            safesearch=SAFESEARCH,
            backend="auto",  # 최신 문서 권장 설정 (다양한 검색 엔진 자동 라우팅)
            max_results=max_results
        )

        # Generator가 반환될 수 있으므로 list 변환 및 순회 방어
        if not results:
            return docs

        for i, r in enumerate(results, 1):
            body = r.get("body", "")
            if not body:
                continue
            docs.append({
                "index": i,
                "title": r.get("title", ""),
                "snippet": body,
                "url": r.get("href", ""),
                "source": "Web Search (DDGS)"
            })
    except Exception as e:
        logger.error(f"❌ [DDGS Search] 검색 실패: {e}")

    return docs


# ──────────────────────────────────────────────────────────────────────────────
# 메인 오케스트레이터
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_webrag(
        question: str,
        search_query: Optional[str] = None,
        *,
        max_results_search: int = 5,  # 결과를 핵심만 빠르게 가져오도록 기본값 축소
        use_embedding: bool = False,
        select_top: bool = False
) -> Dict[str, Any]:
    q_for_search = (search_query or question).strip()

    # [검색 의도 파악] 키워드가 없으면 검색을 수행하지 않음
    if not is_search_needed(q_for_search):
        return {
            "query": question,
            "search_query": q_for_search,
            "generated_at": now_utc_iso(),
            "docs": [],
            "best_text": "검색 결과가 없습니다. (검색어 미감지)",
            "debug_info": {"engine_used": "None", "doc_count": 0}
        }

    logger.info(f"🔹 [Web-RAG] Searching for: {q_for_search}")

    # DDGS 검색 실행
    docs = fetch_ddgs(q_for_search, max_results_search)
    used_engine = "DDGS" if docs else "None"

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