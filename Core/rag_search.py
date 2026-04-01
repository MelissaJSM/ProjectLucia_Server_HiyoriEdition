import sys
import time
import logging
from typing import List, Dict, Any, Optional

# 최신 DDGS 메타검색 라이브러리
from ddgs import DDGS

# ──────────────────────────────────────────────────────────────────────────────
# 로거 설정
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("llm.rag_search")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = sys.stdout
    handler = logging.StreamHandler(console_handler)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)

# ──────────────────────────────────────────────────────────────────────────────
# 의도 분류 키워드 정의
# ──────────────────────────────────────────────────────────────────────────────
INTENT_KEYWORDS = {
    "news": ["뉴스", "속보", "사건", "보도", "기사", "news", "최근 소식"],
    "book": ["책", "도서", "작가", "출판", "소설", "book", "author"],
    "media": ["사진", "이미지", "영상", "동영상", "유튜브", "image", "video", "youtube"],
    "info": ["날씨", "주가", "환율", "지도", "위치", "맛집", "가격", "최저가"] # 일반 text()가 유리한 항목
}

def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def detect_search_intent(query: str) -> str:
    """질문의 의도를 분석하여 적절한 DDGS 메서드 타입을 반환합니다."""
    q_lower = query.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(k in q_lower for k in keywords):
            return intent
    return "general"

# ──────────────────────────────────────────────────────────────────────────────
# 본문 추출 (Deep RAG) 기능
# ──────────────────────────────────────────────────────────────────────────────
def fetch_full_content(url: str) -> str:
    """검색 결과의 URL에서 실제 본문 내용을 마크다운으로 추출합니다."""
    try:
        logger.info(f"📄 [Extract] 본문 추출 중: {url}")
        result = DDGS().extract(url, fmt="text_markdown")
        return result.get("content", "")
    except Exception as e:
        logger.warning(f"⚠️ [Extract] 추출 실패 ({url}): {e}")
        return ""

# ──────────────────────────────────────────────────────────────────────────────
# 분기형 검색 엔진: DDGS Multi-Route
# ──────────────────────────────────────────────────────────────────────────────
def fetch_ddgs_smart(query: str, max_results: int) -> List[Dict[str, Any]]:
    """의도에 따라 news, books, text 등 최적의 메서드를 호출합니다."""
    docs = []
    intent = detect_search_intent(query)
    ddgs = DDGS()
    
    logger.info(f"🚀 [DDGS] 의도 감지: {intent} | 검색어: {query}")
    
    try:
        if intent == "news":
            results = ddgs.news(query, region="kr-kr", safesearch="moderate", max_results=max_results)
        elif intent == "book":
            results = ddgs.books(query, max_results=max_results)
        else:
            # 날씨, 주가, 일반 지식 등은 text()가 가장 정확함
            results = ddgs.text(query, region="kr-kr", safesearch="moderate", max_results=max_results)

        if not results:
            return []

        for i, r in enumerate(results, 1):
            # 필드명이 메서드마다 조금씩 다름 (href vs url, body vs description)
            url = r.get("href") or r.get("url")
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
            
            # [Deep RAG] 최상위 결과 1~2개는 본문을 직접 긁어옴 (LLM 답변 질 향상)
            if i <= 2 and url:
                full_text = fetch_full_content(url)
                if full_text:
                    doc["full_content"] = full_text[:2000] # 너무 길면 2000자에서 자름
            
            docs.append(doc)
            
    except Exception as e:
        logger.error(f"❌ [DDGS] 검색 실패: {e}")
        
    return docs

# ──────────────────────────────────────────────────────────────────────────────
# 메인 오케스트레이터
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_webrag(
        question: str,
        search_query: Optional[str] = None,
        *,
        max_results_search: int = 3,
        use_embedding: bool = False,  # 👈 기존 시스템에서 넘겨주는 파라미터 복구
        select_top: bool = False,     # 👈 기존 시스템에서 넘겨주는 파라미터 복구
        **kwargs                      # 👈 혹시 모를 추가 파라미터 방어용
) -> Dict[str, Any]:
    q_for_search = (search_query or question).strip()
    
    # 검색 실행
    docs = fetch_ddgs_smart(q_for_search, max_results_search)
    
    # LLM에 전달할 컨텍스트 조립
    formatted_texts = []
    for d in docs:
        content = d['full_content'] if d['full_content'] else d['snippet']
        formatted_texts.append(
            f"### [문서 {d['index']}] {d['title']}\n"
            f"- 출처: {d['url']}\n"
            f"- 내용: {content}\n"
        )
    
    final_context = "\n".join(formatted_texts) if docs else "검색 결과를 찾지 못했습니다."

    return {
        "query": question,
        "search_query": q_for_search,
        "docs": docs,
        "best_text": final_context,
        "generated_at": now_utc_iso()
    }
