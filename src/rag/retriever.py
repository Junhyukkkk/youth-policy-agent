# ============================================================
# rag/retriever.py — 사용자 질문과 유사한 청크를 Pinecone에서 검색
#
# 역할: ingest.py가 저장해 놓은 벡터 데이터에서,
#       사용자 질문을 벡터로 변환한 뒤 코사인 유사도가 가장 높은
#       청크 k개를 찾아서 반환한다.
#
# 흐름:
#   질문 텍스트 → Gemini 임베딩 → 768차원 벡터
#       → Pinecone에서 유사 벡터 top-k 검색
#       → 해당 청크의 원문 텍스트 + 메타데이터 반환
#
# 이 모듈은 orchestrator.py의 rag_search 도구에서 호출된다.
# ============================================================

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import settings


def _get_vectorstore() -> PineconeVectorStore:
    # Gemini 임베딩 모델: 검색 시에도 ingest 때와 동일한 모델로 벡터를 만들어야 한다.
    # (다른 모델을 쓰면 벡터 공간이 달라져서 유사도 검색이 엉망이 됨)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.google_api_key,
    )
    # 이미 존재하는 Pinecone 인덱스에 연결 (ingest.py가 미리 데이터를 넣어둔 인덱스)
    return PineconeVectorStore.from_existing_index(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
    )


def get_retriever(
    k: int = 4,
    filter: dict | None = None,
) -> VectorStoreRetriever:
    """Pinecone retriever 반환. filter 예: {"region": "서울"}"""
    # k: 몇 개의 유사 청크를 가져올지 (기본 4개)
    search_kwargs: dict = {"k": k}
    if filter:
        # 메타데이터 필터: 예를 들어 서울 지역 PDF만 검색하고 싶을 때
        # {"region": "서울"} 을 넘기면 서울 문서에서만 검색함
        search_kwargs["filter"] = filter
    return _get_vectorstore().as_retriever(search_kwargs=search_kwargs)


def retrieve(
    query: str,
    k: int = 4,
    filter: dict | None = None,
) -> list[Document]:
    """쿼리 + 선택적 메타데이터 필터로 관련 청크 반환.

    반환값: Document 리스트
        - doc.page_content: 청크 원문 텍스트
        - doc.metadata: {"region": ..., "category": ..., "source_file": ..., "page": ...}
    """
    # get_retriever().invoke(query) = 질문을 벡터로 변환 → Pinecone 유사도 검색 → 상위 k개 반환
    return get_retriever(k=k, filter=filter).invoke(query)
