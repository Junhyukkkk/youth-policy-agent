from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import settings


def _get_vectorstore() -> PineconeVectorStore:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.google_api_key,
    )
    return PineconeVectorStore.from_existing_index(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
    )


def get_retriever(
    k: int = 4,
    filter: dict | None = None,
) -> VectorStoreRetriever:
    """Pinecone retriever 반환. filter 예: {"region": "서울"}"""
    search_kwargs: dict = {"k": k}
    if filter:
        search_kwargs["filter"] = filter
    return _get_vectorstore().as_retriever(search_kwargs=search_kwargs)


def retrieve(
    query: str,
    k: int = 4,
    filter: dict | None = None,
) -> list[Document]:
    """쿼리 + 선택적 메타데이터 필터로 관련 청크 반환."""
    return get_retriever(k=k, filter=filter).invoke(query)
