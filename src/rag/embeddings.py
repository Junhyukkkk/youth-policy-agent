# google-genai SDK (v1 API) 기반 임베딩 클래스
# langchain-google-genai의 GoogleGenerativeAIEmbeddings는 v1beta API를 사용해서
# text-embedding-004 모델이 404 에러가 나는 문제가 있다.
# google-genai SDK는 v1 API를 사용하므로 이 문제가 없다.

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

from src.config import settings


class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str = "gemini-embedding-001"):
        # 이 API 키에서 사용 가능한 임베딩 모델: gemini-embedding-001 (3072차원)
        # text-embedding-004는 이 키에서 지원 안 됨 (ListModels로 확인)
        self._client = genai.Client(
            api_key=settings.google_api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return response.embeddings[0].values
