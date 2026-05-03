"""Pinecone 인덱스 생성 스크립트. 이미 존재하면 skip (idempotent)."""
from pinecone import Pinecone, ServerlessSpec

from src.config import settings

DIMENSION = 768  # Gemini embedding-001
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"


def create_index() -> None:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [idx.name for idx in pc.list_indexes()]

    if settings.pinecone_index_name in existing:
        print(f"Index '{settings.pinecone_index_name}' already exists — skipping.")
        return

    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    print(f"Index '{settings.pinecone_index_name}' created.")


if __name__ == "__main__":
    create_index()
