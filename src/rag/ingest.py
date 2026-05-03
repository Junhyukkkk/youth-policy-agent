import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from rich.console import Console
from rich.progress import track

from src.config import settings
from src.rag.splitter import get_splitter

console = Console()


class FileMeta(NamedTuple):
    region: str
    category: str
    doc_name: str


def parse_filename(filename: str) -> FileMeta:
    """파일명 규칙 {지역}_{카테고리}_{문서명}.pdf 에서 메타데이터 파싱."""
    stem = Path(filename).stem
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return FileMeta(region=parts[0], category=parts[1], doc_name=parts[2])
    if len(parts) == 2:
        return FileMeta(region=parts[0], category=parts[1], doc_name=stem)
    return FileMeta(region="unknown", category="unknown", doc_name=stem)


def _safe_id(text: str) -> str:
    return re.sub(r"[^\w\-]", "_", text)


def load_pdf(pdf_path: Path) -> list[Document]:
    meta = parse_filename(pdf_path.name)
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for page in pages:
        page.metadata.update(
            {
                "region": meta.region,
                "category": meta.category,
                "doc_name": meta.doc_name,
                "source_file": pdf_path.name,
            }
        )
    return pages


def _make_ids(chunks: list[Document], stem: str) -> list[str]:
    """청크별 결정적 ID: {stem}-p{page}-c{chunk_within_page} — upsert 중복 방지."""
    page_counts: dict[int, int] = defaultdict(int)
    ids: list[str] = []
    for chunk in chunks:
        page = int(chunk.metadata.get("page", 0))
        idx = page_counts[page]
        ids.append(f"{stem}-p{page}-c{idx}")
        page_counts[page] += 1
    return ids


def ingest_directory(path: Path) -> tuple[int, int]:
    """PDF 디렉토리를 Pinecone에 upsert. (적재 문서 수, 총 청크 수) 반환."""
    pdf_files = sorted(path.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]경고: {path} 에 PDF 파일이 없습니다.[/yellow]")
        return 0, 0

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.google_api_key,
    )
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
    )
    splitter = get_splitter()

    total_docs = 0
    total_chunks = 0

    for pdf_path in track(pdf_files, description="PDF 처리 중..."):
        try:
            pages = load_pdf(pdf_path)
            chunks = splitter.split_documents(pages)
            stem = _safe_id(pdf_path.stem)
            ids = _make_ids(chunks, stem)
            vectorstore.add_documents(chunks, ids=ids)
            total_docs += 1
            total_chunks += len(chunks)
            console.print(f"  [green]✓[/green] {pdf_path.name} → {len(chunks)}청크")
        except Exception as e:
            console.print(f"  [red]✗[/red] {pdf_path.name}: {e}")

    return total_docs, total_chunks
