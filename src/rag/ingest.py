# ============================================================
# rag/ingest.py — PDF → 청크 → 벡터 임베딩 → Pinecone 저장 (1회성 작업)
#
# 전체 흐름:
#   1. data/policies/ 폴더 안의 PDF 파일을 스캔
#   2. 각 PDF를 페이지 단위로 읽고, 파일명에서 메타데이터(지역·분류) 추출
#   3. 긴 텍스트를 1000자짜리 청크로 분할
#   4. 각 청크를 Gemini 임베딩 모델로 768차원 벡터로 변환
#   5. Pinecone 인덱스에 저장 (upsert = 중복이면 덮어쓰기)
#
# 이 파일은 CLI의 "policy-agent ingest" 명령으로만 실행된다.
# 한 번 저장해두면 retriever.py가 이 데이터를 검색해서 쓴다.
# ============================================================

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

# Rich 콘솔 객체 — print 대신 컬러/아이콘 있는 출력을 위해 사용
console = Console()


# ── 파일명 메타데이터 파싱 ────────────────────────────────────────────

class FileMeta(NamedTuple):
    # PDF 파일명에서 뽑아낸 구조화된 메타데이터
    # 예: "서울_주거_청년월세지원.pdf" → region="서울", category="주거", doc_name="청년월세지원"
    region: str
    category: str
    doc_name: str


def parse_filename(filename: str) -> FileMeta:
    """파일명 규칙 {지역}_{카테고리}_{문서명}.pdf 에서 메타데이터 파싱."""
    stem = Path(filename).stem         # ".pdf" 확장자 제거한 파일명
    parts = stem.split("_", 2)         # 최대 3덩어리로 분리 (언더스코어 기준)
    if len(parts) >= 3:
        return FileMeta(region=parts[0], category=parts[1], doc_name=parts[2])
    if len(parts) == 2:
        # 문서명이 없는 경우 → 파일명 전체를 doc_name으로 사용
        return FileMeta(region=parts[0], category=parts[1], doc_name=stem)
    # 언더스코어가 없는 파일 → 알 수 없음 처리
    return FileMeta(region="unknown", category="unknown", doc_name=stem)


def _safe_id(text: str) -> str:
    # Pinecone 벡터 ID에 특수문자가 들어가면 에러가 나므로
    # 영숫자·하이픈·언더스코어 외의 문자를 언더스코어로 치환
    return re.sub(r"[^\w\-]", "_", text)


# ── PDF 로딩 ──────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> list[Document]:
    # 파일명에서 메타데이터(지역, 분류, 문서명) 추출
    meta = parse_filename(pdf_path.name)

    # LangChain의 PyPDFLoader: PDF를 페이지 단위 Document 리스트로 변환
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()  # 각 페이지 = Document(page_content=텍스트, metadata={page:0, ...})

    # 각 페이지의 metadata에 우리가 필요한 정보를 추가
    for page in pages:
        page.metadata.update(
            {
                "region": meta.region,           # 지역 (검색 필터로 사용 가능)
                "category": meta.category,       # 정책 분류
                "doc_name": meta.doc_name,       # 문서명
                "source_file": pdf_path.name,    # 원본 파일명 (출처 표시용)
            }
        )
    return pages


# ── 청크 ID 생성 ──────────────────────────────────────────────────────

def _make_ids(chunks: list[Document], stem: str) -> list[str]:
    """청크별 결정적 ID: {stem}-p{page}-c{chunk_within_page} — upsert 중복 방지."""
    # 같은 PDF를 두 번 ingest해도 ID가 동일하면 Pinecone이 덮어쓰기(upsert)를 하므로
    # 중복 저장 없이 안전하게 업데이트된다.
    page_counts: dict[int, int] = defaultdict(int)  # 페이지별 청크 번호 카운터
    ids: list[str] = []
    for chunk in chunks:
        page = int(chunk.metadata.get("page", 0))
        idx = page_counts[page]          # 해당 페이지에서 몇 번째 청크인지
        ids.append(f"{stem}-p{page}-c{idx}")
        page_counts[page] += 1
    return ids


# ── 디렉토리 전체 Pinecone 적재 ───────────────────────────────────────

def ingest_directory(path: Path) -> tuple[int, int]:
    """PDF 디렉토리를 Pinecone에 upsert. (적재 문서 수, 총 청크 수) 반환."""
    # path 안의 모든 .pdf 파일을 알파벳 순으로 가져옴
    pdf_files = sorted(path.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]경고: {path} 에 PDF 파일이 없습니다.[/yellow]")
        return 0, 0

    # ── Gemini 임베딩 모델 초기화 ──────────────────────────────────
    # 텍스트 → 768차원 숫자 벡터로 변환해주는 모델
    # "embedding-001"은 Gemini의 텍스트 임베딩 전용 모델
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.google_api_key,
    )

    # ── 기존 Pinecone 인덱스에 연결 ────────────────────────────────
    # 인덱스는 Pinecone 콘솔에서 미리 만들어 둬야 한다.
    # (인덱스 생성은 별도 스크립트 또는 Pinecone 웹에서 수동으로)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
    )

    # 텍스트 분할기 (1000자 청크, 150자 overlap)
    splitter = get_splitter()

    total_docs = 0
    total_chunks = 0

    # track(): Rich의 진행바를 보여주면서 반복 (tqdm과 유사)
    for pdf_path in track(pdf_files, description="PDF 처리 중..."):
        try:
            pages = load_pdf(pdf_path)           # PDF → 페이지 리스트
            chunks = splitter.split_documents(pages)  # 페이지 → 작은 청크 리스트
            stem = _safe_id(pdf_path.stem)       # 파일명 → 안전한 ID 접두사
            ids = _make_ids(chunks, stem)        # 각 청크의 고유 ID 생성

            # Pinecone에 저장: 청크 텍스트를 벡터로 변환 후 upsert
            vectorstore.add_documents(chunks, ids=ids)

            total_docs += 1
            total_chunks += len(chunks)
            console.print(f"  [green]✓[/green] {pdf_path.name} → {len(chunks)}청크")
        except Exception as e:
            # 한 파일이 실패해도 나머지 파일은 계속 처리
            console.print(f"  [red]✗[/red] {pdf_path.name}: {e}")

    return total_docs, total_chunks
