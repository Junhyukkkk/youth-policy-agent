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

import pdfplumber
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from rich.console import Console
from rich.progress import track

from src.config import settings
from src.rag.embeddings import GeminiEmbeddings
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

def _table_to_text(table: list[list]) -> str:
    """pdfplumber 표 데이터를 읽기 쉬운 텍스트로 변환.

    예: [["소득분위", "지원금액"], ["1분위", "240,000원"]]
        → "소득분위 | 지원금액\n1분위 | 240,000원"
    표를 통째로 한 줄로 붙이는 것보다 행 단위로 나눠야
    청킹 후에도 각 행의 맥락(어느 열인지)이 유지된다.
    """
    rows = []
    for row in table:
        # None 셀은 빈 문자열로 치환, 셀 사이는 " | " 구분
        rows.append(" | ".join(str(cell or "").strip() for cell in row))
    return "\n".join(rows)


def load_pdf(pdf_path: Path) -> list[Document]:
    """pdfplumber로 PDF를 페이지 단위 Document 리스트로 변환.

    PyPDFLoader 대비 개선점:
    - 다단 편집 레이아웃의 텍스트 순서를 더 정확하게 복원
    - 표(Table)를 별도로 추출해서 본문 뒤에 구조화된 형태로 추가
      (표 셀이 뒤섞이는 문제 방지)
    """
    meta = parse_filename(pdf_path.name)
    pages = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            # x_tolerance/y_tolerance: 같은 줄로 볼 글자 간격 허용치 (픽셀)
            # 값이 너무 크면 다른 줄 텍스트가 합쳐지고, 너무 작으면 단어가 쪼개짐
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

            # 페이지 안의 표를 모두 찾아서 텍스트 뒤에 추가
            # extract_tables()는 표를 행·열 리스트로 반환
            tables = page.extract_tables()
            if tables:
                table_texts = [_table_to_text(t) for t in tables if t]
                # 본문과 표 사이에 빈 줄을 두어 청킹 시 자연스럽게 분리되도록
                text = text + "\n\n" + "\n\n".join(table_texts)

            pages.append(Document(
                page_content=text,
                metadata={
                    "region": meta.region,
                    "category": meta.category,
                    "doc_name": meta.doc_name,
                    "source_file": pdf_path.name,
                    "page": i,
                },
            ))

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
    # google-genai SDK (v1 API) 기반 커스텀 클래스 사용
    # langchain-google-genai의 GoogleGenerativeAIEmbeddings는 v1beta를 써서 404 에러 발생
    embeddings = GeminiEmbeddings()

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
