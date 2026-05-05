# ============================================================
# rag/splitter.py — PDF 텍스트를 일정 크기로 나누는 도구(스플리터) 생성
#
# 역할: PDF에서 추출한 긴 텍스트를 Pinecone에 저장하기 적합한
#       작은 조각(청크)으로 잘라주는 스플리터 객체를 반환한다.
#
# 왜 잘라야 하나?
#   - LLM에는 한 번에 보낼 수 있는 텍스트 양에 한계(컨텍스트 윈도우)가 있다.
#   - 벡터 검색은 짧고 의미 집중적인 단위일수록 정밀도가 높다.
#   - 따라서 PDF 전체를 통째로 저장하지 않고 1000자 단위로 쪼개서 저장한다.
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def get_splitter() -> RecursiveCharacterTextSplitter:
    # RecursiveCharacterTextSplitter:
    #   문단(\n\n) → 줄바꿈(\n) → 문장 → 단어 순서로 재귀적으로 나눈다.
    #   가능한 한 자연스러운 경계(문단, 줄)에서 잘라줌.
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,       # 한 청크의 최대 글자 수 (기본 1000)
        chunk_overlap=settings.chunk_overlap, # 앞뒤 청크가 겹치는 글자 수 (기본 150)
        # overlap이 있어야 문장이 청크 경계에서 끊겨도 의미가 유실되지 않음
    )
