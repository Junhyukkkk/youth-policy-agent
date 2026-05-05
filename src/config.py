# ============================================================
# config.py — 프로젝트 전체 환경변수 관리
#
# 역할: .env 파일에 저장된 API 키와 설정값을 읽어서
#       프로젝트 어디서든 settings.변수명 으로 꺼내 쓸 수 있게 해준다.
#       (직접 os.environ["KEY"] 를 쓰는 것보다 안전하고 타입 검증도 된다)
# ============================================================

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env 파일을 UTF-8로 읽도록 설정
    # pydantic이 자동으로 .env → 클래스 필드 매핑
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── 필수 환경변수 (없으면 앱 시작 시 바로 에러남) ──────────────
    google_api_key: str          # Gemini LLM + 임베딩 모델 인증키
    pinecone_api_key: str        # 벡터 DB(Pinecone) 인증키

    # ── 선택 환경변수 (기본값 있음) ───────────────────────────────
    pinecone_index_name: str = "youth-policy-index"  # Pinecone에 만들어 둔 인덱스 이름
    youth_policy_api_key: str = ""                   # 온통청년 OpenAPI 인증키 (MCP 기능 사용 시 필요)
    gemini_model: str = "gemini-2.5-flash"           # 사용할 Gemini 모델 이름

    # ── RAG 청킹 파라미터 ─────────────────────────────────────────
    chunk_size: int = 1000    # 한 청크의 최대 글자 수 (너무 크면 검색 정밀도↓, 너무 작으면 맥락↓)
    chunk_overlap: int = 150  # 인접 청크끼리 겹치는 글자 수 (문장이 중간에 잘리지 않도록)


# 모듈 최상단에서 한 번만 인스턴스를 만든다.
# 다른 파일에서는 "from src.config import settings" 로 임포트해서 바로 사용.
settings = Settings()
