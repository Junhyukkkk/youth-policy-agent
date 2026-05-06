# Youth Policy Agent

청년 정책 RAG + MCP 기반 AI 에이전트. CLI 환경에서 동작.

## 아키텍처

이중 데이터 소스 라우팅 구조.

- **RAG (정적)**: 정책 가이드라인 PDF → 청킹 → Pinecone 임베딩
- **MCP (동적)**: 온통청년 OpenAPI 실시간 호출 (목록/상세)
- **Agent**: 쿼리 성격에 따라 RAG / MCP 라우팅, 출처 강제

질문 유형별 라우팅 원칙:
- "어떤 지원이 있어?", "자격 조건은?" → RAG
- "지금 열린 공고 있어?", "마감일은?" → MCP
- MCP 실패 시 → RAG fallback

## 스택

- Python 3.11+, Poetry
- LangChain + Gemini 1.5 (google-generativeai)
- Pinecone (Starter 무료 플랜)
- mcp (공식 SDK), httpx
- Rich (CLI 렌더링)
- pydantic, python-dotenv

## 디렉토리 구조

```
youth-policy-agent/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env.example
├── docs/
│   ├── plan.md
│   ├── day-1.md ~ day-7.md
├── data/
│   └── policies/        # 정책 PDF 원본
├── src/
│   ├── agent/
│   │   ├── orchestrator.py   # LLM + 라우팅
│   │   └── prompts.py
│   ├── rag/
│   │   ├── ingest.py         # PDF 로드/청킹/임베딩
│   │   ├── retriever.py      # 메타데이터 필터 retriever
│   │   └── splitter.py
│   ├── mcp_server/
│   │   ├── server.py         # MCP stdio 서버
│   │   ├── tools.py          # get_policy_list, get_policy_detail
│   │   └── client.py         # 온통청년 API httpx 클라이언트
│   ├── cli/
│   │   └── main.py           # entry point (Rich)
│   └── config.py             # env 로딩
└── tests/
```

## 컨벤션

- 함수/변수: snake_case, 클래스: PascalCase
- 타입 힌트 필수
- 환경변수는 `src/config.py`에서만 접근 (직접 `os.environ` 금지)
- API 키 등 민감값은 절대 로그에 찍지 않기
- 외부 API 호출은 반드시 timeout + try/except
- LLM 응답은 출처(source) 메타데이터 함께 반환

## 개발 원칙

- 한 Day의 완료 기준을 통과해야 다음 Day로 진행
- 각 Day는 별도 Claude Code 세션으로 시작 (컨텍스트 분리)
- 작업 시작 전 `docs/day-N.md` 먼저 읽기
- 커밋 단위는 체크박스 1개 단위 권장

## 금지 사항

- API 키 하드코딩 (반드시 .env)
- Pinecone 외 다른 벡터 DB로 임의 변경
- LLM 라우팅 없이 단일 소스만 사용하는 구조로 단순화

## Pinecone 사용 메모

- Starter (무료) 플랜 기준으로 설계
- 인덱스 1개로 충분 (이름: `youth-policy-index`)
- 리전: AWS us-east-1 (Starter 제한)
- 임베딩 차원: `gemini-embedding-001` 사용 시 **3072차원** (768 아님)
- PDF 파일명 규칙: `{region}_{category}_{docname}.pdf` — **반드시 ASCII 영문**
  - Pinecone Vector ID가 ASCII만 허용하므로 한글 파일명 사용 불가
  - 예: `national_finance_youth_hope_account_faq.pdf`
  - region: national / seoul / busan 등, category: finance / housing / policy 등
- 3주 비활성 시 인덱스 paused → 다시 호출하면 자동 깨어남
