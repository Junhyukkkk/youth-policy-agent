# Youth Policy Agent

> 청년 정책을 자연어로 물어보면, 가이드라인은 RAG로 / 실시간 공고는 MCP로 조회해서 답해주는 CLI 에이전트.

```bash
$ policy-agent ask "서울 청년 월세 지원 지금 신청 가능한 거 있어?"

[MCP] 실시간 공고 조회 결과

서울특별시에서 현재 모집 중인 청년 월세 지원 사업이 2건 있습니다.
1. 서울시 청년월세 한시 특별지원 (마감: 2026-06-30)
2. 청년 부동산 중개보수 지원 (마감: 2026-05-15)

📎 출처: 온통청년 API (정책ID: R2026-0312, R2026-0287)
```

## 왜 만들었나

청년 정책 정보는 두 종류다.

- **잘 안 변하는 정보** — 자격 요건, 신청 절차, 지원 내용 같은 가이드라인
- **계속 변하는 정보** — 어떤 공고가 지금 열려있는지, 마감일은 언제인지

이 둘을 한 LLM에 통째로 욱여넣으면 답변이 부정확해진다. 가이드라인은 PDF에 잘 정리돼 있는데 굳이 매번 API를 칠 필요가 없고, 실시간 공고는 PDF에 박제하면 금방 낡는다.

그래서 두 데이터 소스를 분리하고, LLM이 질문 성격에 따라 알아서 라우팅하도록 만들었다.

- 정적 가이드라인 → **RAG** (벡터 검색)
- 실시간 공고 → **MCP** (외부 API 호출)
- 외부 API 실패 시 → **RAG로 fallback**

## 아키텍처

```
                         ┌──────────────────┐
                         │   사용자 쿼리    │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  LLM (Gemini)    │
                         │  + Tool Binding  │
                         └────────┬─────────┘
                                  │ 라우팅 판단
                  ┌───────────────┼───────────────┐
                  │               │               │
            ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
            │ RAG Tool  │   │ MCP Tool  │   │ MCP Tool  │
            │ (검색)    │   │ (목록)    │   │ (상세)    │
            └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                  │               │               │
            ┌─────▼─────┐   ┌─────▼───────────────▼─────┐
            │ Pinecone  │   │   온통청년 OpenAPI         │
            │ (정책 PDF)│   │   (실시간 공고 데이터)     │
            └───────────┘   └────────────────────────────┘
```

질문이 들어오면 LLM이 Tool description을 보고 어느 경로로 갈지 결정한다. 외부 API가 죽으면 RAG로 fallback해서 서비스가 끊기지 않게 했다.

## 기술 스택

| 영역 | 사용 기술 | 선택 이유 |
|------|----------|----------|
| 언어 | Python 3.11+ | LangChain/MCP 생태계가 가장 성숙 |
| LLM | Gemini 1.5 Flash | 한국어 품질 + 비용 |
| 프레임워크 | LangChain | Tool Binding, Retriever 추상화 |
| 벡터 DB | Pinecone | 관리형 인프라, 빠른 프로토타이핑 |
| 외부 통합 | MCP (Model Context Protocol) | 외부 API를 Tool로 표준화 |
| HTTP | httpx (async) | timeout/retry가 깔끔 |
| CLI | Rich | 출처/Tool 뱃지 시각화 |
| 패키지 | Poetry | 의존성/스크립트 관리 |

## 빠른 시작

### 1. 사전 준비 (모두 무료)

- Python 3.11+
- [Pinecone 계정](https://www.pinecone.io) (Starter 무료 플랜)
- [Google AI Studio](https://aistudio.google.com) — Gemini API Key
- [공공데이터포털](https://www.data.go.kr) — 온통청년 API Key

### 2. 클론 및 설치

```bash
git clone https://github.com/Junhyukkkk/youth-policy-agent.git
cd youth-policy-agent

poetry install
cp .env.example .env
# .env 파일 열어서 API 키 3개 입력
```

### 3. Pinecone 인덱스 생성

```bash
poetry run policy-agent init-index
# → 'youth-policy-index' 인덱스 자동 생성 (이미 있으면 skip)
```

### 4. 정책 PDF 적재

```bash
# data/policies/ 디렉토리에 정책 PDF 배치
# 파일명 규칙: {지역}_{카테고리}_{문서명}.pdf
# 예: 서울_주거_청년월세지원.pdf

poetry run policy-agent ingest --path data/policies/
```

### 5. 질문해보기

```bash
poetry run policy-agent ask "서울 청년 월세 지원 자격이 어떻게 돼?"
```

## 환경변수

| 변수명 | 설명 | 필수 |
|--------|------|------|
| `GOOGLE_API_KEY` | Gemini API 키 | ✅ |
| `PINECONE_API_KEY` | Pinecone API 키 | ✅ |
| `PINECONE_INDEX_NAME` | 인덱스 이름 (기본: `youth-policy-index`) | ❌ |
| `YOUTH_POLICY_API_KEY` | 온통청년 OpenAPI 키 | ✅ |
| `GEMINI_MODEL` | 사용할 모델 (기본: `gemini-1.5-flash`) | ❌ |

## 사용 예시

### 가이드라인 질의 (RAG 라우팅)

```bash
$ policy-agent ask "청년도약계좌 가입 조건이 뭐야?"

[RAG] 정책 가이드라인 검색 결과

만 19~34세 청년이며, 개인소득 7,500만 원 이하, 가구소득
중위 180% 이하 조건을 충족해야 합니다. ...

📎 출처: 서울_금융_청년도약계좌.pdf (p.3)
```

### 실시간 공고 질의 (MCP 라우팅)

```bash
$ policy-agent ask "지금 열려있는 청년 일자리 공고 보여줘"

[MCP] 실시간 공고 조회

현재 모집 중인 일자리 지원 사업: ...
```

### Fallback 동작

```bash
# 인터넷 연결이 끊기거나 API가 응답하지 않을 때
$ policy-agent ask "지금 모집 중인 공고 알려줘"

⚠️ [RAG-FALLBACK] 실시간 조회 실패, 저장된 가이드라인 기반 응답

실시간 공고 정보는 현재 조회할 수 없습니다.
저장된 자료에 따르면 ... (가이드라인 기반 답변)
```

## 프로젝트 구조

```
youth-policy-agent/
├── CLAUDE.md                  # 개발 가이드 (Claude Code용)
├── docs/                      # 7-Day 개발 플랜
├── data/policies/             # 정책 PDF 원본
├── src/
│   ├── agent/
│   │   ├── orchestrator.py    # LLM + Tool 라우팅
│   │   └── prompts.py         # 라우팅/출처 강제 프롬프트
│   ├── rag/
│   │   ├── ingest.py          # PDF 청킹 + Pinecone 적재
│   │   └── retriever.py       # 메타데이터 필터 검색
│   ├── mcp_server/
│   │   ├── server.py          # MCP stdio 서버
│   │   ├── tools.py           # get_policy_list, get_policy_detail
│   │   └── client.py          # 온통청년 API 클라이언트
│   ├── cli/main.py            # Rich 기반 CLI
│   └── config.py              # 환경변수 로딩
└── tests/
```

## 설계 의사결정

### 왜 RAG와 MCP를 분리했나

단일 RAG로 가면 실시간 공고 데이터를 주기적으로 재임베딩해야 한다. 데이터 신선도 SLA를 RAG에 두는 건 비효율적이라 판단했다. 반대로 모든 걸 MCP(API 호출)로 처리하면 LLM이 매 질문마다 외부 API를 치게 되고, 가이드라인 같은 정적 정보 조회가 비싸진다.

**라우팅 판단의 책임은 LLM에 위임**하되, Tool description을 명확히 작성해서 정확도를 끌어올렸다.

### 왜 Pinecone을 선택했나

프로토타입 단계에서는 인프라 관리 비용을 최소화하고 핵심 로직(라우팅, MCP 통합, 환각 방지)에 집중하는 게 우선이었다. Pinecone Starter 플랜은 무료이면서 2GB / 200만 write units / 100만 read units를 제공하는데, 이 프로젝트 규모에는 충분하다.

스케일이 커져 비용이나 데이터 주권 이슈가 생기면 pgvector(Postgres) 또는 Weaviate로 마이그레이션을 검토할 수 있다. LangChain의 `VectorStore` 추상화 덕분에 교체 비용은 낮다.

### 왜 Fallback을 RAG로 두고 MCP는 안 두나

RAG에 실시간 공고를 박제하면 오답을 자신 있게 하는 최악의 경우가 생긴다. 반면 가이드라인은 빠르게 안 변하므로 MCP 실패 시 RAG로 떨어뜨려도 정보 정합성에 큰 문제가 없다. 단, 사용자에게는 fallback 사실을 명시하도록 했다.

### 왜 환각 방지에 출처 강제를 썼나

LLM이 그럴듯한 답을 만들어내는 건 막기 어렵다. 그래서 시스템 프롬프트에서 "검색 결과에 없으면 '확인된 정보 없음'으로 답할 것"을 강제하고, 답변 후처리로 출처에 없는 고유명사/숫자가 등장하면 경고를 띄운다. 완벽하진 않지만 휴리스틱으로도 체감 정확도가 올라간다.

## 트러블슈팅

**Pinecone 인덱스가 안 보여요**
- Starter 플랜은 3주 비활성 시 인덱스가 자동 paused 됨. 다시 호출하면 깨어남.
- 콘솔에서 인덱스 상태 확인: https://app.pinecone.io

**임베딩 차원 오류**
- 인덱스 생성 시 차원과 임베딩 모델 차원이 일치해야 함 (Gemini `embedding-001` = 768).
- 차원 변경 불가, 인덱스 삭제 후 재생성 필요.

**MCP 서버 연결 실패**
- `mcp` SDK 버전 확인 (자주 변경됨)
- stdio 통신은 stdout이 JSON-RPC 전용. 디버그 출력은 stderr로.

## 향후 계획

- [ ] LangGraph 기반 멀티 스텝 에이전트로 확장 (현재는 단일 라우팅)
- [ ] 사용자 프로필 기반 정책 추천 (나이/지역/관심분야)
- [ ] 웹 인터페이스 (FastAPI + 간단한 프론트)
- [ ] 평가 데이터셋 구축 및 라우팅 정확도 측정 자동화
- [ ] pgvector 마이그레이션 (스케일 확장 시)

## 라이선스

MIT
