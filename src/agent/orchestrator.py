# ============================================================
# agent/orchestrator.py — 에이전트의 심장부: LLM + 도구 라우팅 + 루프
#
# 역할: 사용자 질문을 받아서 LLM이 적절한 도구(RAG/MCP)를 선택하게 하고,
#       도구 실행 결과를 다시 LLM에 넘겨 최종 자연어 답변을 만든다.
#
# 핵심 개념 — ReAct 패턴 (Reasoning + Acting):
#   1. LLM이 질문을 보고 "어떤 도구를 쓸지" 결정 (Reasoning)
#   2. 결정한 도구를 실행해서 결과를 가져옴 (Acting)
#   3. 결과를 보고 LLM이 최종 답변 생성 (또는 다시 1번으로)
#   → 이 사이클을 최대 _MAX_ITER번 반복
#
# 도구 3종:
#   - rag_search: PDF 기반 정적 정보 검색 (Pinecone)
#   - get_policy_list: 실시간 정책 목록 (온통청년 API)
#   - get_policy_detail: 실시간 정책 상세 (온통청년 API)
#
# MCP 실패 시 자동 RAG fallback:
#   온통청년 API 호출이 실패하면 자동으로 RAG 검색으로 전환하고
#   LLM에게 "API 실패로 과거 자료 기반 답변임"을 알린다.
# ============================================================
"""청년정책 에이전트 오케스트레이터.

LLM에 rag_search / get_policy_list / get_policy_detail 세 도구를 바인딩하고,
쿼리 성격에 따라 LLM이 적절한 도구를 선택·실행한 뒤 자연어 답변을 생성한다.
MCP 도구 실패 시 RAG fallback으로 전환한다.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from langchain_core.messages import (
    AIMessage,      # LLM이 생성한 메시지 (도구 호출 포함 가능)
    BaseMessage,
    HumanMessage,   # 사용자 메시지
    SystemMessage,  # 시스템 프롬프트 메시지
    ToolMessage,    # 도구 실행 결과 메시지
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.agent.prompts import SYSTEM_PROMPT
from src.config import settings
from src.mcp_server.tools import call_tool
from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)

_MAX_ITER = 5       # 도구 호출 최대 반복 횟수 (무한 루프 방지)
_MCP_TOOLS = {"get_policy_list", "get_policy_detail"}  # 실시간 API 도구 이름 집합

# MCP 실패 시 LLM에게 전달하는 fallback 안내 메시지
# 이 메시지가 있으면 LLM이 prompts.py의 "실시간 API 실패 시" 규칙을 따름
_FALLBACK_PREFIX = (
    "⚠️ 실시간 API 호출 실패 — RAG 기반 정보로 대체합니다.\n"
    "답변 시작 부분에 '⚠️ 실시간 정보 조회에 실패하여 기존 자료 기반으로 답변합니다.'를 반드시 명시하세요.\n\n"
)

# 2자리 이상 숫자 패턴 (환각 감지용)
# 예: "30만원", "19세", "2024년" 같은 구체적인 수치가 출처에 없으면 환각 의심
_NUM_PATTERN = re.compile(r"\d{2,}")


# ============================================================
# 응답 구조체 — ask() 메서드가 반환하는 결과 객체
# ============================================================


@dataclass
class AgentResponse:
    answer: str                          # LLM이 생성한 최종 답변 텍스트
    sources: list[str] = field(default_factory=list)      # RAG 출처 목록 (파일명·페이지)
    tools_used: list[str] = field(default_factory=list)   # 실제로 사용된 도구 이름 목록
    fallback_used: bool = False           # MCP 실패 → RAG fallback 발생 여부
    hallucination_warning: bool = False   # 출처 미확인 숫자 포함 여부


def _check_hallucination(answer: str, source_texts: list[str], query: str = "") -> bool:
    """답변에 출처·질문 어디에도 없는 2자리 이상 숫자가 있으면 True를 반환한다.

    원리: 도구 결과(source_texts)와 질문(query)에 있는 숫자를 기준으로,
          LLM 답변에 그 외의 숫자가 등장하면 LLM이 스스로 만들어낸 것이므로 환각 의심.
    예: 출처에 없는 "300만원", "24세" 같은 수치가 답변에 있으면 경고.
    """
    if not source_texts:
        return False
    combined = " ".join(source_texts) + " " + query
    for num in set(_NUM_PATTERN.findall(answer)):  # 답변에서 숫자 추출
        if num not in combined:                    # 출처+질문 어디에도 없는 숫자면
            logger.debug("[hallucination] 출처 미확인 숫자: %s", num)
            return True
    return False


# ============================================================
# Tool 정의 — LangChain @tool 데코레이터로 LLM이 호출할 수 있는 함수 등록
# ============================================================


@tool
def rag_search(query: str) -> str:
    """정책의 자격요건, 신청절차, 지원내용 등 일반적인 안내사항을 검색합니다.
    변하지 않는 정책 가이드라인 정보를 찾을 때 사용하세요.
    예: '청년 월세 지원 자격', '청년도약계좌 신청 절차', '일자리 지원 대상 조건'"""
    # retriever.py의 retrieve()를 호출해 Pinecone에서 유사 청크 4개 검색
    docs = retrieve(query, k=4)
    if not docs:
        return "관련 정보를 찾을 수 없습니다."

    # 각 청크를 "[출처: 파일명, p.페이지]\n본문" 형식으로 포맷팅
    chunks: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source_file", "알 수 없음")
        page = doc.metadata.get("page", "")
        tag = f"[출처: {source}" + (f", p.{page}]" if page else "]")
        chunks.append(f"{tag}\n{doc.page_content}")

    # 청크들을 구분선으로 연결해서 하나의 문자열로 반환
    # LLM이 이 텍스트를 읽고 답변 생성에 활용함
    return "\n\n---\n\n".join(chunks)


# LLM이 get_policy_list를 호출할 때 넘길 파라미터의 타입/설명 정의
class _PolicyListInput(BaseModel):
    keyword: str = Field(default="", description="정책 키워드. 예: '주거지원', '청년도약'")
    region: str = Field(
        default="", description="법정시군구코드 5자리. 예: '11000'=서울, '26110'=부산"
    )
    category: str = Field(
        default="",
        description="정책 대분류. 예: '일자리', '주거', '교육', '금융･복지･문화'",
    )
    policy_name: str = Field(default="", description="정책명 검색어")
    page: int = Field(default=1, description="페이지 번호")
    page_size: int = Field(default=10, description="페이지당 결과 수")


class _PolicyDetailInput(BaseModel):
    policy_no: str = Field(description="정책 번호 (plcyNo). get_policy_list 결과에서 획득")


@tool(args_schema=_PolicyListInput)
def get_policy_list(
    keyword: str = "",
    region: str = "",
    category: str = "",
    policy_name: str = "",
    page: int = 1,
    page_size: int = 10,
) -> str:
    """현재 모집 중인 청년 정책 공고의 실시간 목록을 조회합니다.
    마감일, 신규 공고, 지금 신청 가능한 정책 등 시점에 따라 변하는 최신 정보가 필요할 때 사용하세요.
    예: '지금 열린 주거 지원 공고', '서울 청년 일자리 공고', '이번 달 마감 정책'"""
    # asyncio.run(): 이 함수는 동기 함수지만 내부적으로 비동기 API 호출이 필요하므로
    # asyncio.run()으로 비동기 코루틴을 동기 컨텍스트에서 실행
    results = asyncio.run(
        call_tool(
            "get_policy_list",
            {
                "keyword": keyword,
                "region": region,
                "category": category,
                "policy_name": policy_name,
                "page": page,
                "page_size": page_size,
            },
        )
    )
    return results[0].text if results else "결과 없음"


@tool(args_schema=_PolicyDetailInput)
def get_policy_detail(policy_no: str) -> str:
    """특정 청년 정책의 상세 정보를 조회합니다.
    신청 방법, 지원 내용, 자격 조건, 신청 기간 등 구체적인 정보가 필요할 때 사용하세요.
    반드시 get_policy_list로 얻은 policy_no(정책번호)를 먼저 확인한 후 사용하세요."""
    results = asyncio.run(
        call_tool("get_policy_detail", {"policy_no": policy_no})
    )
    return results[0].text if results else "결과 없음"


# 등록된 도구 리스트 — LLM이 선택할 수 있는 도구 전체
_TOOLS = [rag_search, get_policy_list, get_policy_detail]
# 이름으로 빠르게 찾기 위한 딕셔너리 {"rag_search": <function>, ...}
_TOOLS_BY_NAME: dict[str, object] = {t.name: t for t in _TOOLS}  # type: ignore[assignment]


# ============================================================
# 에이전트 클래스 — 질문 → 도구 실행 → 답변 생성 루프
# ============================================================


class PolicyAgent:
    """RAG + MCP Tool 라우팅 에이전트."""

    def __init__(self) -> None:
        # Gemini LLM 초기화
        # temperature=0.0: 창의성 없이 주어진 정보만 사용 (환각 방지)
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
        )
        # bind_tools(): LLM에게 "이 도구들을 호출할 수 있다"고 알려줌
        # LLM이 응답할 때 tool_calls 필드에 호출할 도구와 파라미터를 채워 반환함
        self._llm_with_tools = _llm.bind_tools(_TOOLS)

    def ask(self, query: str) -> AgentResponse:
        """쿼리를 받아 Tool을 선택·실행하고 자연어 답변을 반환한다.

        대화 메시지 구조 (messages 리스트):
            [SystemMessage(시스템 프롬프트)]
            [HumanMessage(사용자 질문)]
            [AIMessage(LLM 응답 — tool_calls 포함)]
            [ToolMessage(도구 실행 결과)]
            [AIMessage(최종 답변)]  ← tool_calls가 없으면 루프 종료
        """
        # 대화 히스토리 초기화 — 시스템 프롬프트 + 사용자 질문으로 시작
        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        tools_used: list[str] = []      # 이번 ask()에서 사용된 도구 이름 기록
        sources: list[str] = []         # RAG 출처 목록
        tool_results: list[str] = []    # 도구 실행 결과 텍스트 (환각 감지에 사용)
        fallback_used: bool = False      # MCP fallback 발생 여부

        # ── ReAct 루프: 최대 _MAX_ITER(5)번 반복 ─────────────────────
        for _ in range(_MAX_ITER):
            # LLM 호출: 현재까지의 대화를 넘기고 다음 응답을 받음
            response: AIMessage = self._llm_with_tools.invoke(messages)
            logger.debug("[llm-response] tool_calls=%s", response.tool_calls)
            messages.append(response)  # LLM 응답을 대화 히스토리에 추가

            # tool_calls가 없으면 LLM이 최종 답변을 생성한 것 → 루프 종료
            if not response.tool_calls:
                break

            # ── 도구 실행 ───────────────────────────────────────────
            for tc in response.tool_calls:
                name: str = tc["name"]    # 호출할 도구 이름
                args: dict = tc["args"]   # 도구에 전달할 파라미터
                tools_used.append(name)
                logger.info("[tool] %s %s", name, args)

                tool_fn = _TOOLS_BY_NAME.get(name)
                if tool_fn is None:
                    # LLM이 존재하지 않는 도구를 요청한 경우
                    result: str = f"알 수 없는 tool: {name}"
                    mcp_failed = False
                else:
                    try:
                        # 실제 도구 실행 (rag_search → Pinecone, get_policy_* → API)
                        result = tool_fn.invoke(args)  # type: ignore[union-attr]
                        mcp_failed = False
                    except Exception as exc:
                        logger.warning("[tool-error] %s: %s", name, exc)
                        if name in _MCP_TOOLS:
                            # ★ MCP(실시간 API) 실패 → RAG fallback 자동 전환
                            result = self._rag_fallback(query, args)
                            tools_used.append("rag_search")  # fallback도 도구 사용 기록
                            fallback_used = True
                            mcp_failed = True
                            logger.info(
                                "[mcp-fallback] %s 실패 → RAG fallback 실행", name
                            )
                        else:
                            # RAG 자체가 실패한 경우 (비정상 상황)
                            result = f"Tool 오류: {exc}"
                            mcp_failed = False

                # RAG 결과(또는 fallback 결과)에서 출처 정보 추출
                # "[출처: 파일명, p.3]" 형태의 줄을 찾아서 sources 리스트에 저장
                if (name == "rag_search" or mcp_failed) and isinstance(result, str):
                    for line in result.splitlines():
                        if line.startswith("[출처:"):
                            sources.append(line.strip("[]").replace("출처: ", ""))

                tool_results.append(str(result))

                # fallback인 경우 LLM에게 "API 실패" 사실을 알리는 접두어 추가
                tool_content = (
                    _FALLBACK_PREFIX + result if mcp_failed else str(result)
                )
                # 도구 실행 결과를 대화 히스토리에 추가
                # tool_call_id: LLM이 요청한 tool_call과 결과를 매칭하는 ID
                messages.append(
                    ToolMessage(content=tool_content, tool_call_id=tc["id"])
                )
        # ── 루프 종료 후 마지막 메시지에서 최종 답변 추출 ─────────────
        last = messages[-1]
        answer = getattr(last, "content", "") or ""
        if not answer:
            answer = "확인된 정보 없음"

        # 환각 감지: 도구 결과에 없는 숫자가 답변에 있는지 확인
        hallucination_warning = _check_hallucination(answer, tool_results, query)
        if hallucination_warning:
            logger.warning("[hallucination] 출처 미확인 숫자가 답변에 포함됨")

        return AgentResponse(
            answer=answer,
            sources=sources,
            tools_used=tools_used,
            fallback_used=fallback_used,
            hallucination_warning=hallucination_warning,
        )

    @staticmethod
    def _rag_fallback(query: str, failed_args: dict) -> str:
        """MCP 실패 시 RAG로 대체 검색한다.

        검색 키워드 우선순위:
          1. MCP 호출 시 사용한 keyword 파라미터
          2. MCP 호출 시 사용한 policy_name 파라미터
          3. 원래 사용자 질문 전체
        """
        fallback_query = (
            failed_args.get("keyword")
            or failed_args.get("policy_name")
            or query  # 위 둘이 없으면 원래 질문으로 RAG 검색
        )
        try:
            return rag_search.invoke({"query": fallback_query})  # type: ignore[return-value]
        except Exception as exc:
            logger.error("[rag-fallback-error] %s", exc)
            return "RAG 조회도 실패했습니다. 잠시 후 다시 시도해 주세요."
