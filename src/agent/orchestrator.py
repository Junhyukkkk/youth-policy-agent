"""청년정책 에이전트 오케스트레이터.

LLM에 rag_search / get_policy_list / get_policy_detail 세 도구를 바인딩하고,
쿼리 성격에 따라 LLM이 적절한 도구를 선택·실행한 뒤 자연어 답변을 생성한다.
MCP 도구 실패 시 RAG fallback으로 전환한다.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.agent.prompts import SYSTEM_PROMPT
from src.config import settings
from src.mcp_server.tools import call_tool
from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)

_MAX_ITER = 5
_MCP_TOOLS = {"get_policy_list", "get_policy_detail"}

_FALLBACK_PREFIX = (
    "⚠️ 실시간 API 호출 실패 — RAG 기반 정보로 대체합니다.\n"
    "답변 시작 부분에 '⚠️ 실시간 정보 조회에 실패하여 기존 자료 기반으로 답변합니다.'를 반드시 명시하세요.\n\n"
)


# ---------------------------------------------------------------------------
# 응답 구조체
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Tool 정의
# ---------------------------------------------------------------------------


@tool
def rag_search(query: str) -> str:
    """정책의 자격요건, 신청절차, 지원내용 등 일반적인 안내사항을 검색합니다.
    변하지 않는 정책 가이드라인 정보를 찾을 때 사용하세요.
    예: '청년 월세 지원 자격', '청년도약계좌 신청 절차', '일자리 지원 대상 조건'"""
    docs = retrieve(query, k=4)
    if not docs:
        return "관련 정보를 찾을 수 없습니다."

    chunks: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "알 수 없음")
        page = doc.metadata.get("page", "")
        tag = f"[출처: {source}" + (f", p.{page}]" if page else "]")
        chunks.append(f"{tag}\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)


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


_TOOLS = [rag_search, get_policy_list, get_policy_detail]
_TOOLS_BY_NAME: dict[str, object] = {t.name: t for t in _TOOLS}  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 에이전트
# ---------------------------------------------------------------------------


class PolicyAgent:
    """RAG + MCP Tool 라우팅 에이전트."""

    def __init__(self) -> None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
        )
        self._llm_with_tools = _llm.bind_tools(_TOOLS)

    def ask(self, query: str) -> AgentResponse:
        """쿼리를 받아 Tool을 선택·실행하고 자연어 답변을 반환한다."""
        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        tools_used: list[str] = []
        sources: list[str] = []
        fallback_used: bool = False

        for _ in range(_MAX_ITER):
            response: AIMessage = self._llm_with_tools.invoke(messages)
            logger.debug("[llm-response] tool_calls=%s", response.tool_calls)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                name: str = tc["name"]
                args: dict = tc["args"]
                tools_used.append(name)
                logger.info("[tool] %s %s", name, args)

                tool_fn = _TOOLS_BY_NAME.get(name)
                if tool_fn is None:
                    result: str = f"알 수 없는 tool: {name}"
                    mcp_failed = False
                else:
                    try:
                        result = tool_fn.invoke(args)  # type: ignore[union-attr]
                        mcp_failed = False
                    except Exception as exc:
                        logger.warning("[tool-error] %s: %s", name, exc)
                        if name in _MCP_TOOLS:
                            result = self._rag_fallback(query, args)
                            tools_used.append("rag_search")
                            fallback_used = True
                            mcp_failed = True
                            logger.info(
                                "[mcp-fallback] %s 실패 → RAG fallback 실행", name
                            )
                        else:
                            result = f"Tool 오류: {exc}"
                            mcp_failed = False

                # 출처 추출 (rag_search 직접 호출 또는 fallback 결과)
                if (name == "rag_search" or mcp_failed) and isinstance(result, str):
                    for line in result.splitlines():
                        if line.startswith("[출처:"):
                            sources.append(line.strip("[]").replace("출처: ", ""))

                tool_content = (
                    _FALLBACK_PREFIX + result if mcp_failed else str(result)
                )
                messages.append(
                    ToolMessage(content=tool_content, tool_call_id=tc["id"])
                )

        last = messages[-1]
        answer = getattr(last, "content", "") or ""
        if not answer:
            answer = "확인된 정보 없음"

        return AgentResponse(
            answer=answer,
            sources=sources,
            tools_used=tools_used,
            fallback_used=fallback_used,
        )

    @staticmethod
    def _rag_fallback(query: str, failed_args: dict) -> str:
        """MCP 실패 시 RAG로 대체 검색한다."""
        fallback_query = (
            failed_args.get("keyword")
            or failed_args.get("policy_name")
            or query
        )
        try:
            return rag_search.invoke({"query": fallback_query})  # type: ignore[return-value]
        except Exception as exc:
            logger.error("[rag-fallback-error] %s", exc)
            return "RAG 조회도 실패했습니다. 잠시 후 다시 시도해 주세요."
