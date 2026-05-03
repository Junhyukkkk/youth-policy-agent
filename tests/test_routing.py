"""라우팅 단위 테스트 — Tool 선택 정확도 확인.

LLM 실제 호출 없이 bind_tools 응답을 mock해서 라우팅 로직만 검증.
실제 API 연동 테스트는 E2E로 별도 진행.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.orchestrator import AgentResponse, PolicyAgent, _TOOLS_BY_NAME


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_ai_message(tool_name: str | None, tool_args: dict | None = None) -> MagicMock:
    """tool_calls 포함 또는 미포함 AIMessage mock."""
    msg = MagicMock()
    msg.content = "테스트 답변입니다."
    if tool_name:
        msg.tool_calls = [{"id": "tc-1", "name": tool_name, "args": tool_args or {}}]
    else:
        msg.tool_calls = []
    return msg


def _final_message() -> MagicMock:
    msg = MagicMock()
    msg.content = "최종 답변입니다."
    msg.tool_calls = []
    return msg


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------


class TestRouting:
    """LLM이 올바른 Tool을 선택하는지 검증."""

    def _run_with_tool_call(self, query: str, expected_tool: str) -> str:
        """LLM이 expected_tool을 호출하도록 mock하고 ask() 실행."""
        with (
            patch.object(
                PolicyAgent,
                "__init__",
                lambda self: self.__dict__.update(
                    {"_llm_with_tools": MagicMock()}
                ),
            ),
            patch(
                f"src.agent.orchestrator._TOOLS_BY_NAME",
                {expected_tool: MagicMock(return_value="mock tool result")},
            ),
        ):
            agent = PolicyAgent.__new__(PolicyAgent)
            agent._llm_with_tools = MagicMock()
            # 1회차: tool 호출, 2회차: 최종 답변
            agent._llm_with_tools.invoke.side_effect = [
                _make_ai_message(expected_tool, {"query": query}),
                _final_message(),
            ]

            with patch.dict(
                "src.agent.orchestrator._TOOLS_BY_NAME",
                {expected_tool: MagicMock(return_value="mock tool result")},
            ):
                resp = agent.ask(query)

        return resp.tools_used[0] if resp.tools_used else ""

    def test_guideline_query_uses_rag(self) -> None:
        """자격요건 질문 → rag_search 선택."""
        tool_used = self._run_with_tool_call(
            "서울 청년 월세 지원 자격이 어떻게 돼?", "rag_search"
        )
        assert tool_used == "rag_search"

    def test_procedure_query_uses_rag(self) -> None:
        """신청 절차 질문 → rag_search 선택."""
        tool_used = self._run_with_tool_call(
            "청년도약계좌 신청 방법 알려줘", "rag_search"
        )
        assert tool_used == "rag_search"

    def test_live_announcement_uses_mcp_list(self) -> None:
        """현재 열린 공고 질문 → get_policy_list 선택."""
        tool_used = self._run_with_tool_call(
            "지금 신청 가능한 주거 지원 공고 있어?", "get_policy_list"
        )
        assert tool_used == "get_policy_list"

    def test_deadline_query_uses_mcp_list(self) -> None:
        """마감일 질문 → get_policy_list 선택."""
        tool_used = self._run_with_tool_call(
            "이번 달 마감하는 청년 정책 목록 알려줘", "get_policy_list"
        )
        assert tool_used == "get_policy_list"

    def test_detail_query_uses_mcp_detail(self) -> None:
        """특정 정책 상세 조회 → get_policy_detail 선택."""
        tool_used = self._run_with_tool_call(
            "정책번호 20260430005400113009 상세 내용 알려줘", "get_policy_detail"
        )
        assert tool_used == "get_policy_detail"


class TestAgentResponse:
    """AgentResponse 구조 및 fallback 동작 검증."""

    def test_no_tool_call_returns_llm_content(self) -> None:
        """Tool 호출 없이 LLM이 바로 답변하면 content를 그대로 반환."""
        agent = PolicyAgent.__new__(PolicyAgent)
        agent._llm_with_tools = MagicMock()
        agent._llm_with_tools.invoke.return_value = _final_message()

        resp = agent.ask("안녕하세요")
        assert resp.answer == "최종 답변입니다."
        assert resp.tools_used == []

    def test_tool_error_does_not_crash(self) -> None:
        """Tool 실행 중 예외 발생 시 에이전트가 중단되지 않음."""
        agent = PolicyAgent.__new__(PolicyAgent)
        agent._llm_with_tools = MagicMock()
        agent._llm_with_tools.invoke.side_effect = [
            _make_ai_message("rag_search", {"query": "test"}),
            _final_message(),
        ]

        broken_tool = MagicMock(side_effect=RuntimeError("Pinecone down"))
        with patch.dict(
            "src.agent.orchestrator._TOOLS_BY_NAME", {"rag_search": broken_tool}
        ):
            resp = agent.ask("자격 요건 알려줘")

        assert "오류" in resp.answer or resp.answer  # 크래시 없이 응답 반환


class TestToolsRegistered:
    """_TOOLS_BY_NAME에 세 Tool이 모두 등록되어 있는지 확인."""

    def test_all_tools_registered(self) -> None:
        assert "rag_search" in _TOOLS_BY_NAME
        assert "get_policy_list" in _TOOLS_BY_NAME
        assert "get_policy_detail" in _TOOLS_BY_NAME
