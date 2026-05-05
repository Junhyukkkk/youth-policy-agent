# ============================================================
# cli/main.py — 터미널 진입점 (사용자가 직접 실행하는 곳)
#
# 역할: "policy-agent" 명령어로 실행되는 CLI 인터페이스.
#       argparse로 서브커맨드를 파싱하고, 각 명령에 맞는 함수를 호출한다.
#
# 사용 가능한 명령어:
#   policy-agent chat             → 대화형 채팅 모드 (계속 질문 가능)
#   policy-agent ask "질문"       → 단발성 질문
#   policy-agent ingest --path .. → PDF를 Pinecone에 저장
#   policy-agent test-mcp --tool  → MCP 도구 직접 테스트 (LLM 없이)
#
# Rich 라이브러리:
#   컬러, 마크다운 렌더링, 진행바, 패널 등 예쁜 터미널 출력을 위해 사용.
#   일반 print()보다 훨씬 읽기 쉬운 결과를 보여준다.
# ============================================================

import argparse
import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown    # LLM 답변을 마크다운으로 렌더링
from rich.markup import escape        # Rich 특수문자 이스케이프
from rich.panel import Panel          # 테두리 있는 패널 UI

from src.rag.ingest import ingest_directory

# highlight=False: 숫자나 특수문자를 자동으로 색칠하는 기능 끔 (오히려 헷갈릴 수 있음)
console = Console(highlight=False)

# MCP 도구 이름 집합 — 배지(badge) 표시 시 MCP/RAG 구분에 사용
_MCP_TOOL_NAMES = {"get_policy_list", "get_policy_detail"}

# ── 컬러 배지 문자열 정의 (Rich 마크업 형식) ─────────────────────────
_BADGE_RAG = f"[bold green]{escape('[RAG]')}[/bold green]"           # 초록 [RAG]
_BADGE_MCP = f"[bold cyan]{escape('[MCP]')}[/bold cyan]"             # 청록 [MCP]
_BADGE_FALLBACK = f"[bold yellow]{escape('[RAG-FALLBACK]')}[/bold yellow]"  # 노랑 [RAG-FALLBACK]


def _build_badge_line(tools_used: list[str], fallback_used: bool) -> str:
    """사용된 도구에 따라 배지 줄 생성.

    배지는 답변 위에 표시되어 어떤 데이터 소스를 사용했는지 한눈에 보여준다.
    예: [MCP]   [RAG]  또는  [RAG-FALLBACK]
    """
    parts: list[str] = []
    has_mcp = any(t in _MCP_TOOL_NAMES for t in tools_used)
    has_rag = "rag_search" in tools_used

    if fallback_used:
        # MCP 실패 → RAG로 대체된 경우 노란 FALLBACK 배지
        parts.append(_BADGE_FALLBACK)
    else:
        if has_mcp:
            parts.append(_BADGE_MCP)
        if has_rag:
            parts.append(_BADGE_RAG)

    return "  ".join(parts)


def _print_sources_panel(
    sources: list[str], tools_used: list[str], fallback_used: bool
) -> None:
    """출처 정보를 패널로 출력.

    RAG 출처는 파일명·페이지, MCP 출처는 "온통청년 API (실시간 조회)" 로 표시.
    """
    lines: list[str] = []
    for src in sources:
        lines.append(f"• {src}")
    # MCP를 사용했고 fallback이 아닌 경우 → 실시간 API 출처 추가
    if any(t in _MCP_TOOL_NAMES for t in tools_used) and not fallback_used:
        lines.append("• 온통청년 API (실시간 조회)")

    content = "\n".join(lines) if lines else "출처 정보 없음"
    # Panel: 테두리가 있는 박스 형태로 출력, dim(흐린) 스타일
    console.print(Panel(content, title="[dim]출처[/dim]", border_style="dim"))


def _display_response(resp) -> None:  # type: ignore[no-untyped-def]
    """AgentResponse를 Rich로 출력한다.

    출력 순서:
      1. 사용 도구 배지 ([RAG] / [MCP] / [RAG-FALLBACK])
      2. 구분선
      3. LLM 답변 (마크다운 렌더링)
      4. 출처 패널 (있는 경우)
      5. 환각 경고 패널 (출처 미확인 숫자가 있는 경우)
    """
    badge_line = _build_badge_line(resp.tools_used, resp.fallback_used)
    if badge_line:
        console.print(badge_line)

    console.rule()  # 수평 구분선 출력
    # Markdown(): LLM이 생성한 마크다운 텍스트를 터미널에서 예쁘게 렌더링
    console.print(Markdown(resp.answer))

    has_sources = resp.sources or (
        any(t in _MCP_TOOL_NAMES for t in resp.tools_used) and not resp.fallback_used
    )
    if has_sources:
        _print_sources_panel(resp.sources, resp.tools_used, resp.fallback_used)

    # 환각 감지 경고: 빨간 경계선 패널로 강조 표시
    if resp.hallucination_warning:
        console.print(
            Panel(
                "답변에 출처에서 확인되지 않은 숫자가 포함되어 있을 수 있습니다.\n"
                "중요한 수치(금액·날짜·자격 연령 등)는 원문을 직접 확인하세요.",
                title="[bold red]⚠ 수치 검증 필요[/bold red]",
                border_style="red",
            )
        )


# ============================================================
# 서브커맨드 핸들러 함수들
# ============================================================

def cmd_ingest(args: argparse.Namespace) -> None:
    """policy-agent ingest --path <디렉토리> 실행 시 호출.

    지정된 디렉토리의 PDF를 읽어서 Pinecone에 업로드한다.
    처음 한 번만 실행하면 되고, PDF를 추가하거나 수정했을 때 다시 실행한다.
    """
    path = Path(args.path)
    if not path.exists():
        console.print(f"[red]경로를 찾을 수 없습니다: {path}[/red]")
        raise SystemExit(1)

    console.print(f"[bold]Pinecone 적재 시작:[/bold] {path.resolve()}")
    docs, chunks = ingest_directory(path)
    console.print(f"\n[bold green]{docs}개 문서, {chunks}개 청크 적재 완료[/bold green]")


async def _run_test_mcp(args: argparse.Namespace) -> None:
    """MCP 도구를 LLM 없이 직접 실행하는 비동기 내부 함수.

    개발 중에 API 연동이 잘 되는지 확인할 때 사용한다.
    예: policy-agent test-mcp --tool get_policy_list --keyword 주거
    """
    from src.mcp_server.tools import call_tool

    tool_name = args.tool
    arguments: dict = {}

    if tool_name == "get_policy_list":
        # 각 옵션이 입력된 경우에만 arguments에 추가 (빈 문자열 제외)
        if args.region:
            arguments["region"] = args.region
        if args.category:
            arguments["category"] = args.category
        if args.keyword:
            arguments["keyword"] = args.keyword
        arguments["page"] = args.page
        arguments["page_size"] = args.display
    elif tool_name == "get_policy_detail":
        if not args.policy_id:
            console.print("[red]--policy-id 가 필요합니다.[/red]")
            raise SystemExit(1)
        arguments["policy_no"] = args.policy_id
    else:
        console.print(f"[red]알 수 없는 tool: {tool_name}[/red]")
        raise SystemExit(1)

    # 실행할 도구와 파라미터를 먼저 출력 (투명성)
    console.print(f"[bold cyan]Tool:[/bold cyan] {tool_name}")
    console.print(f"[bold cyan]Arguments:[/bold cyan] {json.dumps(arguments, ensure_ascii=False)}")
    console.rule()

    try:
        # 실제 API 호출 — LLM 없이 도구만 직접 실행
        results = await call_tool(tool_name, arguments)
        for content in results:
            print(content.text)
    except RuntimeError as e:
        console.print(f"[red]오류:[/red] {e}")
        raise SystemExit(1)


def cmd_test_mcp(args: argparse.Namespace) -> None:
    """비동기 함수를 동기 컨텍스트에서 실행하는 래퍼."""
    asyncio.run(_run_test_mcp(args))


def cmd_ask(args: argparse.Namespace) -> None:
    """policy-agent ask "질문" 실행 시 호출.

    단 한 번 질문하고 답변을 받는 일회성 모드.
    채팅 모드와 달리 대화 히스토리가 유지되지 않는다.
    """
    from src.agent.orchestrator import PolicyAgent

    query: str = args.query
    console.print(f"\n[bold]질문:[/bold] {query}\n")

    # status(): 답변 생성 중에 스피너 애니메이션 표시
    with console.status("[bold cyan]답변 생성 중...[/bold cyan]", spinner="dots"):
        agent = PolicyAgent()
        resp = agent.ask(query)

    _display_response(resp)


def cmd_chat(_args: argparse.Namespace | None = None) -> None:
    """policy-agent chat (또는 인수 없이 실행) 시 호출.

    대화형 인터랙티브 모드. exit/quit 또는 Ctrl+C로 종료.
    매 질문마다 새로운 ask()를 호출하므로 대화 히스토리는 누적되지 않는다.
    (각 질문이 독립적으로 처리됨)
    """
    from src.agent.orchestrator import PolicyAgent

    # 시작 화면 안내 패널 출력
    console.print(
        Panel(
            "[bold]청년 정책 에이전트[/bold]\n"
            "[dim]자격조건·신청방법은 RAG, 실시간 공고는 MCP로 자동 라우팅됩니다.[/dim]\n"
            "[dim]종료: exit / quit / Ctrl+C[/dim]",
            border_style="cyan",
        )
    )

    # 에이전트는 한 번만 초기화 (LLM 연결은 재사용)
    agent = PolicyAgent()

    while True:
        try:
            # 사용자 입력 대기 (Ctrl+C or Ctrl+D로 탈출 가능)
            query = console.input("\n[bold cyan]질문>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]종료합니다.[/dim]")
            break

        if not query:
            continue  # 빈 입력은 무시하고 다시 대기
        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]종료합니다.[/dim]")
            break

        try:
            with console.status("[bold cyan]답변 생성 중...[/bold cyan]", spinner="dots"):
                resp = agent.ask(query)
            _display_response(resp)
        except Exception as exc:
            console.print(f"[red]오류:[/red] {exc}")


def main() -> None:
    """CLI 진입점 — pyproject.toml의 scripts에 등록되어 있음.

    "poetry run policy-agent" 또는 "policy-agent"로 실행.
    """
    # argparse: 터미널 인수 파싱 라이브러리
    parser = argparse.ArgumentParser(prog="policy-agent", description="청년 정책 에이전트 CLI")
    # subparsers: "policy-agent chat", "policy-agent ask" 같은 서브커맨드 지원
    subparsers = parser.add_subparsers(dest="command", required=False)

    # ── chat 서브커맨드 ────────────────────────────────────────────
    subparsers.add_parser("chat", help="대화형 채팅 모드 (기본값)").set_defaults(func=cmd_chat)

    # ── ingest 서브커맨드 ──────────────────────────────────────────
    ingest_parser = subparsers.add_parser("ingest", help="PDF를 Pinecone에 적재")
    ingest_parser.add_argument("--path", required=True, help="PDF 디렉토리 경로")
    ingest_parser.set_defaults(func=cmd_ingest)

    # ── ask 서브커맨드 ─────────────────────────────────────────────
    ask_parser = subparsers.add_parser("ask", help="단발성 질문")
    ask_parser.add_argument("query", help="질문 내용 (예: '서울 청년 월세 지원 자격')")
    ask_parser.set_defaults(func=cmd_ask)

    # ── test-mcp 서브커맨드 ────────────────────────────────────────
    mcp_parser = subparsers.add_parser("test-mcp", help="MCP Tool 단독 테스트 (LLM 없이)")
    mcp_parser.add_argument("--tool", required=True, choices=["get_policy_list", "get_policy_detail"])
    mcp_parser.add_argument("--region", default="", help="지역 코드 (예: 003002001)")
    mcp_parser.add_argument("--category", default="", help="사업 유형 코드 (예: 023010)")
    mcp_parser.add_argument("--keyword", default="", help="검색 키워드")
    mcp_parser.add_argument("--page", type=int, default=1)
    mcp_parser.add_argument("--display", type=int, default=5)
    mcp_parser.add_argument("--policy-id", default="", help="정책 ID (get_policy_detail용)")
    mcp_parser.set_defaults(func=cmd_test_mcp)

    args = parser.parse_args()

    # 서브커맨드 없이 그냥 "policy-agent"만 실행하면 채팅 모드로 진입
    func = getattr(args, "func", cmd_chat)
    func(args)


if __name__ == "__main__":
    main()
