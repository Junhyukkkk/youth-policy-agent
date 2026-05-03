import argparse
import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.pretty import pprint

from src.rag.ingest import ingest_directory

console = Console()


def cmd_ingest(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.exists():
        console.print(f"[red]경로를 찾을 수 없습니다: {path}[/red]")
        raise SystemExit(1)

    console.print(f"[bold]Pinecone 적재 시작:[/bold] {path.resolve()}")
    docs, chunks = ingest_directory(path)
    console.print(f"\n[bold green]{docs}개 문서, {chunks}개 청크 적재 완료[/bold green]")


async def _run_test_mcp(args: argparse.Namespace) -> None:
    from src.mcp_server.tools import call_tool

    tool_name = args.tool
    arguments: dict = {}

    if tool_name == "get_policy_list":
        if args.region:
            arguments["region"] = args.region
        if args.category:
            arguments["category"] = args.category
        if args.keyword:
            arguments["keyword"] = args.keyword
        arguments["page"] = args.page
        arguments["display"] = args.display
    elif tool_name == "get_policy_detail":
        if not args.policy_id:
            console.print("[red]--policy-id 가 필요합니다.[/red]")
            raise SystemExit(1)
        arguments["policy_id"] = args.policy_id
    else:
        console.print(f"[red]알 수 없는 tool: {tool_name}[/red]")
        raise SystemExit(1)

    console.print(f"[bold cyan]Tool:[/bold cyan] {tool_name}")
    console.print(f"[bold cyan]Arguments:[/bold cyan] {json.dumps(arguments, ensure_ascii=False)}")
    console.rule()

    try:
        results = await call_tool(tool_name, arguments)
        for content in results:
            console.print(content.text)
    except RuntimeError as e:
        console.print(f"[red]오류:[/red] {e}")
        raise SystemExit(1)


def cmd_test_mcp(args: argparse.Namespace) -> None:
    asyncio.run(_run_test_mcp(args))


def main() -> None:
    parser = argparse.ArgumentParser(prog="policy-agent", description="청년 정책 에이전트 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest 서브커맨드
    ingest_parser = subparsers.add_parser("ingest", help="PDF를 Pinecone에 적재")
    ingest_parser.add_argument("--path", required=True, help="PDF 디렉토리 경로")
    ingest_parser.set_defaults(func=cmd_ingest)

    # test-mcp 서브커맨드
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
    args.func(args)


if __name__ == "__main__":
    main()
