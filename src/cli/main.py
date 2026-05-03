import argparse
from pathlib import Path

from rich.console import Console

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


def main() -> None:
    parser = argparse.ArgumentParser(prog="policy-agent", description="청년 정책 에이전트 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="PDF를 Pinecone에 적재")
    ingest_parser.add_argument("--path", required=True, help="PDF 디렉토리 경로")
    ingest_parser.set_defaults(func=cmd_ingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
