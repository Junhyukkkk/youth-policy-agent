# ============================================================
# mcp_server/server.py — MCP 서버 진입점 (표준 입출력 기반 통신)
#
# 역할: MCP 프로토콜 서버를 stdio(표준 입출력)로 실행한다.
#       LLM 클라이언트(orchestrator.py)가 이 서버에 연결해서
#       "어떤 도구가 있나요?" / "이 도구를 실행해 주세요" 메시지를 보낸다.
#
# stdio 서버란?
#   - HTTP 서버가 아니라 표준 입력(stdin)/출력(stdout)으로 메시지를 주고받는다.
#   - 같은 프로세스 안에서 서브프로세스로 실행되거나,
#     파이프를 통해 다른 프로그램과 통신할 때 사용된다.
#   - 이 프로젝트에서는 orchestrator.py가 직접 tools.py를 import해서
#     사용하므로, 이 서버 파일 자체는 독립 실행보다 MCP 규격 준수를 위한 구조.
#
# 파일 구조:
#   server.py (진입점 + 라우팅)
#     ↕ MCP 프로토콜
#   tools.py (실제 도구 로직)
#     ↕ HTTP
#   client.py (온통청년 API)
# ============================================================
"""MCP stdio 서버 진입점."""
from mcp.server import Server
from mcp.server.stdio import stdio_server

from src.mcp_server.tools import list_tools, call_tool

# MCP 서버 인스턴스 생성 (이름과 버전 지정)
app = Server("youth-policy-mcp", version="0.1.0")


@app.list_tools()  # type: ignore[misc]
async def handle_list_tools():  # type: ignore[return]
    # 클라이언트가 "어떤 도구가 있나요?" 라고 물으면 이 핸들러가 호출됨
    # tools.py의 TOOLS 리스트를 그대로 반환
    return list_tools()


@app.call_tool()  # type: ignore[misc]
async def handle_call_tool(name: str, arguments: dict):  # type: ignore[return]
    # 클라이언트가 특정 도구를 실행해달라고 요청하면 이 핸들러가 호출됨
    # name: 실행할 도구 이름 ("get_policy_list" 또는 "get_policy_detail")
    # arguments: 도구에 전달할 파라미터 딕셔너리
    return await call_tool(name, arguments)


async def run() -> None:
    # stdio_server(): 표준 입출력을 통해 MCP 메시지를 읽고 쓰는 스트림 반환
    # app.run(): 서버 메인 루프 시작 — 클라이언트 메시지를 받아서 처리하고 응답
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())
