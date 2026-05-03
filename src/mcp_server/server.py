"""MCP stdio 서버 진입점."""
from mcp.server import Server
from mcp.server.stdio import stdio_server

from src.mcp_server.tools import list_tools, call_tool

app = Server("youth-policy-mcp", version="0.1.0")


@app.list_tools()  # type: ignore[misc]
async def handle_list_tools():  # type: ignore[return]
    return list_tools()


@app.call_tool()  # type: ignore[misc]
async def handle_call_tool(name: str, arguments: dict):  # type: ignore[return]
    return await call_tool(name, arguments)


async def run() -> None:
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())
