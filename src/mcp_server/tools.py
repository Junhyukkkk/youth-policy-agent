"""MCP Tool 2종: get_policy_list, get_policy_detail."""
from __future__ import annotations

from mcp.types import Tool, TextContent

from src.mcp_server.client import YouthPolicyClient

TOOLS = [
    Tool(
        name="get_policy_list",
        description=(
            "실시간 청년 정책 공고 목록을 조회합니다. "
            "현재 모집 중인 공고, 마감일, 지역별·분야별 정책 현황 등 "
            "최신 정보가 필요할 때 사용하세요. "
            "예: '지금 신청 가능한 주거 지원 있어?', '서울 청년 취업 공고 알려줘'"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "거주 지역 코드 (예: 003002001=서울, 003002002=부산). 전국이면 생략.",
                },
                "category": {
                    "type": "string",
                    "description": "사업 유형 코드 (bizTycdSel). 예: 023010=일자리, 023020=주거.",
                },
                "keyword": {
                    "type": "string",
                    "description": "자유 검색 키워드. 예: '청년도약계좌', '전세대출'.",
                },
                "page": {
                    "type": "integer",
                    "description": "페이지 번호 (기본값 1).",
                    "default": 1,
                },
                "display": {
                    "type": "integer",
                    "description": "페이지당 결과 수 (기본값 10, 최대 100).",
                    "default": 10,
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="get_policy_detail",
        description=(
            "청년 정책 상세 정보를 조회합니다. "
            "신청 방법, 지원 내용, 자격 조건, 마감일 등 구체적인 정보가 필요할 때 사용하세요. "
            "반드시 get_policy_list로 얻은 policy_id(bizId)를 전달하세요."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "policy_id": {
                    "type": "string",
                    "description": "정책 고유 ID (bizId). 예: R2023081800261.",
                },
            },
            "required": ["policy_id"],
        },
    ),
]


def list_tools() -> list[Tool]:
    return TOOLS


async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    async with YouthPolicyClient() as client:
        if name == "get_policy_list":
            result = await client.get_policy_list(
                region=arguments.get("region", ""),
                category=arguments.get("category", ""),
                keyword=arguments.get("keyword", ""),
                page=int(arguments.get("page", 1)),
                display=int(arguments.get("display", 10)),
            )
            lines = [f"총 {result.total_count}건 (페이지 {result.page_index})\n"]
            for item in result.items:
                lines.append(
                    f"[{item.biz_id}] {item.plcy_nm}\n"
                    f"  기관: {item.cnsg_nmor} | 연령: {item.age_info}\n"
                    f"  설명: {item.plcy_expl[:80]}{'...' if len(item.plcy_expl) > 80 else ''}"
                )
            return [TextContent(type="text", text="\n".join(lines))]

        if name == "get_policy_detail":
            policy_id = arguments.get("policy_id", "")
            if not policy_id:
                return [TextContent(type="text", text="오류: policy_id가 필요합니다.")]
            detail = await client.get_policy_detail(policy_id)
            text = (
                f"정책명: {detail.plcy_nm}\n"
                f"주관기관: {detail.cnsg_nmor}\n"
                f"지원내용: {detail.spor_cn}\n"
                f"신청대상: {detail.aplcn_trget}\n"
                f"신청기간: {detail.rqut_prd_cn}\n"
                f"신청URL: {detail.rqut_urla}\n"
                f"연령조건: {detail.age_info}\n"
                f"설명: {detail.plcy_expl}"
            )
            return [TextContent(type="text", text=text)]

        return [TextContent(type="text", text=f"알 수 없는 tool: {name}")]
