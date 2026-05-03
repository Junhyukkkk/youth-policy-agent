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
            "예: '지금 신청 가능한 주거 지원 있어?', '서울 청년 일자리 공고 알려줘'"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "정책 키워드 (plcyKywdNm). 예: '주거지원', '바우처'.",
                },
                "region": {
                    "type": "string",
                    "description": "법정시군구코드 5자리 (zipCd). 예: '11000'=서울, '26110'=부산.",
                },
                "category": {
                    "type": "string",
                    "description": "정책 대분류명 (lclsfNm). 예: '일자리', '주거', '교육', '금융･복지･문화'.",
                },
                "policy_name": {
                    "type": "string",
                    "description": "정책명 검색어. 예: '청년도약계좌'.",
                },
                "page": {
                    "type": "integer",
                    "description": "페이지 번호 (기본값 1).",
                    "default": 1,
                },
                "page_size": {
                    "type": "integer",
                    "description": "페이지당 결과 수 (기본값 10).",
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
            "신청 방법, 지원 내용, 자격 조건, 신청 기간 등 구체적인 정보가 필요할 때 사용하세요. "
            "반드시 get_policy_list로 얻은 policy_no(plcyNo)를 전달하세요."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "policy_no": {
                    "type": "string",
                    "description": "정책 번호 (plcyNo). 예: '20260430005400113009'.",
                },
            },
            "required": ["policy_no"],
        },
    ),
]


def list_tools() -> list[Tool]:
    return TOOLS


async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    async with YouthPolicyClient() as client:
        if name == "get_policy_list":
            result = await client.get_policy_list(
                keyword=arguments.get("keyword", ""),
                region=arguments.get("region", ""),
                category=arguments.get("category", ""),
                policy_name=arguments.get("policy_name", ""),
                page=int(arguments.get("page", 1)),
                page_size=int(arguments.get("page_size", 10)),
            )
            lines = [
                f"총 {result.pagging.tot_count}건 "
                f"(페이지 {result.pagging.page_num}/{-(-result.pagging.tot_count // result.pagging.page_size)})\n"
            ]
            for item in result.items:
                age = f"{item.sprt_trgt_min_age}~{item.sprt_trgt_max_age}세" if item.sprt_trgt_min_age else ""
                lines.append(
                    f"[{item.plcy_no}] {item.plcy_nm}\n"
                    f"  분류: {item.lclsf_nm} > {item.mclsf_nm} | 연령: {age}\n"
                    f"  기관: {item.sprvsn_inst_cd_nm} | 신청기간: {item.aply_ymd}\n"
                    f"  설명: {item.plcy_expln_cn[:80]}{'...' if len(item.plcy_expln_cn) > 80 else ''}"
                )
            return [TextContent(type="text", text="\n".join(lines))]

        if name == "get_policy_detail":
            policy_no = arguments.get("policy_no", "")
            if not policy_no:
                return [TextContent(type="text", text="오류: policy_no가 필요합니다.")]
            detail = await client.get_policy_detail(policy_no)
            text = (
                f"■ {detail.plcy_nm}\n"
                f"분류: {detail.lclsf_nm} > {detail.mclsf_nm}\n"
                f"주관기관: {detail.sprvsn_inst_cd_nm}\n"
                f"연령: {detail.sprt_trgt_min_age}~{detail.sprt_trgt_max_age}세\n"
                f"신청기간: {detail.aply_ymd}\n\n"
                f"[지원내용]\n{detail.plcy_sprt_cn}\n\n"
                f"[신청방법]\n{detail.plcy_aply_mthd_cn}\n\n"
                f"[제출서류]\n{detail.sbmsn_dcmnt_cn}\n\n"
                f"[소득조건]\n{detail.earn_etc_cn}\n"
                f"신청URL: {detail.aply_url_addr}\n"
                f"참고URL: {detail.ref_url_addr1}"
            )
            return [TextContent(type="text", text=text)]

        return [TextContent(type="text", text=f"알 수 없는 tool: {name}")]
