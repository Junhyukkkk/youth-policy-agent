# ============================================================
# mcp_server/tools.py — MCP 도구 2종 정의 + 실행 로직
#
# 역할: LLM(Gemini)이 "어떤 도구를 호출할 수 있는지"를 알 수 있도록
#       도구의 이름·설명·입력 스키마를 정의하고,
#       실제로 도구가 호출됐을 때 client.py를 통해 API를 호출한다.
#
# MCP(Model Context Protocol)란?
#   - Anthropic이 만든 표준 프로토콜로, LLM이 외부 도구를 호출하는 방식을 표준화한 것.
#   - 이 파일에서 Tool 객체로 선언한 스키마를 LLM이 읽고,
#     필요할 때 call_tool()을 통해 실제 API를 호출한다.
#
# 도구 목록:
#   1. get_policy_list  — 현재 모집 중인 정책 목록 조회 (실시간)
#   2. get_policy_detail — 특정 정책의 상세 정보 조회 (실시간)
# ============================================================
"""MCP Tool 2종: get_policy_list, get_policy_detail."""
from __future__ import annotations

from mcp.types import Tool, TextContent

from src.mcp_server.client import YouthPolicyClient

# ── 도구 스키마 정의 ──────────────────────────────────────────────────
# Tool 객체는 LLM에게 "이런 도구가 있고, 이런 입력을 받는다"고 알려주는 명세서.
# 설명(description)이 좋을수록 LLM이 언제 이 도구를 써야 할지 잘 판단한다.

TOOLS = [
    Tool(
        name="get_policy_list",
        description=(
            "실시간 청년 정책 공고 목록을 조회합니다. "
            "현재 모집 중인 공고, 마감일, 지역별·분야별 정책 현황 등 "
            "최신 정보가 필요할 때 사용하세요. "
            "예: '지금 신청 가능한 주거 지원 있어?', '서울 청년 일자리 공고 알려줘'"
        ),
        # inputSchema: JSON Schema 형식으로 이 도구가 받을 수 있는 파라미터를 정의
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
            "required": [],  # 모든 파라미터가 선택사항 (빈 조건으로 전체 목록 조회 가능)
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
            "required": ["policy_no"],  # 정책 번호는 필수 (없으면 뭘 조회할지 모름)
        },
    ),
]


def list_tools() -> list[Tool]:
    # MCP 서버가 "어떤 도구가 있나요?" 라는 요청에 응답할 때 호출됨
    return TOOLS


async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # MCP 서버가 실제로 도구를 실행할 때 호출됨
    # 반환값: TextContent 리스트 (텍스트 형태의 결과)

    # 비동기 컨텍스트 매니저로 HTTP 클라이언트 열기
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
            # API 결과를 사람이 읽기 쉬운 텍스트로 포맷팅
            lines = [
                f"총 {result.pagging.tot_count}건 "
                # -(-a // b) = 올림 나눗셈 트릭: 전체 페이지 수 계산
                f"(페이지 {result.pagging.page_num}/{-(-result.pagging.tot_count // result.pagging.page_size)})\n"
            ]
            for item in result.items:
                age = f"{item.sprt_trgt_min_age}~{item.sprt_trgt_max_age}세" if item.sprt_trgt_min_age else ""
                lines.append(
                    f"[{item.plcy_no}] {item.plcy_nm}\n"
                    f"  분류: {item.lclsf_nm} > {item.mclsf_nm} | 연령: {age}\n"
                    f"  기관: {item.sprvsn_inst_cd_nm} | 신청기간: {item.aply_ymd}\n"
                    # 설명이 너무 길면 80자로 자르고 "..." 추가
                    f"  설명: {item.plcy_expln_cn[:80]}{'...' if len(item.plcy_expln_cn) > 80 else ''}"
                )
            return [TextContent(type="text", text="\n".join(lines))]

        if name == "get_policy_detail":
            policy_no = arguments.get("policy_no", "")
            if not policy_no:
                return [TextContent(type="text", text="오류: policy_no가 필요합니다.")]
            detail = await client.get_policy_detail(policy_no)
            # 상세 정보를 섹션별로 포맷팅해서 LLM이 답변 생성하기 쉽게 만듦
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

        # 알 수 없는 도구 이름이 들어온 경우 에러 메시지 반환
        return [TextContent(type="text", text=f"알 수 없는 tool: {name}")]
