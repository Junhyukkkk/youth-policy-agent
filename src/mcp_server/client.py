"""온통청년 OpenAPI httpx 클라이언트.

응답 형식: XML (UTF-8)
인증: openApiVlak 쿼리 파라미터
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from src.config import settings

BASE_URL = "https://www.youthcenter.go.kr/opi"
TIMEOUT = httpx.Timeout(10.0, connect=5.0)


# ---------------------------------------------------------------------------
# Pydantic 모델
# ---------------------------------------------------------------------------

class PolicyListItem(BaseModel):
    """목록 조회 1건."""
    biz_id: str = Field(alias="bizId")
    plcy_nm: str = Field(alias="plcyNm", default="")          # 정책명
    plcy_expl: str = Field(alias="plcyExpl", default="")      # 정책 설명
    plcy_kywrd: str = Field(alias="plcyKywrd", default="")    # 키워드
    biz_ty_nm: str = Field(alias="bizTyNm", default="")       # 사업 유형명
    poly_biz_secd: str = Field(alias="polyBizSecd", default="")  # 정책 분류 코드
    cnsg_nmor: str = Field(alias="cnsgNmor", default="")      # 주관 기관
    age_info: str = Field(alias="ageInfo", default="")        # 연령 조건
    empm_stts_cd: str = Field(alias="empmSttsCd", default="") # 취업 상태 코드
    spor_cn: str = Field(alias="sporCn", default="")          # 지원 내용

    model_config = {"populate_by_name": True}


class PolicyDetail(BaseModel):
    """상세 조회 응답."""
    biz_id: str = Field(alias="bizId")
    plcy_nm: str = Field(alias="plcyNm", default="")
    plcy_expl: str = Field(alias="plcyExpl", default="")
    plcy_kywrd: str = Field(alias="plcyKywrd", default="")
    biz_ty_nm: str = Field(alias="bizTyNm", default="")
    cnsg_nmor: str = Field(alias="cnsgNmor", default="")      # 주관 기관
    spor_cn: str = Field(alias="sporCn", default="")          # 지원 내용
    aplcn_trget: str = Field(alias="aplcnTrget", default="")  # 신청 대상
    rqut_prd_cn: str = Field(alias="rqutPrdCn", default="")   # 신청 기간
    rqut_urla: str = Field(alias="rqutUrla", default="")      # 신청 URL
    age_info: str = Field(alias="ageInfo", default="")
    empm_stts_cd: str = Field(alias="empmSttsCd", default="")
    mrg_info: str = Field(alias="mrgInfo", default="")        # 결혼 조건
    edu_rfn_cd: str = Field(alias="eduRfnCd", default="")     # 학력 코드
    majr_cd: str = Field(alias="majrCd", default="")          # 전공 코드
    spclf_cd: str = Field(alias="splfCd", default="")         # 특화 분야 코드
    empl_stts_cd: str = Field(alias="emplSttsCd", default="") # 고용 상태 코드
    prcptn_limit: str = Field(alias="prcptnLimit", default="") # 소득 조건
    rstd_area: str = Field(alias="rstdArea", default="")       # 거주 지역 조건

    model_config = {"populate_by_name": True}


class PolicyListResponse(BaseModel):
    total_count: int
    page_index: int
    items: list[PolicyListItem]


# ---------------------------------------------------------------------------
# XML 파서 헬퍼
# ---------------------------------------------------------------------------

def _text(el: ET.Element, tag: str) -> str:
    child = el.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _parse_list_xml(xml_text: str) -> PolicyListResponse:
    root = ET.fromstring(xml_text)

    total_count = int(_text(root, "totCount") or "0")
    page_index = int(_text(root, "pageIndex") or "1")

    items: list[PolicyListItem] = []
    for item_el in root.findall(".//youthPolicy"):
        raw = {child.tag: (child.text or "").strip() for child in item_el}
        # bizId 없으면 건너뜀
        if not raw.get("bizId"):
            continue
        items.append(PolicyListItem.model_validate(raw))

    return PolicyListResponse(
        total_count=total_count,
        page_index=page_index,
        items=items,
    )


def _parse_detail_xml(xml_text: str) -> PolicyDetail:
    root = ET.fromstring(xml_text)

    # 상세 응답은 루트 바로 아래 혹은 <youthPolicy> 태그 안에 존재
    policy_el = root.find(".//youthPolicy") or root
    raw = {child.tag: (child.text or "").strip() for child in policy_el}
    return PolicyDetail.model_validate(raw)


# ---------------------------------------------------------------------------
# API 클라이언트
# ---------------------------------------------------------------------------

class YouthPolicyClient:
    """온통청년 OpenAPI 비동기 클라이언트."""

    def __init__(self) -> None:
        api_key = settings.youth_policy_api_key
        if not api_key:
            raise ValueError("YOUTH_POLICY_API_KEY가 설정되지 않았습니다.")
        self._api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "YouthPolicyClient":
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=TIMEOUT,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()

    def _base_params(self) -> dict[str, str]:
        return {"openApiVlak": self._api_key}

    async def get_policy_list(
        self,
        *,
        region: str = "",
        category: str = "",
        keyword: str = "",
        page: int = 1,
        display: int = 10,
    ) -> PolicyListResponse:
        """청년정책 목록 조회.

        Args:
            region: 거주 지역 (예: 서울, 경기). 비어 있으면 전국.
            category: 정책 분류 코드 (srchPolyBizSecd). 예: 023010.
            keyword: 검색 키워드.
            page: 페이지 번호 (1-based).
            display: 페이지당 결과 수.
        """
        assert self._client is not None, "async context manager 안에서 호출해야 합니다."

        params: dict[str, str] = {
            **self._base_params(),
            "pageIndex": str(page),
            "display": str(display),
        }
        if region:
            params["lclScCode"] = region
        if category:
            params["bizTycdSel"] = category
        if keyword:
            params["srchPolyBizSecd"] = keyword

        try:
            resp = await self._client.get("/youthPlcyList.do", params=params)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise RuntimeError(f"온통청년 API 응답 timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"온통청년 API HTTP 오류 {e.response.status_code}: {e.response.text[:200]}"
            ) from e

        return _parse_list_xml(resp.text)

    async def get_policy_detail(self, policy_id: str) -> PolicyDetail:
        """청년정책 상세 조회.

        Args:
            policy_id: 정책 고유 ID (bizId).
        """
        assert self._client is not None, "async context manager 안에서 호출해야 합니다."

        params: dict[str, str] = {
            **self._base_params(),
            "bizId": policy_id,
        }

        try:
            resp = await self._client.get("/youthPlcyDtl.do", params=params)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise RuntimeError(f"온통청년 API 응답 timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"온통청년 API HTTP 오류 {e.response.status_code}: {e.response.text[:200]}"
            ) from e

        return _parse_detail_xml(resp.text)
