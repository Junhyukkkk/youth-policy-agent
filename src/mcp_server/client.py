"""온통청년 OpenAPI httpx 클라이언트.

엔드포인트: https://www.youthcenter.go.kr/go/ythip/getPlcy
응답 형식: JSON
인증: apiKeyNm 쿼리 파라미터
"""
from __future__ import annotations

from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, model_validator

from src.config import settings

BASE_URL = "https://www.youthcenter.go.kr"
TIMEOUT = httpx.Timeout(10.0, connect=5.0)


# ---------------------------------------------------------------------------
# Pydantic 모델
# ---------------------------------------------------------------------------

class _NullSafeModel(BaseModel):
    """API가 null 반환 시 빈 문자열로 변환."""

    @model_validator(mode="before")
    @classmethod
    def coerce_null(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: ("" if v is None else v) for k, v in data.items()}
        return data


class PolicyListItem(_NullSafeModel):
    plcy_no: str = Field(alias="plcyNo", default="")            # 정책번호
    plcy_nm: str = Field(alias="plcyNm", default="")            # 정책명
    plcy_kywd_nm: str = Field(alias="plcyKywdNm", default="")   # 키워드
    plcy_expln_cn: str = Field(alias="plcyExplnCn", default="") # 정책설명
    lclsf_nm: str = Field(alias="lclsfNm", default="")          # 정책대분류명
    mclsf_nm: str = Field(alias="mclsfNm", default="")          # 정책중분류명
    plcy_sprt_cn: str = Field(alias="plcySprtCn", default="")   # 지원내용
    sprvsn_inst_cd_nm: str = Field(alias="sprvsnInstCdNm", default="")  # 주관기관
    sprt_trgt_min_age: str = Field(alias="sprtTrgtMinAge", default="")  # 최소연령
    sprt_trgt_max_age: str = Field(alias="sprtTrgtMaxAge", default="")  # 최대연령
    aply_url_addr: str = Field(alias="aplyUrlAddr", default="")  # 신청URL
    aply_ymd: str = Field(alias="aplyYmd", default="")           # 신청기간
    zip_cd: str = Field(alias="zipCd", default="")               # 법정시군구코드

    model_config = {"populate_by_name": True}


class PolicyDetail(_NullSafeModel):
    """상세 조회 — 목록과 동일한 스키마, 상세 필드 포함."""
    plcy_no: str = Field(alias="plcyNo", default="")
    plcy_nm: str = Field(alias="plcyNm", default="")
    plcy_kywd_nm: str = Field(alias="plcyKywdNm", default="")
    plcy_expln_cn: str = Field(alias="plcyExplnCn", default="")
    lclsf_nm: str = Field(alias="lclsfNm", default="")
    mclsf_nm: str = Field(alias="mclsfNm", default="")
    plcy_sprt_cn: str = Field(alias="plcySprtCn", default="")
    sprvsn_inst_cd_nm: str = Field(alias="sprvsnInstCdNm", default="")
    oper_inst_cd_nm: str = Field(alias="operInstCdNm", default="")       # 운영기관
    sprt_trgt_min_age: str = Field(alias="sprtTrgtMinAge", default="")
    sprt_trgt_max_age: str = Field(alias="sprtTrgtMaxAge", default="")
    plcy_aply_mthd_cn: str = Field(alias="plcyAplyMthdCn", default="")  # 신청방법
    aply_url_addr: str = Field(alias="aplyUrlAddr", default="")
    aply_ymd: str = Field(alias="aplyYmd", default="")
    sbmsn_dcmnt_cn: str = Field(alias="sbmsnDcmntCn", default="")        # 제출서류
    earn_etc_cn: str = Field(alias="earnEtcCn", default="")              # 소득조건
    mrg_stts_cd: str = Field(alias="mrgSttsCd", default="")              # 결혼상태코드
    zip_cd: str = Field(alias="zipCd", default="")
    ref_url_addr1: str = Field(alias="refUrlAddr1", default="")          # 참고URL1

    model_config = {"populate_by_name": True}


class Pagging(_NullSafeModel):
    tot_count: int = Field(alias="totCount", default=0)
    page_num: int = Field(alias="pageNum", default=1)
    page_size: int = Field(alias="pageSize", default=10)

    model_config = {"populate_by_name": True}


class PolicyListResponse(BaseModel):
    pagging: Pagging
    items: list[PolicyListItem]


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
            follow_redirects=False,
            verify=False,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()

    def _base_params(self) -> dict[str, str]:
        return {"apiKeyNm": self._api_key, "rtnType": "json"}

    async def get_policy_list(
        self,
        *,
        keyword: str = "",
        region: str = "",
        category: str = "",
        policy_name: str = "",
        page: int = 1,
        page_size: int = 10,
    ) -> PolicyListResponse:
        """청년정책 목록 조회.

        Args:
            keyword: 정책 키워드 (plcyKywdNm). 예: '주거지원,청년'.
            region: 법정시군구코드 5자리 (zipCd). 예: '11000' (서울).
            category: 정책대분류명 (lclsfNm). 예: '일자리', '주거'.
            policy_name: 정책명 검색 (plcyNm).
            page: 페이지 번호 (1-based).
            page_size: 페이지당 결과 수.
        """
        assert self._client is not None

        params: dict[str, str] = {
            **self._base_params(),
            "pageNum": str(page),
            "pageSize": str(page_size),
            "pageType": "1",
        }
        if keyword:
            params["plcyKywdNm"] = keyword
        if region:
            params["zipCd"] = region
        if category:
            params["lclsfNm"] = category
        if policy_name:
            params["plcyNm"] = policy_name

        try:
            resp = await self._client.get("/go/ythip/getPlcy", params=params)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise RuntimeError(f"온통청년 API 응답 timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"온통청년 API HTTP 오류 {e.response.status_code}"
            ) from e

        data = resp.json()
        result = data.get("result", {})
        pagging = Pagging.model_validate(result.get("pagging", {}))
        items = [
            PolicyListItem.model_validate(item)
            for item in result.get("youthPolicyList", [])
        ]
        return PolicyListResponse(pagging=pagging, items=items)

    async def get_policy_detail(self, policy_no: str) -> PolicyDetail:
        """청년정책 상세 조회.

        Args:
            policy_no: 정책 번호 (plcyNo). 목록 조회 결과의 plcyNo 사용.
        """
        assert self._client is not None

        params: dict[str, str] = {
            **self._base_params(),
            "pageType": "2",
            "plcyNo": policy_no,
        }

        try:
            resp = await self._client.get("/go/ythip/getPlcy", params=params)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise RuntimeError(f"온통청년 API 응답 timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"온통청년 API HTTP 오류 {e.response.status_code}"
            ) from e

        data = resp.json()
        result = data.get("result", {})
        items = result.get("youthPolicyList", [])
        if not items:
            raise RuntimeError(f"정책 {policy_no}를 찾을 수 없습니다.")
        return PolicyDetail.model_validate(items[0])
