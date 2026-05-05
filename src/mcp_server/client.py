"""온통청년 OpenAPI httpx 클라이언트.

엔드포인트: https://www.youthcenter.go.kr/go/ythip/getPlcy
응답 형식: JSON
인증: apiKeyNm 쿼리 파라미터
"""
# ============================================================
# mcp_server/client.py — 온통청년 공공 API 실제 HTTP 호출 담당
#
# 역할: 정부 온통청년 포털(youthcenter.go.kr)의 OpenAPI를 호출해서
#       현재 모집 중인 청년 정책 목록 또는 특정 정책의 상세 정보를 가져온다.
#
# 왜 httpx인가?
#   - requests는 동기(blocking) 라이브러리라 async/await 환경에서 느리다.
#   - httpx는 async를 지원해서 MCP 서버(비동기 환경)와 잘 맞는다.
#
# 사용 패턴 (비동기 컨텍스트 매니저):
#   async with YouthPolicyClient() as client:
#       result = await client.get_policy_list(keyword="주거")
# ============================================================
from __future__ import annotations

from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, model_validator

from src.config import settings

BASE_URL = "https://www.youthcenter.go.kr"
TIMEOUT = httpx.Timeout(10.0, connect=5.0)  # 전체 응답 10초, 연결 5초 타임아웃


# ============================================================
# Pydantic 데이터 모델 — API 응답 JSON을 파이썬 객체로 변환
# ============================================================

class _NullSafeModel(BaseModel):
    """API가 null 반환 시 빈 문자열로 변환.

    온통청년 API는 값이 없을 때 None(null)을 반환하는 경우가 있다.
    이를 처리하지 않으면 str 타입 필드에서 TypeErrror가 발생하므로
    모든 None을 "" 로 바꿔주는 공통 로직을 부모 클래스에 정의.
    """

    @model_validator(mode="before")  # 필드 검증 전에 전체 딕셔너리를 먼저 처리
    @classmethod
    def coerce_null(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: ("" if v is None else v) for k, v in data.items()}
        return data


class PolicyListItem(_NullSafeModel):
    # API 응답의 JSON 키(plcyNo 등)를 파이썬 속성명(plcy_no 등)으로 매핑
    # alias= 가 API 응답의 실제 JSON 키
    plcy_no: str = Field(alias="plcyNo", default="")            # 정책번호 (고유 식별자)
    plcy_nm: str = Field(alias="plcyNm", default="")            # 정책명
    plcy_kywd_nm: str = Field(alias="plcyKywdNm", default="")   # 키워드
    plcy_expln_cn: str = Field(alias="plcyExplnCn", default="") # 정책설명
    lclsf_nm: str = Field(alias="lclsfNm", default="")          # 정책대분류명 (일자리, 주거 등)
    mclsf_nm: str = Field(alias="mclsfNm", default="")          # 정책중분류명
    plcy_sprt_cn: str = Field(alias="plcySprtCn", default="")   # 지원내용
    sprvsn_inst_cd_nm: str = Field(alias="sprvsnInstCdNm", default="")  # 주관기관명
    sprt_trgt_min_age: str = Field(alias="sprtTrgtMinAge", default="")  # 지원 최소 연령
    sprt_trgt_max_age: str = Field(alias="sprtTrgtMaxAge", default="")  # 지원 최대 연령
    aply_url_addr: str = Field(alias="aplyUrlAddr", default="")  # 신청 URL
    aply_ymd: str = Field(alias="aplyYmd", default="")           # 신청기간 (시작일~종료일)
    zip_cd: str = Field(alias="zipCd", default="")               # 법정시군구코드 (지역 코드)

    model_config = {"populate_by_name": True}  # alias 말고 파이썬 속성명으로도 접근 허용


class PolicyDetail(_NullSafeModel):
    """상세 조회 — 목록과 동일한 스키마, 상세 필드 포함.

    get_policy_detail API(pageType=2)의 응답.
    목록보다 더 많은 필드(신청방법, 제출서류, 소득조건 등)가 포함됨.
    """
    plcy_no: str = Field(alias="plcyNo", default="")
    plcy_nm: str = Field(alias="plcyNm", default="")
    plcy_kywd_nm: str = Field(alias="plcyKywdNm", default="")
    plcy_expln_cn: str = Field(alias="plcyExplnCn", default="")
    lclsf_nm: str = Field(alias="lclsfNm", default="")
    mclsf_nm: str = Field(alias="mclsfNm", default="")
    plcy_sprt_cn: str = Field(alias="plcySprtCn", default="")
    sprvsn_inst_cd_nm: str = Field(alias="sprvsnInstCdNm", default="")
    oper_inst_cd_nm: str = Field(alias="operInstCdNm", default="")       # 운영기관 (주관기관과 다를 수 있음)
    sprt_trgt_min_age: str = Field(alias="sprtTrgtMinAge", default="")
    sprt_trgt_max_age: str = Field(alias="sprtTrgtMaxAge", default="")
    plcy_aply_mthd_cn: str = Field(alias="plcyAplyMthdCn", default="")  # 신청 방법 (온라인/방문 등)
    aply_url_addr: str = Field(alias="aplyUrlAddr", default="")
    aply_ymd: str = Field(alias="aplyYmd", default="")
    sbmsn_dcmnt_cn: str = Field(alias="sbmsnDcmntCn", default="")        # 제출 서류 목록
    earn_etc_cn: str = Field(alias="earnEtcCn", default="")              # 소득/재산 조건
    mrg_stts_cd: str = Field(alias="mrgSttsCd", default="")              # 결혼 상태 조건 코드
    zip_cd: str = Field(alias="zipCd", default="")
    ref_url_addr1: str = Field(alias="refUrlAddr1", default="")          # 참고 URL

    model_config = {"populate_by_name": True}


class Pagging(_NullSafeModel):
    # 페이지네이션 정보 — 전체 결과가 몇 건인지, 현재 몇 페이지인지
    tot_count: int = Field(alias="totCount", default=0)   # 전체 검색 결과 수
    page_num: int = Field(alias="pageNum", default=1)     # 현재 페이지 번호
    page_size: int = Field(alias="pageSize", default=10)  # 페이지당 결과 수

    model_config = {"populate_by_name": True}


class PolicyListResponse(BaseModel):
    # get_policy_list 호출 결과를 하나로 묶은 컨테이너
    pagging: Pagging                  # 전체 건수, 현재 페이지 정보
    items: list[PolicyListItem]       # 실제 정책 목록


# ============================================================
# HTTP 클라이언트 클래스
# ============================================================

class YouthPolicyClient:
    """온통청년 OpenAPI 비동기 클라이언트.

    반드시 async with 구문으로 사용해야 HTTP 연결이 제대로 열리고 닫힌다:
        async with YouthPolicyClient() as client:
            await client.get_policy_list(...)
    """

    def __init__(self) -> None:
        api_key = settings.youth_policy_api_key
        if not api_key:
            raise ValueError("YOUTH_POLICY_API_KEY가 설정되지 않았습니다.")
        self._api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None  # 아직 연결 안 된 상태

    async def __aenter__(self) -> "YouthPolicyClient":
        # async with 진입 시: HTTP 클라이언트 객체 생성 (연결 풀 초기화)
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=TIMEOUT,
            follow_redirects=False,
            verify=False,  # SSL 인증서 검증 비활성화 (정부 API 인증서 이슈 대응)
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        # async with 블록을 벗어날 때: HTTP 연결 정상 종료
        if self._client:
            await self._client.aclose()

    def _base_params(self) -> dict[str, str]:
        # 모든 API 호출에 공통으로 붙는 파라미터
        # apiKeyNm: 인증 키, rtnType: 응답 형식(json)
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

        # 공통 파라미터 + 목록 조회 전용 파라미터 합치기
        params: dict[str, str] = {
            **self._base_params(),
            "pageNum": str(page),
            "pageSize": str(page_size),
            "pageType": "1",  # pageType=1 이 목록 조회 모드
        }
        # 비어있지 않은 필터만 파라미터에 추가
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
            resp.raise_for_status()  # HTTP 4xx/5xx 에러 시 예외 발생
        except httpx.TimeoutException as e:
            raise RuntimeError(f"온통청년 API 응답 timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"온통청년 API HTTP 오류 {e.response.status_code}"
            ) from e

        # JSON 응답 파싱: {"result": {"pagging": {...}, "youthPolicyList": [...]}}
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
            이 번호는 get_policy_list 결과에서 가져와야 한다.
        """
        assert self._client is not None

        params: dict[str, str] = {
            **self._base_params(),
            "pageType": "2",       # pageType=2 가 상세 조회 모드
            "plcyNo": policy_no,   # 조회할 정책 번호
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
        # 상세 조회는 리스트지만 항상 1개만 반환됨
        return PolicyDetail.model_validate(items[0])
