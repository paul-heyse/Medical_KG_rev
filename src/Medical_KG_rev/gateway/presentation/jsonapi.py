"""JSON:API presenter implementation for REST responses."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .errors import ErrorDetail
from .interface import ResponsePresenter

JSONAPI_CONTENT_TYPE = "application/vnd.api+json"


def _normalise_payload(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, dict)):
        return [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in data
        ]
    return data


class JSONAPIPresenter(ResponsePresenter):
    """Presenter producing JSON:API compliant envelopes."""

    media_type = JSONAPI_CONTENT_TYPE

    def success(
        self,
        data: Any,
        *,
        status_code: int = 200,
        meta: Mapping[str, Any] | None = None,
    ) -> JSONResponse:
        payload = {"data": _normalise_payload(data), "meta": dict(meta or {})}
        return JSONResponse(payload, status_code=status_code, media_type=self.media_type)

    def error(
        self,
        detail: Any,
        *,
        status_code: int = 400,
    ) -> JSONResponse:
        if isinstance(detail, ErrorDetail):
            payload = {"errors": [detail.as_json()]}
        elif isinstance(detail, Mapping):
            payload = {"errors": [_normalise_payload(detail)]}
        else:
            payload = {"errors": [{"detail": str(detail)}]}
        return JSONResponse(payload, status_code=status_code, media_type=self.media_type)
