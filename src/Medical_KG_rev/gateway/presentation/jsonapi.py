"""JSON:API presenter implementation for REST responses."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from typing import Any, Mapping, MutableMapping

from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .errors import ErrorDetail
from .interface import ResponsePresenter
from .lifecycle import current_lifecycle
from Medical_KG_rev.utils.logging import get_correlation_id

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

    def __init__(self, *, correlation_header: str = "X-Correlation-ID") -> None:
        self._correlation_header = correlation_header

    def _finalise(
        self,
        response: Response,
        *,
        status_code: int,
        cache_control: str | None,
        compression: str | None,
    ) -> Response:
        lifecycle = current_lifecycle()
        if lifecycle:
            lifecycle.set_cache_control(cache_control)
            lifecycle.set_compression(compression)
            lifecycle.complete(status_code)
            correlation_id = lifecycle.correlation_id
        else:
            correlation_id = get_correlation_id()
        if self._correlation_header and correlation_id:
            response.headers.setdefault(self._correlation_header, correlation_id)
        if lifecycle:
            response.headers.setdefault("X-Response-Time-Ms", f"{lifecycle.duration_ms:.2f}")
        elif "X-Response-Time-Ms" not in response.headers:
            response.headers["X-Response-Time-Ms"] = "0.00"
        if cache_control:
            response.headers.setdefault("Cache-Control", cache_control)
        if compression:
            response.headers.setdefault("Content-Encoding", compression)
        return response

    def _build_payload(
        self,
        *,
        data: Any | None = None,
        meta: Mapping[str, Any] | None = None,
        errors: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if data is not None:
            payload["data"] = _normalise_payload(data)
        if errors is not None:
            payload["errors"] = errors
        payload["meta"] = dict(meta or {})
        return payload

    def success(
        self,
        data: Any,
        *,
        status_code: int = 200,
        meta: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        cache_control: str | None = None,
        compress: bool = False,
    ) -> Response:
        payload = self._build_payload(data=data, meta=meta)
        response_headers: MutableMapping[str, str] = dict(headers or {})
        body = json.dumps(payload, separators=(",", ":"))
        if compress:
            compressed = gzip.compress(body.encode("utf-8"))
            response_headers.setdefault("Content-Encoding", "gzip")
            compression = "gzip"
            response = Response(
                content=compressed,
                status_code=status_code,
                media_type=self.media_type,
                headers=response_headers,
            )
        else:
            compression = None
            response = JSONResponse(
                payload,
                status_code=status_code,
                media_type=self.media_type,
                headers=response_headers,
            )
        return self._finalise(
            response,
            status_code=status_code,
            cache_control=cache_control,
            compression=compression,
        )

    def error(
        self,
        detail: Any,
        *,
        status_code: int = 400,
        headers: Mapping[str, str] | None = None,
        cache_control: str | None = None,
    ) -> Response:
        if isinstance(detail, ErrorDetail):
            errors = [detail.as_json()]
        elif isinstance(detail, Mapping):
            errors = [_normalise_payload(detail)]
        else:
            errors = [{"detail": str(detail)}]
        payload = self._build_payload(errors=errors)
        response = JSONResponse(
            payload,
            status_code=status_code,
            media_type=self.media_type,
            headers=dict(headers or {}),
        )
        return self._finalise(
            response,
            status_code=status_code,
            cache_control=cache_control,
            compression=None,
        )
