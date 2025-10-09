"""JSON:API presenter implementation for REST responses.

This module provides a JSON:API compliant response presenter that formats
API responses according to the JSON:API specification. It handles data
serialization, error formatting, metadata inclusion, and response
optimization features like compression and caching.

Key Responsibilities:
    - JSON:API compliant response formatting
    - Data serialization and normalization
    - Error response formatting
    - Metadata and correlation ID inclusion
    - Response compression and caching headers

Collaborators:
    - Upstream: REST router endpoints
    - Downstream: FastAPI Response objects, lifecycle management

Side Effects:
    - Creates HTTP response objects
    - Sets response headers (Content-Type, Cache-Control, etc.)
    - Compresses response bodies when beneficial

Thread Safety:
    - Thread-safe: Stateless presenter with no shared mutable state

Performance Characteristics:
    - O(n) serialization where n is response size
    - O(1) header operations
    - Compression reduces bandwidth usage

Example:
    >>> from Medical_KG_rev.gateway.presentation.jsonapi import JSONAPIPresenter
    >>> presenter = JSONAPIPresenter()
    >>> response = presenter.success({"data": "value"})

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from Medical_KG_rev.utils.logging import get_correlation_id

from .errors import ErrorDetail
from .interface import ResponsePresenter
from .lifecycle import current_lifecycle

# ==============================================================================
# CONSTANTS
# ==============================================================================

JSONAPI_CONTENT_TYPE = "application/vnd.api+json"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _normalise_payload(data: Any) -> Any:
    """Normalize payload data for JSON:API serialization.

    Args:
        data: Raw data to normalize.

    Returns:
        Normalized data suitable for JSON:API serialization.

    """
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, dict)):
        return [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in data
        ]
    return data


# ==============================================================================
# PRESENTER IMPLEMENTATION
# ==============================================================================

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


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "JSONAPI_CONTENT_TYPE",
    "JSONAPIPresenter",
    "_normalise_payload",
]
