"""Dependency providers for presentation layer components."""

from __future__ import annotations

from functools import lru_cache
from uuid import uuid4

from fastapi import Request

from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.utils.logging import get_correlation_id

from .interface import ResponsePresenter
from .jsonapi import JSONAPIPresenter
from .lifecycle import RequestLifecycle, current_lifecycle


@lru_cache(maxsize=None)
def _presenter_for(header_name: str) -> ResponsePresenter:
    return JSONAPIPresenter(correlation_header=header_name)


def get_response_presenter() -> ResponsePresenter:
    """Return the default response presenter instance."""

    settings = get_settings()
    header = settings.observability.logging.correlation_id_header or "X-Correlation-ID"
    return _presenter_for(header)


def get_request_lifecycle(request: Request) -> RequestLifecycle:
    """Expose the lifecycle tracker bound by the middleware."""

    lifecycle = getattr(request.state, "lifecycle", None) or current_lifecycle()
    if lifecycle is None:
        correlation_id = getattr(request.state, "correlation_id", None)
        correlation_id = correlation_id or get_correlation_id() or str(uuid4())
        lifecycle = RequestLifecycle(
            method=request.method,
            path=request.url.path,
            correlation_id=correlation_id,
        )
        request.state.lifecycle = lifecycle
        request.state.correlation_id = correlation_id
    return lifecycle


__all__ = ["get_response_presenter", "get_request_lifecycle"]
