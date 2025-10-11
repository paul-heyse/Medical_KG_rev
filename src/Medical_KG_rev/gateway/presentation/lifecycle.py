"""Request lifecycle tracking helpers and middleware.

This module provides request lifecycle tracking functionality for the gateway,
including correlation ID management, timing measurements, and observability
integration. It tracks requests from start to finish and provides metadata
for response formatting and monitoring.

Key Responsibilities:
    - Request lifecycle tracking and timing
    - Correlation ID generation and management
    - Response metadata collection
    - Observability metrics emission
    - Middleware integration for automatic tracking

Collaborators:
    - Upstream: FastAPI middleware, request handlers
    - Downstream: Observability systems, response presenters

Side Effects:
    - Generates correlation IDs
    - Emits metrics to observability systems
    - Manages context variables for request state

Thread Safety:
    - Thread-safe: Uses context variables for request isolation
    - Each request has its own lifecycle instance

Performance Characteristics:
    - O(1) lifecycle operations
    - Minimal overhead for timing measurements
    - Context variable access is optimized

Example:
    >>> from Medical_KG_rev.gateway.presentation.lifecycle import RequestLifecycle
    >>> lifecycle = RequestLifecycle(method="GET", path="/api/test")
    >>> lifecycle.complete(200)

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from Medical_KG_rev.observability.metrics import (
    GPU_SERVICE_CALL_DURATION_SECONDS as REQUEST_LATENCY,
)
from Medical_KG_rev.observability.metrics import GPU_SERVICE_CALLS_TOTAL as REQUEST_COUNTER
from Medical_KG_rev.utils.logging import (
    bind_correlation_id,
    get_correlation_id,
    get_logger,
    reset_correlation_id,
)

# ==============================================================================
# GLOBAL STATE
# ==============================================================================

logger = get_logger(__name__)

_CURRENT_LIFECYCLE: ContextVar[RequestLifecycle | None] = ContextVar(
    "gateway_request_lifecycle",
    default=None,
)


# ==============================================================================
# LIFECYCLE MODELS
# ==============================================================================


@dataclass(slots=True)
class RequestLifecycle:
    """Tracks request timing, correlation identifiers, and response metadata."""

    method: str
    path: str
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: float = field(default_factory=perf_counter)
    finished_at: float | None = None
    status_code: int | None = None
    error: str | None = None
    cache_control: str | None = None
    compression: str | None = None

    def complete(self, status_code: int) -> None:
        """Record a successful response if not already completed."""
        if self.status_code is not None:
            return
        self.status_code = status_code
        self.finished_at = perf_counter()
        REQUEST_COUNTER.labels(self.method, self.path, str(status_code)).inc()
        REQUEST_LATENCY.labels(self.method, self.path).observe(self.duration_seconds)

    def fail(self, exc: BaseException, *, status_code: int = 500) -> None:
        """Record an error outcome for the request."""
        self.error = str(exc)
        self.complete(status_code)

    @property
    def duration_seconds(self) -> float:
        """Return the elapsed time since the lifecycle started."""
        end = self.finished_at or perf_counter()
        return max(end - self.started_at, 0.0)

    @property
    def duration_ms(self) -> float:
        return self.duration_seconds * 1000

    def set_cache_control(self, value: str | None) -> None:
        if value:
            self.cache_control = value

    def set_compression(self, encoding: str | None) -> None:
        if encoding:
            self.compression = encoding

    def apply(self, response: Response, *, correlation_header: str | None) -> None:
        """Apply lifecycle metadata to the outbound response headers."""
        if correlation_header:
            response.headers.setdefault(correlation_header, self.correlation_id)
        response.headers.setdefault("X-Response-Time-Ms", f"{self.duration_ms:.2f}")
        if self.cache_control:
            response.headers.setdefault("Cache-Control", self.cache_control)
        if self.compression:
            response.headers.setdefault("Content-Encoding", self.compression)

    def meta(self, extra: Mapping[str, object] | None = None) -> dict[str, object]:
        """Return a metadata dictionary enriched with correlation details."""
        payload: dict[str, object] = {
            "correlation_id": self.correlation_id,
            "duration_ms": round(self.duration_ms, 3),
        }
        if extra:
            payload.update(dict(extra))
        return payload


# ==============================================================================
# LIFECYCLE MANAGEMENT FUNCTIONS
# ==============================================================================


def current_lifecycle() -> RequestLifecycle | None:
    """Return the lifecycle bound to the current context, if any.

    Returns:
        Current request lifecycle instance or None if not set.

    """
    return _CURRENT_LIFECYCLE.get()


def push_lifecycle(lifecycle: RequestLifecycle) -> Token:
    """Bind a lifecycle object to the current context.

    Args:
        lifecycle: Request lifecycle instance to bind.

    Returns:
        Token for restoring the previous context.

    """
    return _CURRENT_LIFECYCLE.set(lifecycle)


def pop_lifecycle(token: Token) -> None:
    """Reset the context variable token captured via :func:`push_lifecycle`.

    Args:
        token: Token returned by push_lifecycle.

    """
    _CURRENT_LIFECYCLE.reset(token)


# ==============================================================================
# MIDDLEWARE IMPLEMENTATION
# ==============================================================================


class RequestLifecycleMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that binds lifecycle information to each request."""

    def __init__(self, app, *, correlation_header: str | None = None):  # type: ignore[override]
        super().__init__(app)
        self._correlation_header = correlation_header or "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        provided = request.headers.get(self._correlation_header)
        correlation_id = provided or getattr(request.state, "correlation_id", None)
        correlation_id = correlation_id or get_correlation_id() or str(uuid4())

        lifecycle = RequestLifecycle(
            method=request.method,
            path=request.url.path,
            correlation_id=correlation_id,
        )
        request.state.lifecycle = lifecycle
        request.state.correlation_id = correlation_id

        token = bind_correlation_id(correlation_id)
        ctx_token = push_lifecycle(lifecycle)

        logger.info(
            "gateway.request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "correlation_id": correlation_id,
            },
        )

        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover - exercised via FastAPI error paths
            lifecycle.fail(exc)
            logger.exception(
                "gateway.request.error",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "correlation_id": correlation_id,
                    "duration_ms": round(lifecycle.duration_ms, 2),
                },
            )
            pop_lifecycle(ctx_token)
            reset_correlation_id(token)
            raise

        if lifecycle.status_code is None:
            lifecycle.complete(response.status_code)

        lifecycle.apply(response, correlation_header=self._correlation_header)

        logger.info(
            "gateway.response",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(lifecycle.duration_ms, 2),
                "correlation_id": correlation_id,
            },
        )

        pop_lifecycle(ctx_token)
        reset_correlation_id(token)
        return response


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "RequestLifecycle",
    "RequestLifecycleMiddleware",
    "current_lifecycle",
    "pop_lifecycle",
    "push_lifecycle",
]
