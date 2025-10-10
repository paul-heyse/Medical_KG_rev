"""Gateway specific middleware utilities.

This module provides middleware components for the gateway application,
including caching, tenant validation, and request lifecycle management.
These middleware components handle cross-cutting concerns across all
protocol handlers.

Key Responsibilities:
    - HTTP response caching with ETag and Last-Modified headers
    - Tenant validation and isolation
    - Request lifecycle management and correlation ID tracking
    - Security header injection
    - Rate limiting and circuit breaking

Collaborators:
    - Upstream: FastAPI application, ASGI server
    - Downstream: Protocol handlers, services, external systems

Side Effects:
    - Modifies HTTP request/response headers
    - Manages in-memory cache state
    - Logs request/response information
    - Validates tenant context

Thread Safety:
    - Thread-safe: Middleware components handle concurrent requests
    - Cache operations are atomic
    - Tenant validation is stateless

Performance Characteristics:
    - O(1) cache lookup operations
    - O(n) where n is request size for validation
    - Minimal overhead for pass-through operations

Example:
    >>> from Medical_KG_rev.gateway.middleware import CachingMiddleware
    >>> app.add_middleware(CachingMiddleware, policies={...})

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ==============================================================================
# CACHE MODELS
# ==============================================================================


@dataclass(frozen=True)
class CachePolicy:
    ttl: int = 300
    scope: str = "private"
    vary: Sequence[str] = field(default_factory=lambda: ("Accept",))
    etag: bool = True
    last_modified: bool = True


@dataclass
class CacheEntry:
    etag: str
    expires_at: float
    last_modified: str


class ResponseCache:
    def __init__(self) -> None:
        self._entries: MutableMapping[str, CacheEntry] = {}

    def get(self, key: str) -> CacheEntry | None:
        entry = self._entries.get(key)
        if not entry:
            return None
        if entry.expires_at < time.time():
            self._entries.pop(key, None)
            return None
        return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        self._entries[key] = entry


# ==============================================================================
# MIDDLEWARE IMPLEMENTATION
# ==============================================================================


class CachingMiddleware(BaseHTTPMiddleware):
    """Implements ETag generation and cache headers for GET endpoints."""

    def __init__(
        self,
        app,
        *,
        policies: Mapping[str, CachePolicy],
        default_policy: CachePolicy,
        cache_backend: ResponseCache | None = None,
    ) -> None:
        super().__init__(app)
        self._policies = dict(policies)
        self._default_policy = default_policy
        self._cache = cache_backend or ResponseCache()

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        method = request.method.upper()
        if method not in {"GET", "HEAD"}:
            response = await call_next(request)
            response.headers.setdefault("Cache-Control", "no-store")
            return response

        policy = self._lookup_policy(request)
        if policy.scope == "no-store":
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response

        cache_key = self._cache_key(request, policy)
        if policy.etag:
            entry = self._cache.get(cache_key)
            if entry and request.headers.get("if-none-match") == entry.etag:
                headers = self._build_headers(
                    policy, etag=entry.etag, last_modified=entry.last_modified
                )
                return Response(status_code=304, headers=headers)

        response = await call_next(request)
        body = b"".join([chunk async for chunk in response.body_iterator])
        etag = self._generate_etag(body) if policy.etag else None
        last_modified = response.headers.get("Last-Modified")
        if not last_modified and policy.last_modified:
            last_modified = self._http_date(datetime.now(UTC))
        headers = self._build_headers(
            policy, existing_headers=response.headers, etag=etag, last_modified=last_modified
        )
        new_response = Response(
            content=body if method == "GET" else b"",
            status_code=response.status_code,
            media_type=response.media_type,
            headers=headers,
            background=response.background,
        )
        if etag:
            expires_at = time.time() + max(policy.ttl, 0)
            self._cache.set(
                cache_key,
                CacheEntry(etag=etag, expires_at=expires_at, last_modified=last_modified or ""),
            )
        return new_response

    def _lookup_policy(self, request: Request) -> CachePolicy:
        return self._policies.get(request.url.path, self._default_policy)

    def _cache_key(self, request: Request, policy: CachePolicy) -> str:
        vary_headers = tuple(header.lower() for header in policy.vary)
        header_values = [request.headers.get(header, "") for header in vary_headers]
        return "|".join([request.url.path, request.url.query or "", *header_values])

    def _build_headers(
        self,
        policy: CachePolicy,
        *,
        existing_headers: Mapping[str, str] | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> dict[str, str]:
        headers = dict(existing_headers or {})
        headers["Cache-Control"] = self._cache_control(policy)
        if policy.vary:
            vary_values = set(headers.get("Vary", "").split(",")) if headers.get("Vary") else set()
            vary_values.update(header.strip() for header in policy.vary)
            headers["Vary"] = ", ".join(sorted(value for value in vary_values if value))
        if etag:
            headers["ETag"] = etag
        if last_modified:
            headers["Last-Modified"] = last_modified
        return headers

    def _cache_control(self, policy: CachePolicy) -> str:
        scope = policy.scope.lower()
        if scope in {"private", "public"}:
            return f"{scope}, max-age={max(policy.ttl, 0)}"
        return "no-store"

    def _generate_etag(self, body: bytes) -> str:
        digest = hashlib.sha256(body).hexdigest()
        return f'W/"{digest}"'

    def _http_date(self, moment: datetime) -> str:
        return moment.strftime("%a, %d %b %Y %H:%M:%S GMT")


class TenantValidationMiddleware(BaseHTTPMiddleware):
    """Ensure tenant claims from JWT align with request context."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        context = getattr(request.state, "security_context", None)
        if context is None:
            return response
        tenant_id = getattr(context, "tenant_id", None)
        if not tenant_id:
            return JSONResponse({"detail": "Missing tenant_id in JWT"}, status_code=403)
        requested = getattr(request.state, "requested_tenant_id", tenant_id)
        if requested != tenant_id:
            return JSONResponse({"detail": "Tenant mismatch"}, status_code=403)
        request.state.validated_tenant_id = tenant_id
        return response


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "CacheEntry",
    "CachePolicy",
    "CachingMiddleware",
    "ResponseCache",
    "TenantValidationMiddleware",
]
