"""FastAPI dependencies for authentication and authorization."""

from __future__ import annotations

"""FastAPI dependencies wiring authentication subsystems together.

This module exposes dependency factories and helper utilities used by the REST
gateway. The helpers integrate API key validation, OAuth token verification,
and rate limiting into a cohesive dependency graph that can be re-used by any
endpoint requiring authenticated access.
"""

# ============================================================================
# IMPORTS
# ============================================================================

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, Header, HTTPException, Request, status

from .api_keys import APIKeyManager, build_api_key_manager
from .context import SecurityContext
from .jwt import AuthenticationError, JWTAuthenticator, build_authenticator
from .rate_limit import RateLimiter, RateLimitExceeded, build_rate_limiter

# ============================================================================
# DEPENDENCY FACTORIES
# ============================================================================


def _get_authenticator() -> JWTAuthenticator:
    """Return a configured :class:`JWTAuthenticator` instance."""
    return build_authenticator()


def _get_rate_limiter() -> RateLimiter:
    """Return the process-wide :class:`RateLimiter`."""
    return build_rate_limiter()


def _get_api_key_manager() -> APIKeyManager:
    """Return the API key manager used for header authentication."""
    return build_api_key_manager()


# ============================================================================
# AUTHENTICATION HELPERS
# ============================================================================


async def get_security_context(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
    authenticator: JWTAuthenticator = Depends(_get_authenticator),
    api_keys: APIKeyManager = Depends(_get_api_key_manager),
) -> SecurityContext:
    """Authenticate the incoming request and populate :class:`SecurityContext`.

    Args:
        request: Current FastAPI request instance.
        authorization: Optional bearer token header.
        api_key: Optional API key header.
        authenticator: JWT authenticator dependency.
        api_keys: API key manager dependency.

    Returns:
        Authenticated :class:`SecurityContext` with scopes and claims.

    Raises:
        HTTPException: When authentication fails for any reason.

    """
    if api_key:
        try:
            key_id, record = api_keys.authenticate(api_key)
        except PermissionError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            ) from exc
        context = SecurityContext(
            subject=f"api-key:{key_id}",
            tenant_id=record.tenant_id,
            scopes=set(record.scopes),
            expires_at=None,
            claims={"api_key": True},
            auth_type="api_key",
            key_id=key_id,
        )
        request.state.security_context = context
        return context

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication"
        )

    token = authorization.split(" ", 1)[1]
    cache: dict[str, dict[str, Any]] = getattr(request.app.state, "jwt_cache", {})
    if not hasattr(request.app.state, "jwt_cache"):
        request.app.state.jwt_cache = cache
    now = datetime.now(UTC)
    cached = cache.get(token)
    payload: dict[str, Any]
    if cached and cached["expires_at"] > now:  # type: ignore[index]
        payload = cached["payload"]  # type: ignore[index]
    else:
        try:
            payload = await authenticator.authenticate(token)
        except AuthenticationError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        exp = payload.get("exp")
        expires_at = (
            datetime.fromtimestamp(exp, tz=UTC)
            if isinstance(exp, (int, float))
            else now + timedelta(minutes=5)
        )
        cache[token] = {"payload": payload, "expires_at": expires_at}
    tenant_id = payload.get("tenant_id") or payload.get("tenant")
    if not tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing tenant claim")
    raw_scopes = payload.get("scope") or payload.get("scopes") or []
    if isinstance(raw_scopes, str):
        scopes = {scope for scope in raw_scopes.split() if scope}
    else:
        scopes = set(raw_scopes)
    expires_at = cache[token]["expires_at"]
    context = SecurityContext(
        subject=str(payload.get("sub", "anonymous")),
        tenant_id=str(tenant_id),
        scopes=scopes,
        expires_at=expires_at,
        claims=payload,
        auth_type="oauth",
        token=token,
    )
    request.state.security_context = context
    return context


def secure_endpoint(*, scopes: Sequence[str], endpoint: str) -> Callable[..., SecurityContext]:
    """Create a dependency enforcing scopes and rate limits for an endpoint.

    Args:
        scopes: Required scopes for accessing the endpoint.
        endpoint: Logical endpoint identifier used for rate limiting.

    Returns:
        FastAPI dependency that yields an authenticated :class:`SecurityContext`.

    """

    async def dependency(
        context: SecurityContext = Depends(get_security_context),
        rate_limiter: RateLimiter = Depends(_get_rate_limiter),
    ) -> SecurityContext:
        """Validate rate limits and scope membership for the request.

        Returns:
            Authenticated security context passed through when checks succeed.

        Raises:
            HTTPException: When rate limits are exceeded or scopes are missing.

        """
        try:
            rate_limiter.check(context.identity, endpoint)
        except RateLimitExceeded as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": f"{int(exc.retry_after)}"},
            ) from exc
        missing = [scope for scope in scopes if not context.has_scope(scope)]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scopes: {', '.join(missing)}",
            )
        return context

    return dependency


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["get_security_context", "secure_endpoint"]
