"""JWT authentication utilities for verifying OAuth access tokens.

The gateway uses JSON Web Tokens (JWTs) issued by external identity providers
to authenticate requests. This module handles fetching signing keys via JWKS,
validating tokens, and exposing a dependency-friendly authenticator factory.
"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import asyncio
import time
from collections.abc import Iterable
from typing import Any

import httpx
from jose import JWTError, jwt

from ..config.settings import AppSettings, get_settings


# ============================================================================
# JWKS CACHE
# ============================================================================


class JWKSCache:
    """Cache JWKS responses to reduce network calls for token validation.

    Attributes:
        _url: JWKS endpoint URL provided by the identity provider.
        _ttl: Cache TTL in seconds.
        _expires_at: Epoch timestamp for when the cache should refresh.
        _keys: Cached key material keyed by ``kid``.
    """

    def __init__(self, url: str, *, ttl: int = 300) -> None:
        """Initialize the cache with the JWKS endpoint and TTL.

        Args:
            url: HTTPS URL to the JWKS endpoint.
            ttl: Number of seconds to cache keys before refreshing.
        """

        self._url = url
        self._ttl = ttl
        self._expires_at = 0.0
        self._keys: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get_key(self, kid: str) -> dict[str, Any] | None:
        """Return key material for the given ``kid`` if available.

        Args:
            kid: Key identifier extracted from the JWT header.

        Returns:
            Dictionary with JWKS key material or ``None`` when not found.
        """

        await self._ensure_keys()
        return self._keys.get(kid)

    async def _ensure_keys(self) -> None:
        """Refresh cached keys when stale or missing."""

        async with self._lock:
            if self._keys and time.time() < self._expires_at:
                return
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self._url)
                response.raise_for_status()
                payload = response.json()
            keys = payload.get("keys", [])
            self._keys = {key["kid"]: key for key in keys if "kid" in key}
            self._expires_at = time.time() + self._ttl


# ============================================================================
# AUTHENTICATOR IMPLEMENTATION
# ============================================================================


class JWTAuthenticator:
    """Validate OAuth 2.0 JWT access tokens using configured JWKS keys.

    Attributes:
        issuer: Expected issuer claim.
        audience: Expected audience claim.
        algorithms: Acceptable signature algorithms.
        cache: :class:`JWKSCache` storing signing keys.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str,
        algorithms: Iterable[str] = ("RS256", "RS384", "RS512"),
        cache_ttl: int = 300,
    ) -> None:
        """Initialize authenticator with issuer metadata and algorithms.

        Args:
            issuer: Expected issuer claim value.
            audience: Expected audience claim value.
            jwks_url: JWKS endpoint URL.
            algorithms: Acceptable signature algorithms.
            cache_ttl: Seconds to cache JWKS responses.
        """

        self.issuer = issuer
        self.audience = audience
        self.algorithms = tuple(algorithms)
        self.cache = JWKSCache(jwks_url, ttl=cache_ttl)

    async def authenticate(self, token: str) -> dict[str, Any]:
        """Validate the provided JWT and return decoded claims.

        Args:
            token: Raw bearer token string from the Authorization header.

        Returns:
            Decoded token claims when validation succeeds.

        Raises:
            AuthenticationError: When token structure or signature is invalid.
        """

        try:
            header = jwt.get_unverified_header(token)
        except JWTError as exc:  # pragma: no cover - defensive
            raise AuthenticationError("Invalid token header") from exc
        kid = header.get("kid")
        if not kid:
            raise AuthenticationError("Token missing key identifier")
        key_data = await self.cache.get_key(kid)
        if not key_data:
            raise AuthenticationError("Signing key not found")
        try:
            payload = jwt.decode(
                token,
                key_data,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
            )
        except JWTError as exc:
            raise AuthenticationError(str(exc)) from exc
        return payload


# ============================================================================
# ERRORS AND FACTORY
# ============================================================================


class AuthenticationError(RuntimeError):
    """Raised when authentication fails."""


def build_authenticator(settings: AppSettings | None = None) -> JWTAuthenticator:
    """Construct a :class:`JWTAuthenticator` using application settings.

    Args:
        settings: Optional settings override for dependency injection.

    Returns:
        Configured :class:`JWTAuthenticator` instance.
    """

    cfg = (settings or get_settings()).security.oauth
    return JWTAuthenticator(
        issuer=cfg.issuer,
        audience=cfg.audience,
        jwks_url=cfg.jwks_url,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["AuthenticationError", "JWTAuthenticator", "JWKSCache", "build_authenticator"]
