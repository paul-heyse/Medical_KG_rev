"""JWT validation utilities."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Iterable, Optional

import httpx
from jose import JWTError, jwt

from ..config.settings import AppSettings, get_settings


class JWKSCache:
    """Caches JWKS responses to avoid repeated network calls."""

    def __init__(self, url: str, *, ttl: int = 300) -> None:
        self._url = url
        self._ttl = ttl
        self._expires_at = 0.0
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get_key(self, kid: str) -> Optional[Dict[str, Any]]:
        await self._ensure_keys()
        return self._keys.get(kid)

    async def _ensure_keys(self) -> None:
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


class JWTAuthenticator:
    """Validates OAuth 2.0 JWT access tokens."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str,
        algorithms: Iterable[str] = ("RS256", "RS384", "RS512"),
        cache_ttl: int = 300,
    ) -> None:
        self.issuer = issuer
        self.audience = audience
        self.algorithms = tuple(algorithms)
        self.cache = JWKSCache(jwks_url, ttl=cache_ttl)

    async def authenticate(self, token: str) -> Dict[str, Any]:
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


class AuthenticationError(RuntimeError):
    """Raised when authentication fails."""


def build_authenticator(settings: Optional[AppSettings] = None) -> JWTAuthenticator:
    cfg = (settings or get_settings()).security.oauth
    return JWTAuthenticator(
        issuer=cfg.issuer,
        audience=cfg.audience,
        jwks_url=cfg.jwks_url,
    )
