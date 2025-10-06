from __future__ import annotations

import hashlib
import time
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from Medical_KG_rev.auth.api_keys import APIKeyManager, build_api_key_manager
from Medical_KG_rev.auth.audit import get_audit_trail
from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.auth.dependencies import secure_endpoint
from Medical_KG_rev.auth.jwt import JWTAuthenticator
from Medical_KG_rev.auth.rate_limit import RateLimiter, RateLimitSettings


@pytest.mark.anyio("asyncio")
async def test_jwt_authenticator_validates_token(monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private_key.public_key().public_numbers()
    n = int.to_bytes(public_numbers.n, (public_numbers.n.bit_length() + 7) // 8, "big")
    e = int.to_bytes(public_numbers.e, (public_numbers.e.bit_length() + 7) // 8, "big")

    def b64(value: bytes) -> str:
        import base64

        return base64.urlsafe_b64encode(value).rstrip(b"=").decode()

    jwks = {
        "keys": [
            {
                "kid": "test",
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "n": b64(n),
                "e": b64(e),
            }
        ]
    }

    class DummyResponse:
        def __init__(self, data: dict[str, Any]):
            self._data = data

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._data

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.calls = []

        async def __aenter__(self) -> DummyClient:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str):  # type: ignore[override]
            self.calls.append(url)
            return DummyResponse(jwks)

    dummy_client = DummyClient()
    monkeypatch.setattr(
        "Medical_KG_rev.auth.jwt.httpx.AsyncClient", lambda *args, **kwargs: dummy_client
    )

    authenticator = JWTAuthenticator(
        issuer="https://issuer",
        audience="medical-kg",
        jwks_url="https://issuer/.well-known/jwks.json",
    )

    payload = {
        "sub": "service",
        "scope": "ingest:write kg:read",
        "tenant_id": "tenant",
        "iss": "https://issuer",
        "aud": "medical-kg",
        "exp": int(time.time()) + 60,
    }
    token = jwt_encode(payload, private_key, kid="test")

    decoded = await authenticator.authenticate(token)
    assert decoded["sub"] == "service"
    assert dummy_client.calls == ["https://issuer/.well-known/jwks.json"]

    with pytest.raises(Exception):
        await authenticator.authenticate(token + "tamper")


@pytest.mark.anyio("asyncio")
async def test_secure_endpoint_enforces_scope_and_rate():
    dependency = secure_endpoint(scopes=["ingest:write"], endpoint="POST /test")
    rate_limiter = RateLimiter(RateLimitSettings(requests_per_minute=1, burst=1))
    context = SecurityContext(subject="user", tenant_id="tenant", scopes={"ingest:write"})

    result = await dependency(context=context, rate_limiter=rate_limiter)
    assert result.subject == "user"

    with pytest.raises(HTTPException) as exc:
        await dependency(context=context, rate_limiter=rate_limiter)
    assert exc.value.status_code == 429

    unauthorized_context = SecurityContext(subject="other", tenant_id="tenant", scopes=set())
    second_limiter = RateLimiter(RateLimitSettings(requests_per_minute=1, burst=1))
    with pytest.raises(HTTPException) as exc:
        await dependency(context=unauthorized_context, rate_limiter=second_limiter)
    assert exc.value.status_code == 403


def jwt_encode(payload: dict[str, Any], private_key, kid: str) -> str:
    from jose import jwt

    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return jwt.encode(payload, pem, algorithm="RS256", headers={"kid": kid})


def test_api_key_manager_generation_and_rotation():
    manager = APIKeyManager()
    created = manager.generate(tenant_id="tenant", scopes=["ingest:write"])
    assert created.raw_secret

    key_id, record = manager.authenticate(created.raw_secret)
    assert record.tenant_id == "tenant"

    rotated = manager.rotate(key_id)
    assert rotated.raw_secret != created.raw_secret
    key_id2, _ = manager.authenticate(rotated.raw_secret)
    assert key_id2 == key_id


def test_audit_trail_records_entries():
    audit = get_audit_trail()
    context = SecurityContext(subject="user", tenant_id="tenant", scopes={"*"})
    audit.record(context=context, action="test", resource="resource", metadata={"value": 1})
    entries = audit.list(tenant_id="tenant")
    assert entries
    assert entries[0].action == "test"


def test_build_api_key_manager_loads_secrets(monkeypatch):
    payload = {
        "keys": {
            "secret": {
                "hashed_secret": hashlib.sha256(b"secret").hexdigest(),
                "tenant_id": "tenant",
                "scopes": ["*"],
            }
        }
    }

    class DummyResolver:
        def __init__(self, *args, **kwargs):
            pass

        def get_secret(self, path: str):
            return payload

    monkeypatch.setattr("Medical_KG_rev.auth.api_keys.SecretResolver", DummyResolver)
    manager = build_api_key_manager()
    key_id, record = manager.authenticate("secret")
    assert key_id == "secret"
    assert record.scopes == ["*"]
