from __future__ import annotations

import hashlib

import pytest

try:  # pragma: no cover - optional dependency for API tests
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional dependency or missing extras
    TestClient = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for non-chunking tests
    from Medical_KG_rev.config.settings import get_settings
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency
    get_settings = None  # type: ignore[assignment]

API_TEST_KEY = "test-api-key"


def pytest_addoption(parser):  # pragma: no cover - option wiring only
    group = parser.getgroup("cov")
    options = (
        ("--cov", {"action": "append", "default": [], "help": "Ignored test coverage option"}),
        (
            "--cov-report",
            {"action": "append", "default": [], "help": "Ignored coverage report option"},
        ),
    )
    for name, kwargs in options:
        try:
            group.addoption(name, **kwargs)
        except ValueError:
            continue
    parser.addini("asyncio_mode", "Asyncio mode stub", default="auto")


def pytest_configure(config):  # pragma: no cover - option wiring only
    config.addinivalue_line("markers", "asyncio: async tests")


@pytest.fixture(autouse=True)
def _configure_security(monkeypatch):
    hashed = hashlib.sha256(API_TEST_KEY.encode()).hexdigest()
    monkeypatch.setenv("MK_SECURITY__ENFORCE_HTTPS", "false")
    monkeypatch.setenv("MK_SECURITY__CORS__ALLOW_ORIGINS", '["http://testserver"]')
    monkeypatch.setenv("MK_SECURITY__API_KEYS__KEYS__default__hashed_secret", hashed)
    monkeypatch.setenv("MK_SECURITY__API_KEYS__KEYS__default__tenant_id", "tenant")
    monkeypatch.setenv("MK_SECURITY__API_KEYS__KEYS__default__scopes", '["*"]')
    monkeypatch.setenv("MK_SECURITY__OAUTH__ISSUER", "https://idp.local/realms/medical")
    monkeypatch.setenv("MK_SECURITY__OAUTH__AUDIENCE", "medical-kg")
    monkeypatch.setenv(
        "MK_SECURITY__OAUTH__TOKEN_URL",
        "https://idp.local/realms/medical/protocol/openid-connect/token",
    )
    monkeypatch.setenv(
        "MK_SECURITY__OAUTH__JWKS_URL",
        "https://idp.local/realms/medical/protocol/openid-connect/certs",
    )
    monkeypatch.setenv("MK_SECURITY__OAUTH__CLIENT_ID", "medical-gateway")
    monkeypatch.setenv("MK_SECURITY__OAUTH__CLIENT_SECRET", "dev-secret")
    monkeypatch.setenv(
        "MK_SECURITY__OAUTH__SCOPES",
        '["ingest:write", "kg:read", "jobs:read", "jobs:write", "process:write", "kg:write", "audit:read"]',
    )
    if get_settings is None:
        yield
        return

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def api_key() -> str:
    return API_TEST_KEY


@pytest.fixture(autouse=True)
def _inject_api_key_header(monkeypatch, api_key: str):
    if TestClient is None:
        return

    original_request = TestClient.request

    def _request(self, method, url, *args, **kwargs):  # type: ignore[override]
        headers = kwargs.get("headers")
        if headers is None:
            headers = {}
            kwargs["headers"] = headers
        headers.setdefault("X-API-Key", api_key)
        return original_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(TestClient, "request", _request)


@pytest.fixture
def anyio_backend():
    return "asyncio"
