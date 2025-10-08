from __future__ import annotations

import asyncio
from pathlib import Path
import textwrap

import httpx
import pytest
import respx
from pybreaker import CircuitBreakerError

from Medical_KG_rev.observability import metrics as metrics_module
from Medical_KG_rev.orchestration.dagster.configuration import (
    ResiliencePolicy,
    ResiliencePolicyLoader,
)

pytestmark = pytest.mark.filterwarnings(
    r"ignore:datetime.datetime.utcnow\(\) is deprecated:DeprecationWarning"
)


def _write_policy(tmp_path: Path, payload: str) -> Path:
    path = tmp_path / "resilience.yaml"
    path.write_text(textwrap.dedent(payload).strip() + "\n")
    return path


def test_resilience_policy_retries_http_requests(tmp_path: Path) -> None:
    policy_path = _write_policy(
        tmp_path,
        """
        policies:
          http-api:
            max_attempts: 3
            timeout_seconds: 5
            backoff:
              strategy: none
              initial: 0.0
              maximum: 0.0
              jitter: false
        """,
    )

    loader = ResiliencePolicyLoader(policy_path)
    loader.load()

    url = "https://example.test/resource"
    router = respx.Router(assert_all_called=True)
    route = router.get(url).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json={"status": "ok"}),
        ]
    )

    client = httpx.Client(transport=httpx.MockTransport(router.handler))
    try:

        def fetch_status() -> str:
            response = client.get(url, timeout=1.0)
            response.raise_for_status()
            return response.json()["status"]

        wrapped = loader.apply("http-api", "ingest", fetch_status)
        assert wrapped() == "ok"
        assert route.call_count == 2
        router.assert_all_called()
    finally:
        client.close()


def test_resilience_policy_opens_circuit_breaker(tmp_path: Path) -> None:
    policy_path = _write_policy(
        tmp_path,
        """
        policies:
          breaker:
            max_attempts: 1
            timeout_seconds: 5
            backoff:
              strategy: none
              initial: 0.0
              maximum: 0.0
              jitter: false
            circuit_breaker:
              failure_threshold: 3
              recovery_timeout: 60.0
        """,
    )

    loader = ResiliencePolicyLoader(policy_path)
    loader.load()

    calls = {"count": 0}

    def always_fail() -> None:
        calls["count"] += 1
        raise RuntimeError("boom")

    wrapped = loader.apply("breaker", "embed", always_fail)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            wrapped()

    assert calls["count"] == 3

    with pytest.raises(CircuitBreakerError):
        wrapped()


def test_resilience_policy_rate_limiter_records_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path = _write_policy(
        tmp_path,
        """
        policies:
          limited:
            max_attempts: 1
            timeout_seconds: 5
            backoff:
              strategy: none
              initial: 0.0
              maximum: 0.0
              jitter: false
            rate_limit:
              rate_limit_per_second: 1.0
        """,
    )

    loader = ResiliencePolicyLoader(policy_path)
    loader.load()

    waits: list[float] = []

    monkeypatch.setattr(
        metrics_module,
        "record_resilience_rate_limit_wait",
        lambda policy, stage, wait: waits.append(wait),
    )

    class FakeLimiter:
        def __init__(self) -> None:
            self.calls = 0

        async def acquire(self) -> None:
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(0)
                return
            await asyncio.sleep(0.02)

    fake_limiter = FakeLimiter()

    monkeypatch.setattr(
        ResiliencePolicy,
        "create_rate_limiter",
        lambda self: fake_limiter,
    )

    wrapped = loader.apply("limited", "index", lambda: "done")

    assert wrapped() == "done"
    assert wrapped() == "done"

    assert fake_limiter.calls == 2
    assert waits and waits[-1] > 0.0
