from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import httpx
from pydantic import Field, SecretStr

from Medical_KG_rev.adapters import (
    AdapterDomain,
    AdapterMetadata,
    AdapterPluginError,
    AdapterPluginManager,
    AdapterRequest,
    AdapterSettings,
    BackoffStrategy,
    ConfigValidationResult,
    FinancialNewsAdapterPlugin,
    LegalPrecedentAdapterPlugin,
    ResilienceConfig,
    ResilientHTTPClient,
    SettingsHotReloader,
    VaultSecretProvider,
    apply_env_overrides,
    circuit_breaker,
    get_plugin_manager,
    list_adapters_by_domain,
    migrate_yaml_to_env,
    rate_limit,
    register_biomedical_plugins,
    retry_on_failure,
    validate_on_startup,
    ValidationOutcome,
)
from Medical_KG_rev.adapters.plugins import bootstrap as plugin_bootstrap
from Medical_KG_rev.adapters.plugins.example import ExampleAdapterPlugin
from pydantic import ValidationError


def test_rate_limit_decorator_controls_throughput():
    config = ResilienceConfig(rate_limit_per_second=1, rate_limit_capacity=1)

    timestamps: list[float] = []

    @rate_limit(config)
    async def limited_call() -> None:
        timestamps.append(time.monotonic())

    async def _run() -> None:
        await asyncio.gather(limited_call(), limited_call())

    asyncio.run(_run())
    assert timestamps[1] - timestamps[0] >= 0.9


def test_circuit_breaker_opens_after_failures():
    config = ResilienceConfig(
        circuit_breaker_failure_threshold=1,
        circuit_breaker_reset_timeout=100,
    )

    @circuit_breaker(config)
    async def failing_call() -> None:
        raise RuntimeError("boom")

    async def _run() -> None:
        with pytest.raises(RuntimeError):
            await failing_call()

        with pytest.raises(CircuitBreakerError):
            await failing_call()

    asyncio.run(_run())


def test_retry_on_failure_retries_until_success():
    config = ResilienceConfig(max_attempts=3, backoff_strategy=BackoffStrategy.LINEAR, backoff_multiplier=0.01)
    attempts: list[int] = []

    @retry_on_failure(config, retry_exceptions=(RuntimeError,))
    def flaky_call() -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("try again")
        return "ok"

    assert flaky_call() == "ok"
    assert len(attempts) == 2


def test_resilient_http_client_uses_retry():
    responses: list[str] = []

    class DummyResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            responses.append("raised")

    class DummyClient:
        async def get(self, url: str, **kwargs):
            return DummyResponse()

        async def aclose(self) -> None:
            responses.append("closed")

    async def _run() -> None:
        client = ResilientHTTPClient(client=DummyClient())
        response = await client.get("https://example.com")
        assert response.status_code == 200
        assert "raised" in responses
        await client.aclose()
        assert "closed" in responses

    asyncio.run(_run())


def test_plugin_manager_registers_and_runs_adapter():
    manager = AdapterPluginManager()
    plugin = ExampleAdapterPlugin()
    metadata = manager.register(plugin)
    assert metadata.name == "example"

    request = AdapterRequest(
        tenant_id="tenant-1",
        correlation_id="corr-1",
        domain=AdapterDomain.BIOMEDICAL,
    )
    result = manager.invoke("example", request)
    assert result.ok is True
    assert result.error is None
    assert {stage.name for stage in result.metrics.stages} == {"fetch", "parse", "validate"}
    assert result.response is not None
    assert result.response.items[0]["tenant"] == "tenant-1"
    assert result.response.metadata["adapter"] == "example"
    response = manager.run("example", request)
    assert manager.check_health("example") is True
    assert manager.list_metadata(domain=AdapterDomain.BIOMEDICAL)
    estimate = manager.estimate_cost("example", request)
    assert estimate.estimated_requests >= 1


def test_plugin_manager_execute_returns_state_and_respects_strict():
    manager = AdapterPluginManager()

    class InvalidAdapter(ExampleAdapterPlugin):
        metadata = AdapterMetadata(
            name="invalid-example",
            version="1.0.0",
            domain=AdapterDomain.BIOMEDICAL,
            summary="Invalid adapter for testing",
        )

        def validate(self, response, request):
            return ValidationOutcome(valid=False, errors=["boom"])

    manager.register(InvalidAdapter())
    request = AdapterRequest(
        tenant_id="tenant-1",
        correlation_id="corr-1",
        domain=AdapterDomain.BIOMEDICAL,
    )

    state = manager.execute("invalid-example", request, strict=False)
    assert state.validation is not None and state.validation.valid is False
    assert state.response is not None

    result = manager.invoke("invalid-example", request, strict=True)
    assert result.error is not None
    assert result.ok is False
    assert any(stage.name == "validate" for stage in result.metrics.stages)

    tolerant = manager.invoke("invalid-example", request, strict=False)
    assert tolerant.error is None
    assert tolerant.validation is not None and tolerant.validation.valid is False

    with pytest.raises(AdapterPluginError):
        manager.execute("invalid-example", request)


def test_adapter_pipeline_telemetry_records_stage_timings():
    manager = AdapterPluginManager()
    manager.register(ExampleAdapterPlugin())
    request = AdapterRequest(
        tenant_id="tenant", correlation_id="corr", domain=AdapterDomain.BIOMEDICAL
    )

    result = manager.invoke("example", request)

    assert result.metrics.duration_ms >= 0
    assert all(stage.duration_ms >= 0 for stage in result.metrics.stages)
    assert result.pipeline["name"] == "default"
    assert result.pipeline["stages"] == ("fetch", "parse", "validate")


def test_register_biomedical_plugins_groups_by_domain():
    manager = AdapterPluginManager()
    metadata = register_biomedical_plugins(manager)
    manager.register(FinancialNewsAdapterPlugin())
    manager.register(LegalPrecedentAdapterPlugin())
    mapping = manager.domains()
    assert AdapterDomain.BIOMEDICAL in mapping
    biomedical_names = mapping[AdapterDomain.BIOMEDICAL]
    assert all(meta.name in biomedical_names for meta in metadata)
    assert AdapterDomain.FINANCIAL in mapping and "financial-news" in mapping[AdapterDomain.FINANCIAL]
    assert AdapterDomain.LEGAL in mapping and "legal-precedent" in mapping[AdapterDomain.LEGAL]



def test_get_plugin_manager_lists_domains(monkeypatch):
    manager = get_plugin_manager(refresh=True)
    assert manager.list_metadata()
    domains = list_adapters_by_domain()
    assert AdapterDomain.BIOMEDICAL in domains
    assert isinstance(domains[AdapterDomain.BIOMEDICAL], tuple)


def test_plugin_discovery_performance():
    start = time.perf_counter()
    manager = AdapterPluginManager()
    register_biomedical_plugins(manager)
    duration = time.perf_counter() - start
    assert duration < 1.0


def test_plugin_framework_feature_flag(monkeypatch):
    monkeypatch.setenv("MK_USE_PLUGIN_FRAMEWORK", "0")
    import importlib

    importlib.reload(plugin_bootstrap)
    with pytest.raises(RuntimeError):
        plugin_bootstrap.get_plugin_manager()
    monkeypatch.setenv("MK_USE_PLUGIN_FRAMEWORK", "1")
    importlib.reload(plugin_bootstrap)


def test_adapter_request_requires_identifiers():
    with pytest.raises(ValidationError):
        AdapterRequest(
            tenant_id="",
            correlation_id="corr",
            domain=AdapterDomain.BIOMEDICAL,
        )


def test_adapter_settings_respect_env_precedence(monkeypatch):
    monkeypatch.setenv("MK_ADAPTER_TIMEOUT_SECONDS", "120")
    settings = AdapterSettings()
    assert settings.timeout_seconds == 120


def test_migrate_yaml_to_env(tmp_path: Path):
    yaml_path = tmp_path / "adapter.yaml"
    yaml_path.write_text("timeout_seconds: 45\n", encoding="utf-8")
    env_mapping = migrate_yaml_to_env(yaml_path)
    assert env_mapping["MK_ADAPTER_TIMEOUT_SECONDS"] == "45"


def test_apply_env_overrides_sets_environment(monkeypatch):
    apply_env_overrides({"MK_ADAPTER_SAMPLE": "value"})
    assert os.environ["MK_ADAPTER_SAMPLE"] == "value"


def test_validate_on_startup_reports_errors():
    class StrictSettings(AdapterSettings):
        timeout_seconds: int = Field(0, ge=60)

    result = validate_on_startup(StrictSettings)
    assert isinstance(result, ConfigValidationResult)
    assert result.valid is False
    assert result.errors


def test_hot_reloader_updates_configuration():
    values = {"timeout": 30}

    def factory() -> AdapterSettings:
        return AdapterSettings(
            timeout_seconds=values["timeout"],
            rate_limit_per_second=1.0,
            retry_max_attempts=1,
        )

    reloader = SettingsHotReloader(factory, interval_seconds=0.1)
    reloader.start()
    try:
        time.sleep(0.15)
        values["timeout"] = 60
        time.sleep(0.2)
        assert reloader.current().timeout_seconds == 60
    finally:
        reloader.stop()


def test_vault_secret_provider_caches_results():
    provider = VaultSecretProvider("https://vault.example.com", SecretStr("token"))

    calls: list[str] = []

    class DummyKV:
        def __init__(self) -> None:
            self.v2 = SimpleNamespace(read_secret_version=self.read)

        def read(self, path: str, mount_point: str):
            calls.append(path)
            return {"data": {"data": {"api_key": "secret"}}}

    provider._client = SimpleNamespace(secrets=SimpleNamespace(kv=DummyKV()))  # type: ignore[attr-defined]
    secret = provider.get("secret/path")
    assert secret["api_key"] == "secret"
    # Second call should hit cache
    secret = provider.get("secret/path")
    assert secret["api_key"] == "secret"
    assert calls == ["secret/path"]

