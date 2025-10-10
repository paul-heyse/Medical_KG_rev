from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("yaml")

from Medical_KG_rev.auth.scopes import Scopes
from Medical_KG_rev.gateway.models import NamespacePolicyUpdateRequest
from Medical_KG_rev.gateway.services import GatewayService
from Medical_KG_rev.gateway.sse.manager import EventStreamManager
from Medical_KG_rev.orchestration import JobLedger
from Medical_KG_rev.services.embedding.namespace.schema import EmbeddingKind, NamespaceConfig


@pytest.fixture()
def service() -> GatewayService:
    gateway = GatewayService(
        events=EventStreamManager(),
        orchestrator=SimpleNamespace(),
        ledger=JobLedger(),
    )
    return gateway


def test_namespace_policy_status_includes_metrics(service: GatewayService) -> None:
    status = service.namespace_policy_status()
    assert status.settings.cache_ttl_seconds >= 0
    assert status.stats["evaluations"] == 0
    assert "cache_hit_ratio" in status.operational


def test_namespace_policy_dry_run_toggle(service: GatewayService) -> None:
    registry = service.namespace_registry
    assert registry is not None
    registry.register(
        "single_vector.restricted.8.v1",
        NamespaceConfig(
            name="restricted",
            kind=EmbeddingKind.SINGLE_VECTOR,
            model_id="demo",
            provider="test",
            dim=8,
            max_tokens=128,
            allowed_tenants=["tenant-allowed"],
            allowed_scopes=[Scopes.EMBED_READ],
        ),
    )
    policy = service.namespace_policy
    assert policy is not None
    denied = policy.evaluate(
        namespace="single_vector.restricted.8.v1",
        tenant_id="tenant-denied",
        required_scope=Scopes.EMBED_READ,
    )
    assert not denied.allowed

    updated = service.update_namespace_policy(NamespacePolicyUpdateRequest(dry_run=True))
    assert updated.settings.dry_run is True

    decision = service.namespace_policy.evaluate(
        namespace="single_vector.restricted.8.v1",
        tenant_id="tenant-denied",
        required_scope=Scopes.EMBED_READ,
    )
    assert decision.allowed
    assert decision.metadata.get("dry_run_denied") is True


def test_namespace_policy_diagnostics_and_invalidation(service: GatewayService) -> None:
    policy = service.namespace_policy
    assert policy is not None
    policy.evaluate(
        namespace="single_vector.qwen3.4096.v1",
        tenant_id="tenant-a",
        required_scope=Scopes.EMBED_READ,
    )
    diagnostics = service.namespace_policy_diagnostics()
    assert diagnostics.cache_keys

    service.invalidate_namespace_policy_cache()
    diagnostics_after = service.namespace_policy_diagnostics()
    assert not diagnostics_after.cache_keys
    metrics = service.namespace_policy_metrics()
    assert metrics.metrics["policy"] == policy.__class__.__name__
    health = service.namespace_policy_health()
    assert health.policy == policy.__class__.__name__
