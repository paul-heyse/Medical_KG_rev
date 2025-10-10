from __future__ import annotations

import pytest

from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from Medical_KG_rev.services.embedding.namespace.schema import EmbeddingKind, NamespaceConfig
from Medical_KG_rev.services.embedding.policy import (
    CustomNamespacePolicy,
    DryRunNamespacePolicy,
    MockNamespacePolicy,
    NamespaceAccessDecision,
    NamespacePolicySettings,
    StandardNamespacePolicy,
    build_policy_chain,
)
from Medical_KG_rev.services.embedding.telemetry import (
    StandardEmbeddingTelemetry,
    TelemetrySettings,
)


@pytest.fixture()
def registry() -> EmbeddingNamespaceRegistry:
    registry = EmbeddingNamespaceRegistry()
    registry.register(
        "single_vector.test.3.v1",
        NamespaceConfig(
            name="test",
            kind=EmbeddingKind.SINGLE_VECTOR,
            model_id="test-model",
            provider="test",
            dim=3,
            max_tokens=128,
            allowed_tenants=["tenant-a"],
            allowed_scopes=["embed:write", "embed:read"],
        ),
    )
    return registry


def test_standard_policy_allows_authorised_tenant(registry: EmbeddingNamespaceRegistry) -> None:
    telemetry = StandardEmbeddingTelemetry(
        TelemetrySettings(enable_logging=False, enable_metrics=False)
    )
    policy = StandardNamespacePolicy(
        registry, telemetry=telemetry, settings=NamespacePolicySettings(cache_ttl_seconds=10)
    )

    decision = policy.evaluate(
        namespace="single_vector.test.3.v1", tenant_id="tenant-a", required_scope="embed:write"
    )

    assert decision.allowed
    assert decision.config is not None
    assert telemetry.snapshot().policy_evaluations == 1

    # Cached evaluation should be counted as a cache hit.
    policy.evaluate(
        namespace="single_vector.test.3.v1", tenant_id="tenant-a", required_scope="embed:write"
    )
    assert policy.stats()["cache_hits"] == 1


def test_standard_policy_denies_cross_tenant(registry: EmbeddingNamespaceRegistry) -> None:
    telemetry = StandardEmbeddingTelemetry(
        TelemetrySettings(enable_logging=False, enable_metrics=False)
    )
    policy = StandardNamespacePolicy(registry, telemetry=telemetry)

    decision = policy.evaluate(
        namespace="single_vector.test.3.v1", tenant_id="tenant-b", required_scope="embed:write"
    )

    assert not decision.allowed
    assert decision.denied_due_to_tenant()
    assert telemetry.snapshot().policy_denials == 1


def test_dry_run_policy_marks_denials(registry: EmbeddingNamespaceRegistry) -> None:
    policy = StandardNamespacePolicy(registry)
    dry_run = DryRunNamespacePolicy(policy)

    decision = dry_run.evaluate(
        namespace="single_vector.test.3.v1", tenant_id="tenant-b", required_scope="embed:write"
    )

    assert decision.allowed
    assert decision.metadata["dry_run_denied"] is True


def test_mock_policy_allows_registered_decision(registry: EmbeddingNamespaceRegistry) -> None:
    mock = MockNamespacePolicy(registry)
    mock.register_decision(
        namespace="single_vector.test.3.v1", tenant_id="tenant-x", scope="embed:read", allowed=True
    )

    decision = mock.evaluate(
        namespace="single_vector.test.3.v1", tenant_id="tenant-x", required_scope="embed:read"
    )

    assert decision.allowed


@pytest.mark.parametrize("allowed", [True, False])
def test_custom_policy_delegates(registry: EmbeddingNamespaceRegistry, allowed: bool) -> None:
    def resolver(namespace: str, tenant_id: str, scope: str) -> NamespaceAccessDecision:
        return NamespaceAccessDecision(
            namespace=namespace,
            tenant_id=tenant_id,
            scope=scope,
            allowed=allowed,
            reason=None if allowed else "Denied",
            config=None,
        )

    policy = CustomNamespacePolicy(registry, resolver)
    decision = policy.evaluate(
        namespace="single_vector.test.3.v1", tenant_id="tenant-a", required_scope="embed:write"
    )
    assert decision.allowed is allowed


def test_policy_update_settings_clears_cache(registry: EmbeddingNamespaceRegistry) -> None:
    policy = StandardNamespacePolicy(registry)
    policy.evaluate(
        namespace="single_vector.test.3.v1", tenant_id="tenant-a", required_scope="embed:write"
    )
    snapshot = policy.debug_snapshot()
    assert snapshot["cache_keys"], "expected cached decisions"

    policy.update_settings(cache_ttl_seconds=0)
    assert not policy.debug_snapshot()["cache_keys"]


def test_build_policy_chain_respects_dry_run(registry: EmbeddingNamespaceRegistry) -> None:
    settings = NamespacePolicySettings(cache_ttl_seconds=5, dry_run=True)
    policy = build_policy_chain(registry, settings=settings, dry_run=settings.dry_run)

    assert isinstance(policy, DryRunNamespacePolicy)
