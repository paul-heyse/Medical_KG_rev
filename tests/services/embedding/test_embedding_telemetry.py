from __future__ import annotations

from Medical_KG_rev.services.embedding.policy import NamespaceAccessDecision
from Medical_KG_rev.services.embedding.telemetry import StandardEmbeddingTelemetry, TelemetrySettings


def _decision(allowed: bool) -> NamespaceAccessDecision:
    return NamespaceAccessDecision(
        namespace="single_vector.test.3.v1",
        tenant_id="tenant-1",
        scope="embed:write",
        allowed=allowed,
        reason=None if allowed else "Tenant 'tenant-1' not allowed",
        config=None,
        metadata={"allowed_tenants": ["tenant-a"]},
    )


def test_standard_telemetry_tracks_policy_events() -> None:
    telemetry = StandardEmbeddingTelemetry(TelemetrySettings(enable_logging=False, enable_metrics=False))

    telemetry.record_policy_evaluation(_decision(True))
    telemetry.record_policy_denied(_decision(False))

    snapshot = telemetry.snapshot()
    assert snapshot.policy_evaluations == 1
    assert snapshot.policy_denials == 1
    metrics = telemetry.operational_metrics()
    assert metrics["denials_by_namespace"]["single_vector.test.3.v1"] == 1


def test_standard_telemetry_records_embedding_lifecycle() -> None:
    telemetry = StandardEmbeddingTelemetry(TelemetrySettings(enable_logging=False, enable_metrics=False))

    telemetry.record_embedding_started(namespace="single_vector.test.3.v1", tenant_id="tenant-1", model="test-model")
    telemetry.record_embedding_completed(
        namespace="single_vector.test.3.v1",
        tenant_id="tenant-1",
        model="test-model",
        provider="test-provider",
        duration_ms=42.0,
        embeddings=2,
    )

    snapshot = telemetry.snapshot()
    assert snapshot.embedding_batches == 1
    assert snapshot.last_duration_ms == 42.0

    telemetry.record_embedding_failure(namespace="single_vector.test.3.v1", tenant_id="tenant-1", error=RuntimeError("boom"))
    assert telemetry.snapshot().embedding_failures == 1
