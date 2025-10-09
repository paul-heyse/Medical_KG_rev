"""Telemetry interfaces for embedding operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Mapping, MutableMapping

import structlog

try:
    from Medical_KG_rev.observability.metrics import (
        CROSS_TENANT_ACCESS_ATTEMPTS,
        observe_job_duration,
        record_business_event,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard

    class _CounterProxy:
        def labels(self, *args: object, **kwargs: object) -> "_CounterProxy":  # noqa: D401 - mimic prometheus API
            return self

        def inc(self, *args: object, **kwargs: object) -> None:
            return None

    CROSS_TENANT_ACCESS_ATTEMPTS = _CounterProxy()

    def observe_job_duration(operation: str, duration_seconds: float) -> None:  # type: ignore[override]
        return None

    def record_business_event(event: str, tenant_id: str) -> None:  # type: ignore[override]
        return None


if TYPE_CHECKING:
    from .persister import PersistenceReport
    from .policy import NamespaceAccessDecision

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class TelemetrySettings:
    """Runtime configuration for embedding telemetry."""

    enable_metrics: bool = True
    enable_logging: bool = True
    sample_rate: float = 1.0


@dataclass(slots=True)
class TelemetrySnapshot:
    """Diagnostic snapshot of telemetry state."""

    policy_evaluations: int = 0
    policy_denials: int = 0
    embedding_batches: int = 0
    embedding_failures: int = 0
    last_duration_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class EmbeddingTelemetry(ABC):
    """Abstract base class for telemetry providers."""

    def __init__(self, settings: TelemetrySettings | None = None) -> None:
        self._settings = settings or TelemetrySettings()
        self._snapshot = TelemetrySnapshot()
        self._logger = logger.bind(component=self.__class__.__name__)

    @property
    def settings(self) -> TelemetrySettings:
        return self._settings

    def update_settings(self, **kwargs: object) -> None:
        values = asdict(self._settings) | kwargs
        self._settings = TelemetrySettings(**values)

    def snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(**asdict(self._snapshot))

    def _record_duration(self, duration_ms: float) -> None:
        self._snapshot.last_duration_ms = duration_ms

    def record_policy_evaluation(self, decision: "NamespaceAccessDecision") -> None:
        self._snapshot.policy_evaluations += 1
        self._record_decision("evaluated", decision)

    def record_policy_denied(self, decision: "NamespaceAccessDecision") -> None:
        self._snapshot.policy_denials += 1
        self._record_decision("denied", decision)

    def record_embedding_started(
        self, *, namespace: str, tenant_id: str, model: str | None = None
    ) -> None:
        if self._settings.enable_logging:
            self._logger.info(
                "embedding.started",
                namespace=namespace,
                tenant_id=tenant_id,
                model=model,
            )

    def record_embedding_completed(
        self,
        *,
        namespace: str,
        tenant_id: str,
        model: str,
        provider: str | None,
        duration_ms: float,
        embeddings: int,
    ) -> None:
        self._snapshot.embedding_batches += 1
        self._record_duration(duration_ms)
        if self._settings.enable_metrics:
            observe_job_duration("embed", duration_ms / 1000)
            if embeddings:
                record_business_event("embeddings_generated", tenant_id)
        if self._settings.enable_logging:
            self._logger.info(
                "embedding.completed",
                namespace=namespace,
                tenant_id=tenant_id,
                model=model,
                provider=provider,
                duration_ms=duration_ms,
                embeddings=embeddings,
            )

    def record_embedding_failure(self, *, namespace: str, tenant_id: str, error: Exception) -> None:
        self._snapshot.embedding_failures += 1
        if self._settings.enable_logging:
            self._logger.error(
                "embedding.failed",
                namespace=namespace,
                tenant_id=tenant_id,
                error=str(error),
            )

    def record_persistence(
        self,
        report: "PersistenceReport",
        *,
        namespace: str,
        tenant_id: str,
    ) -> None:
        self._snapshot.metadata.setdefault("persistence", {})
        persistence = self._snapshot.metadata["persistence"]
        if isinstance(persistence, MutableMapping):
            persistence.setdefault(namespace, 0)
            persistence[namespace] += getattr(report, "persisted", 0)

    @abstractmethod
    def _record_decision(self, event: str, decision: "NamespaceAccessDecision") -> None:
        """Record a namespace policy decision."""


class StandardEmbeddingTelemetry(EmbeddingTelemetry):
    """Telemetry implementation backed by Prometheus metrics and structured logs."""

    def __init__(
        self,
        settings: TelemetrySettings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._denials_by_namespace: MutableMapping[str, int] = {}

    def _record_decision(self, event: str, decision) -> None:  # type: ignore[override]
        namespace = getattr(decision, "namespace", "unknown")
        tenant_id = getattr(decision, "tenant_id", "unknown")
        if event == "denied":
            self._denials_by_namespace[namespace] = self._denials_by_namespace.get(namespace, 0) + 1
            if getattr(decision, "denied_due_to_tenant", lambda: False)():
                allowed = getattr(decision, "metadata", {}).get("allowed_tenants", [])
                target = ",".join(sorted(allowed)) or "restricted"
                if self._settings.enable_metrics:
                    CROSS_TENANT_ACCESS_ATTEMPTS.labels(
                        source_tenant=tenant_id,
                        target_tenant=target,
                    ).inc()
            if self._settings.enable_logging:
                self._logger.warning(
                    "namespace.denied",
                    namespace=namespace,
                    tenant_id=tenant_id,
                    reason=getattr(decision, "reason", None),
                )
        else:
            if self._settings.enable_logging:
                self._logger.debug(
                    "namespace.evaluated",
                    namespace=namespace,
                    tenant_id=tenant_id,
                    allowed=getattr(decision, "allowed", None),
                )

    def operational_metrics(self) -> Mapping[str, object]:
        return {
            "denials_by_namespace": dict(self._denials_by_namespace),
            "snapshot": asdict(self.snapshot()),
        }


__all__ = [
    "EmbeddingTelemetry",
    "StandardEmbeddingTelemetry",
    "TelemetrySettings",
    "TelemetrySnapshot",
]
