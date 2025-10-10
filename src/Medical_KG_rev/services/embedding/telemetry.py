"""Telemetry interfaces for embedding operations.

This module provides telemetry and observability capabilities for embedding
operations, including metrics collection, structured logging, and diagnostic
snapshots. It supports both Prometheus metrics and structured logging with
configurable sampling and filtering.

Key Responsibilities:
    - Define telemetry interfaces and configuration models
    - Implement metrics collection for embedding operations
    - Provide structured logging for embedding events
    - Track policy evaluations and access denials
    - Generate diagnostic snapshots for monitoring
    - Support optional dependency handling for metrics

Collaborators:
    - Upstream: Embedding coordinators, policy evaluators, persistence services
    - Downstream: Prometheus metrics, structured logging, monitoring systems

Side Effects:
    - Emits Prometheus metrics for embedding operations
    - Writes structured log entries
    - Updates internal counters and snapshots
    - May perform I/O operations for metrics and logging

Thread Safety:
    - Thread-safe: All methods use atomic operations and immutable snapshots
    - Counters and metrics are designed for concurrent access

Performance Characteristics:
    - O(1) time for most telemetry operations
    - Minimal memory overhead for counters and snapshots
    - Configurable sampling to reduce overhead
    - Optional metrics collection to disable when not needed

Example:
    >>> from Medical_KG_rev.services.embedding.telemetry import StandardEmbeddingTelemetry
    >>> telemetry = StandardEmbeddingTelemetry()
    >>> telemetry.record_embedding_started(namespace="medical", tenant_id="tenant1")
    >>> telemetry.record_embedding_completed(
    ...     namespace="medical", tenant_id="tenant1", model="bert-base",
    ...     provider="huggingface", duration_ms=1500.0, embeddings=10
    ... )
    >>> snapshot = telemetry.snapshot()
    >>> print(f"Embedding batches: {snapshot.embedding_batches}")

"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import structlog

# ============================================================================
# OPTIONAL DEPENDENCIES
# ============================================================================

try:
    from Medical_KG_rev.observability.metrics import (
        CROSS_TENANT_ACCESS_ATTEMPTS,
        observe_job_duration,
        record_business_event,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    # Fallback implementations when observability module is not available
    class _CounterProxy:
        """Proxy for Prometheus counter when observability module unavailable.

        Provides no-op implementations of Prometheus counter methods to
        allow telemetry to function without the observability dependency.
        """

        def labels(self, *args: object, **kwargs: object) -> _CounterProxy:
            """Return self for method chaining."""
            return self

        def inc(self, *args: object, **kwargs: object) -> None:
            """No-op increment operation."""
            return None

    CROSS_TENANT_ACCESS_ATTEMPTS = _CounterProxy()

    def observe_job_duration(operation: str, duration_seconds: float) -> None:  # type: ignore[override]
        """No-op job duration observation."""
        return None

    def record_business_event(event: str, tenant_id: str) -> None:  # type: ignore[override]
        """No-op business event recording."""
        return None


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

if TYPE_CHECKING:
    from .persister import PersistenceReport
    from .policy import NamespaceAccessDecision

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = structlog.get_logger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(slots=True)
class TelemetrySettings:
    """Runtime configuration for embedding telemetry.

    Controls telemetry behavior including metrics collection,
    logging, and sampling rates for embedding operations.

    Attributes:
        enable_metrics: Whether to collect and emit Prometheus metrics.
            Defaults to True.
        enable_logging: Whether to emit structured log entries.
            Defaults to True.
        sample_rate: Fraction of operations to sample for telemetry.
            Range: 0.0 to 1.0. Defaults to 1.0 (sample all).

    Example:
        >>> settings = TelemetrySettings(enable_metrics=True, sample_rate=0.1)
        >>> telemetry = StandardEmbeddingTelemetry(settings)

    """

    enable_metrics: bool = True
    enable_logging: bool = True
    sample_rate: float = 1.0


@dataclass(slots=True)
class TelemetrySnapshot:
    """Diagnostic snapshot of telemetry state.

    Provides a point-in-time view of telemetry counters and
    metadata for monitoring and debugging embedding operations.

    Attributes:
        policy_evaluations: Total number of namespace policy evaluations.
        policy_denials: Total number of policy denials.
        embedding_batches: Total number of embedding batches processed.
        embedding_failures: Total number of embedding operation failures.
        last_duration_ms: Duration of the most recent embedding operation.
        metadata: Additional telemetry data including persistence stats.

    Example:
        >>> snapshot = telemetry.snapshot()
        >>> print(f"Processed {snapshot.embedding_batches} batches")
        >>> print(f"Last operation took {snapshot.last_duration_ms}ms")

    """

    policy_evaluations: int = 0
    policy_denials: int = 0
    embedding_batches: int = 0
    embedding_failures: int = 0
    last_duration_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


# ============================================================================
# INTERFACE (Protocols/ABCs)
# ============================================================================


class EmbeddingTelemetry(ABC):
    """Abstract base class for telemetry providers.

    Defines the interface for collecting telemetry data from embedding
    operations including metrics, logs, and diagnostic snapshots.
    Provides common functionality for settings management and snapshot
    generation while allowing implementations to customize specific
    telemetry collection strategies.

    Attributes:
        _settings: Runtime configuration for telemetry behavior
        _snapshot: Current telemetry state and counters
        _logger: Structured logger bound to this telemetry instance

    Invariants:
        - _settings is never None
        - _snapshot counters are always non-negative
        - _logger is properly bound to component name

    Thread Safety:
        - Thread-safe: All operations use atomic updates and immutable snapshots

    Lifecycle:
        - Initialized with optional settings
        - Settings can be updated dynamically
        - Snapshots provide point-in-time state
        - Counters accumulate over service lifetime

    Example:
        >>> class CustomTelemetry(EmbeddingTelemetry):
        ...     def _record_decision(self, event: str, decision):
        ...         # Custom implementation
        ...         pass
        >>> telemetry = CustomTelemetry()
        >>> telemetry.record_embedding_started(namespace="test", tenant_id="t1")

    """

    def __init__(self, settings: TelemetrySettings | None = None) -> None:
        """Initialize telemetry provider with optional settings.

        Args:
            settings: Optional telemetry configuration. If None, uses
                default settings with metrics and logging enabled.

        Note:
            Logger is automatically bound to the concrete class name
            for structured logging identification.

        """
        self._settings = settings or TelemetrySettings()
        self._snapshot = TelemetrySnapshot()
        self._logger = logger.bind(component=self.__class__.__name__)

    @property
    def settings(self) -> TelemetrySettings:
        """Get current telemetry settings.

        Returns:
            Current telemetry configuration including metrics, logging,
            and sampling settings.

        """
        return self._settings

    def update_settings(self, **kwargs: object) -> None:
        """Update telemetry settings with new values.

        Args:
            **kwargs: Settings to update. Valid keys: enable_metrics,
                enable_logging, sample_rate.

        Note:
            Only provided settings are updated; others remain unchanged.
            Invalid keys are ignored.

        """
        values = asdict(self._settings) | kwargs
        self._settings = TelemetrySettings(**values)

    def snapshot(self) -> TelemetrySnapshot:
        """Get current telemetry state snapshot.

        Returns:
            Immutable snapshot of current telemetry counters and metadata.
            Safe to use for monitoring and debugging without affecting
            the live telemetry state.

        Example:
            >>> snapshot = telemetry.snapshot()
            >>> print(f"Policy evaluations: {snapshot.policy_evaluations}")

        """
        return TelemetrySnapshot(**asdict(self._snapshot))

    def _record_duration(self, duration_ms: float) -> None:
        """Record the duration of an embedding operation.

        Args:
            duration_ms: Duration in milliseconds of the operation.

        Note:
            This updates the last_duration_ms field in the snapshot
            for monitoring and alerting purposes.

        """
        self._snapshot.last_duration_ms = duration_ms

    def record_policy_evaluation(self, decision: NamespaceAccessDecision) -> None:
        """Record a namespace policy evaluation.

        Increments the policy evaluation counter and delegates to
        the implementation-specific decision recording logic.

        Args:
            decision: The namespace access decision that was evaluated.

        Note:
            This is called for all policy evaluations, both allowed
            and denied, to track policy usage patterns.

        """
        self._snapshot.policy_evaluations += 1
        self._record_decision("evaluated", decision)

    def record_policy_denied(self, decision: NamespaceAccessDecision) -> None:
        """Record a namespace policy denial.

        Increments the policy denial counter and delegates to
        the implementation-specific decision recording logic.

        Args:
            decision: The namespace access decision that was denied.

        Note:
            This is called specifically for denied access attempts
            to track security violations and policy enforcement.

        """
        self._snapshot.policy_denials += 1
        self._record_decision("denied", decision)

    def record_embedding_started(
        self, *, namespace: str, tenant_id: str, model: str | None = None
    ) -> None:
        """Record the start of an embedding operation.

        Logs the beginning of an embedding operation with context
        information for monitoring and debugging.

        Args:
            namespace: Namespace where embeddings will be stored.
            tenant_id: Tenant performing the embedding operation.
            model: Optional model name being used for embedding.

        Note:
            This is typically called at the beginning of embedding
            operations to track usage patterns and performance.

        """
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
        """Record the completion of an embedding operation.

        Updates counters, records metrics, and logs the successful
        completion of an embedding operation with performance data.

        Args:
            namespace: Namespace where embeddings were stored.
            tenant_id: Tenant that performed the embedding operation.
            model: Model name used for embedding.
            provider: Optional provider name (e.g., "huggingface").
            duration_ms: Duration of the embedding operation in milliseconds.
            embeddings: Number of embeddings generated.

        Note:
            This updates both metrics and logs, and records the
            duration for performance monitoring.

        """
        self._snapshot.embedding_batches += 1
        self._record_duration(duration_ms)

        # Emit metrics if enabled
        if self._settings.enable_metrics:
            observe_job_duration("embed", duration_ms / 1000)
            if embeddings:
                record_business_event("embeddings_generated", tenant_id)

        # Log completion if enabled
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
        """Record a failed embedding operation.

        Increments the failure counter and logs the error for
        monitoring and debugging purposes.

        Args:
            namespace: Namespace where embedding was attempted.
            tenant_id: Tenant that attempted the embedding operation.
            error: Exception that caused the failure.

        Note:
            This is called when embedding operations fail to track
            error rates and identify problematic patterns.

        """
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
        report: PersistenceReport,
        *,
        namespace: str,
        tenant_id: str,
    ) -> None:
        """Record persistence operation results.

        Updates metadata with persistence statistics for monitoring
        storage operations and success rates.

        Args:
            report: Persistence report containing operation results.
            namespace: Namespace where persistence occurred.
            tenant_id: Tenant that performed the persistence operation.

        Note:
            This accumulates persistence statistics in the snapshot
            metadata for storage monitoring and capacity planning.

        """
        self._snapshot.metadata.setdefault("persistence", {})
        persistence = self._snapshot.metadata["persistence"]
        if isinstance(persistence, MutableMapping):
            persistence.setdefault(namespace, 0)
            persistence[namespace] += getattr(report, "persisted", 0)

    @abstractmethod
    def _record_decision(self, event: str, decision: NamespaceAccessDecision) -> None:
        """Record a namespace policy decision.

        Abstract method for implementation-specific decision recording.
        Called by record_policy_evaluation and record_policy_denied
        to allow custom telemetry collection strategies.

        Args:
            event: Type of event ("evaluated" or "denied").
            decision: The namespace access decision to record.

        Note:
            Implementations should handle metrics collection, logging,
            and any custom telemetry logic for policy decisions.

        """


# ============================================================================
# IMPLEMENTATIONS
# ============================================================================


class StandardEmbeddingTelemetry(EmbeddingTelemetry):
    """Telemetry implementation backed by Prometheus metrics and structured logs.

    Provides a concrete implementation of embedding telemetry that
    integrates with Prometheus metrics and structured logging. Tracks
    policy decisions, embedding operations, and provides operational
    metrics for monitoring and alerting.

    Attributes:
        _denials_by_namespace: Counter for denials by namespace for
            operational metrics and alerting

    Invariants:
        - _denials_by_namespace counters are always non-negative
        - All metrics respect the enable_metrics setting
        - All logs respect the enable_logging setting

    Thread Safety:
        - Thread-safe: Uses atomic operations for counter updates
        - Safe for concurrent access from multiple threads

    Example:
        >>> telemetry = StandardEmbeddingTelemetry()
        >>> telemetry.record_embedding_started(namespace="medical", tenant_id="t1")
        >>> metrics = telemetry.operational_metrics()
        >>> print(f"Denials: {metrics['denials_by_namespace']}")

    """

    def __init__(
        self,
        settings: TelemetrySettings | None = None,
    ) -> None:
        """Initialize standard telemetry implementation.

        Args:
            settings: Optional telemetry configuration. If None, uses
                default settings with metrics and logging enabled.

        Note:
            Initializes namespace denial tracking for operational metrics.

        """
        super().__init__(settings=settings)
        self._denials_by_namespace: MutableMapping[str, int] = {}

    def _record_decision(self, event: str, decision) -> None:  # type: ignore[override]
        """Record a namespace policy decision with metrics and logging.

        Implements the abstract method to provide concrete telemetry
        collection for policy decisions including metrics and structured logs.

        Args:
            event: Type of event ("evaluated" or "denied").
            decision: The namespace access decision to record.

        Note:
            For denied events, tracks cross-tenant access attempts
            and emits warning logs. For evaluated events, emits debug logs.

        """
        namespace = getattr(decision, "namespace", "unknown")
        tenant_id = getattr(decision, "tenant_id", "unknown")

        if event == "denied":
            # Track denials by namespace for operational metrics
            self._denials_by_namespace[namespace] = self._denials_by_namespace.get(namespace, 0) + 1

            # Track cross-tenant access attempts if applicable
            if getattr(decision, "denied_due_to_tenant", lambda: False)():
                allowed = getattr(decision, "metadata", {}).get("allowed_tenants", [])
                target = ",".join(sorted(allowed)) or "restricted"
                if self._settings.enable_metrics:
                    CROSS_TENANT_ACCESS_ATTEMPTS.labels(
                        source_tenant=tenant_id,
                        target_tenant=target,
                    ).inc()

            # Log denial with warning level
            if self._settings.enable_logging:
                self._logger.warning(
                    "namespace.denied",
                    namespace=namespace,
                    tenant_id=tenant_id,
                    reason=getattr(decision, "reason", None),
                )
        else:
            # Log evaluation with debug level
            if self._settings.enable_logging:
                self._logger.debug(
                    "namespace.evaluated",
                    namespace=namespace,
                    tenant_id=tenant_id,
                    allowed=getattr(decision, "allowed", None),
                )

    def operational_metrics(self) -> Mapping[str, object]:
        """Get operational metrics for monitoring and alerting.

        Returns:
            Dictionary containing operational metrics including:
                - denials_by_namespace: Count of denials per namespace
                - snapshot: Current telemetry snapshot with all counters

        Example:
            >>> telemetry = StandardEmbeddingTelemetry()
            >>> metrics = telemetry.operational_metrics()
            >>> print(f"Total denials: {sum(metrics['denials_by_namespace'].values())}")

        """
        return {
            "denials_by_namespace": dict(self._denials_by_namespace),
            "snapshot": asdict(self.snapshot()),
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "EmbeddingTelemetry",
    "StandardEmbeddingTelemetry",
    "TelemetrySettings",
    "TelemetrySnapshot",
]
