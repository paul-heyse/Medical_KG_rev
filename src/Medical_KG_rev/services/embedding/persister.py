"""Embedding persistence interfaces and implementations.

This module provides the abstraction layer for persisting embedding records to
various storage backends, including vector stores, databases, and hybrid configurations.
It implements a flexible persister pattern that supports caching, telemetry, and
multiple storage strategies.

Key Responsibilities:
    - Define abstract EmbeddingPersister interface for storage operations
    - Implement concrete persisters for different storage backends
    - Provide caching and performance optimization features
    - Support hybrid persistence strategies for different embedding kinds
    - Emit telemetry and metrics for persistence operations

Collaborators:
    - Upstream: EmbeddingCoordinator, embedding generation services
    - Downstream: StorageRouter, vector store implementations, telemetry systems

Side Effects:
    - Writes embedding records to persistent storage
    - Updates in-memory cache for performance
    - Emits persistence metrics and telemetry events
    - Logs persistence operations and errors

Thread Safety:
    - Not thread-safe: Persisters maintain mutable cache state
    - Use separate persister instances per thread or add locking

Performance Characteristics:
    - O(n) persistence time for n records
    - O(1) cache lookups with LRU eviction
    - Configurable cache limits and backend selection

Example:
    >>> from Medical_KG_rev.services.embedding.persister import build_persister
    >>> persister = build_persister(storage_router, settings=settings)
    >>> report = persister.persist_batch(records, context)
    >>> print(f"Persisted {report.persisted} records in {report.duration_ms}ms")

"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import structlog
from Medical_KG_rev.embeddings.ports import EmbeddingRecord
from Medical_KG_rev.embeddings.storage import StorageRouter

from .telemetry import EmbeddingTelemetry

logger = structlog.get_logger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(slots=True)
class PersistenceContext:
    """Context passed to persisters for storing embeddings.

    Contains metadata about the persistence operation including tenant information,
    namespace configuration, model details, and optional tracking identifiers.

    Attributes:
        tenant_id: Unique identifier for the tenant owning the embeddings
        namespace: Namespace within which embeddings are stored
        model: Name of the embedding model used to generate vectors
        provider: Optional provider identifier for the embedding model
        job_id: Optional job identifier for tracking persistence operations
        correlation_id: Optional correlation ID for distributed tracing
        normalize: Whether to normalize embeddings before storage

    Example:
        >>> context = PersistenceContext(
        ...     tenant_id="tenant1",
        ...     namespace="documents",
        ...     model="sentence-transformers/all-MiniLM-L6-v2",
        ...     provider="huggingface",
        ...     job_id="job_123",
        ...     normalize=True
        ... )

    """

    tenant_id: str
    namespace: str
    model: str
    provider: str | None
    job_id: str | None = None
    correlation_id: str | None = None
    normalize: bool = False


@dataclass(slots=True)
class PersistenceReport:
    """Summary returned after persisting embeddings.

    Provides detailed statistics about persistence operations including success counts,
    error information, and performance metrics.

    Attributes:
        persisted: Number of embedding records successfully persisted
        skipped: Number of records skipped (e.g., duplicates, invalid)
        errors: List of error messages encountered during persistence
        duration_ms: Total time taken for persistence operation in milliseconds

    Example:
        >>> report = PersistenceReport()
        >>> report.persisted = 100
        >>> report.duration_ms = 250.5
        >>> print(f"Persisted {report.persisted} records in {report.duration_ms}ms")

    """

    persisted: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def record_error(self, message: str) -> None:
        """Record an error message in the report.

        Args:
            message: Error message describing what went wrong

        Note:
            This method is thread-safe and can be called from multiple threads.

        """
        self.errors.append(message)


@dataclass(slots=True)
class PersisterRuntimeSettings:
    """Runtime configuration describing how persistence should behave.

    Defines the configuration parameters for embedding persistence including
    backend selection, caching behavior, and hybrid persistence strategies.

    Attributes:
        backend: Storage backend to use ("vector_store", "database", "dry_run", "hybrid")
        cache_limit: Maximum number of records to keep in memory cache
        hybrid_backends: Mapping of embedding kinds to backend names for hybrid persistence

    Example:
        >>> settings = PersisterRuntimeSettings(
        ...     backend="hybrid",
        ...     cache_limit=512,
        ...     hybrid_backends={"text": "vector_store", "image": "database"}
        ... )

    """

    backend: str = "vector_store"
    cache_limit: int = 256
    hybrid_backends: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> PersisterRuntimeSettings:
        """Create settings from a configuration mapping.

        Args:
            payload: Configuration dictionary with backend, cache_limit, and hybrid_backends

        Returns:
            PersisterRuntimeSettings instance with parsed configuration

        Note:
            Invalid or missing values use defaults. Hybrid backends must be a mapping.

        """
        if not payload:
            return cls()
        data = dict(payload)
        backend = str(data.get("backend", "vector_store"))
        cache_limit = int(data.get("cache_limit", 256))
        hybrid = data.get("hybrid_backends", {})
        if isinstance(hybrid, Mapping):
            hybrid_backends = {str(key): str(value) for key, value in hybrid.items()}
        else:
            hybrid_backends = {}
        return cls(backend=backend, cache_limit=cache_limit, hybrid_backends=hybrid_backends)


# ============================================================================
# INTERFACE (Protocols/ABCs)
# ============================================================================


class EmbeddingPersister(ABC):
    """Abstract base class describing the persister contract.

    Defines the interface for persisting embedding records to various storage
    backends. Provides caching, telemetry, and error handling capabilities.

    This class implements the Template Method pattern where concrete persisters
    implement the abstract _persist method while inheriting common functionality
    like caching, error handling, and telemetry.

    Attributes:
        _telemetry: Optional telemetry instance for metrics collection
        _logger: Structured logger bound to this persister instance
        _cache: In-memory cache of embedding records for performance
        _recent: LRU queue tracking recently accessed records
        _cache_limit: Maximum number of records to keep in cache

    Invariants:
        - _cache size never exceeds _cache_limit
        - _recent queue length never exceeds _cache_limit
        - Cache eviction follows LRU policy

    Thread Safety:
        - Not thread-safe: Cache operations are not synchronized
        - Use separate instances per thread or add external locking

    Lifecycle:
        - Created via build_persister factory function
        - Configured with runtime settings
        - Used for batch persistence operations
        - Cache persists for lifetime of instance

    Example:
        >>> class CustomPersister(EmbeddingPersister):
        ...     def _persist(self, records, context, report):
        ...         # Custom persistence logic
        ...         pass
        >>> persister = CustomPersister(telemetry=telemetry)
        >>> report = persister.persist_batch(records, context)

    """

    def __init__(self, *, telemetry: EmbeddingTelemetry | None = None) -> None:
        """Initialize the persister with optional telemetry.

        Args:
            telemetry: Optional telemetry instance for metrics collection

        Note:
            Sets up logging, cache, and LRU tracking with default limits.

        """
        self._telemetry = telemetry
        self._logger = logger.bind(persister=self.__class__.__name__)
        self._cache: MutableMapping[str, EmbeddingRecord] = {}
        self._recent: deque[str] = deque(maxlen=256)
        self._cache_limit = 256

    @abstractmethod
    def _persist(
        self,
        records: Sequence[EmbeddingRecord],
        context: PersistenceContext,
        report: PersistenceReport,
    ) -> None:
        """Persist records to the backing store.

        Abstract method that concrete persisters must implement to define
        their specific persistence logic.

        Args:
            records: Sequence of embedding records to persist
            context: Persistence context with metadata and configuration
            report: Report object to update with persistence results

        Note:
            Implementations should update report.persisted, report.skipped,
            and report.errors as appropriate. Cache updates are handled
            by the base class.

        """

    def persist_batch(
        self, records: Sequence[EmbeddingRecord], context: PersistenceContext
    ) -> PersistenceReport:
        """Persist a batch of embedding records.

        Coordinates the persistence process including error handling,
        performance measurement, and telemetry emission.

        Args:
            records: Sequence of embedding records to persist
            context: Persistence context with metadata and configuration

        Returns:
            PersistenceReport with detailed results and metrics

        Note:
            Emits telemetry events and logs errors if telemetry is configured.
            Performance is measured in milliseconds.

        """
        report = PersistenceReport()
        started = perf_counter()
        try:
            self._persist(records, context, report)
        except Exception as exc:  # pragma: no cover - defensive
            report.record_error(str(exc))
            self._logger.exception(
                "persister.persist.error",
                namespace=context.namespace,
                tenant_id=context.tenant_id,
                error=str(exc),
            )
        report.duration_ms = (perf_counter() - started) * 1000
        if self._telemetry:
            self._telemetry.record_persistence(
                report, namespace=context.namespace, tenant_id=context.tenant_id
            )
        return report

    def _remember(self, record: EmbeddingRecord) -> None:
        """Add a record to the cache with LRU eviction.

        Args:
            record: Embedding record to cache

        Note:
            Evicts oldest records if cache limit is exceeded.

        """
        self._cache[record.id] = record
        self._recent.appendleft(record.id)
        while len(self._cache) > self._cache_limit and self._recent:
            evicted = self._recent.pop()
            self._cache.pop(evicted, None)

    def retrieve(self, *, ids: Sequence[str] | None = None) -> list[EmbeddingRecord]:
        """Retrieve embedding records from cache.

        Args:
            ids: Optional sequence of record IDs to retrieve. If None, returns all cached records.

        Returns:
            List of embedding records found in cache

        Note:
            Only returns records that are currently cached.

        """
        if ids is None:
            return list(self._cache.values())
        return [self._cache[item] for item in ids if item in self._cache]

    def delete(self, *, ids: Sequence[str]) -> int:
        """Delete embedding records from cache.

        Args:
            ids: Sequence of record IDs to delete

        Returns:
            Number of records successfully deleted

        Note:
            Safe to call with non-existent IDs.

        """
        removed = 0
        for item_id in ids:
            if self._cache.pop(item_id, None) is not None:
                removed += 1
        return removed

    def search(self, *, metadata: Mapping[str, object] | None = None) -> list[EmbeddingRecord]:
        """Search cached records by metadata.

        Args:
            metadata: Optional metadata filter. If None, returns all cached records.

        Returns:
            List of records matching metadata criteria

        Note:
            Performs exact match on all specified metadata keys.

        """
        if not metadata:
            return list(self._cache.values())
        results: list[EmbeddingRecord] = []
        for record in self._cache.values():
            if all(record.metadata.get(key) == value for key, value in metadata.items()):
                results.append(record)
        return results

    def debug_snapshot(self) -> Mapping[str, object]:
        """Get debug information about cache state.

        Returns:
            Dictionary with cache statistics and recent IDs

        Note:
            Useful for debugging cache behavior and performance.

        """
        return {
            "cached_records": len(self._cache),
            "recent_ids": list(self._recent),
        }

    def health_status(self) -> Mapping[str, object]:
        """Get health status information.

        Returns:
            Dictionary with persister health information

        Note:
            Used by health check endpoints and monitoring systems.

        """
        return {
            "persister": self.__class__.__name__,
            "cached_records": len(self._cache),
        }

    def configure(self, **kwargs: object) -> None:
        """Configure persister runtime settings.

        Args:
            **kwargs: Configuration parameters including cache_limit

        Raises:
            ValueError: If cache_limit is negative

        Note:
            Logs configuration changes for audit purposes.

        """
        if "cache_limit" in kwargs:
            limit = int(kwargs["cache_limit"])
            if limit < 0:
                raise ValueError("cache_limit must be non-negative")
            self._cache_limit = limit
            while len(self._recent) > self._cache_limit and self._recent:
                evicted = self._recent.pop()
                self._cache.pop(evicted, None)
        if kwargs:
            self._logger.info("persister.configure", **kwargs)


# ============================================================================
# IMPLEMENTATIONS
# ============================================================================


class VectorStorePersister(EmbeddingPersister):
    """Persister that writes embeddings via the shared storage router.

    Implements persistence to vector store backends through the StorageRouter
    abstraction. This is the primary persister for production use cases.

    Attributes:
        _storage_router: StorageRouter instance for vector store operations

    Example:
        >>> persister = VectorStorePersister(storage_router, telemetry=telemetry)
        >>> report = persister.persist_batch(records, context)

    """

    def __init__(
        self,
        storage_router: StorageRouter,
        *,
        telemetry: EmbeddingTelemetry | None = None,
    ) -> None:
        """Initialize vector store persister.

        Args:
            storage_router: StorageRouter for vector store operations
            telemetry: Optional telemetry instance for metrics

        """
        super().__init__(telemetry=telemetry)
        self._storage_router = storage_router

    def _persist(
        self,
        records: Sequence[EmbeddingRecord],
        context: PersistenceContext,
        report: PersistenceReport,
    ) -> None:
        """Persist records to vector store via storage router.

        Args:
            records: Embedding records to persist
            context: Persistence context (unused by this implementation)
            report: Report to update with persistence results

        """
        for record in records:
            self._storage_router.persist(record)
            self._remember(record)
            report.persisted += 1


class DatabasePersister(EmbeddingPersister):
    """Persister that stores embeddings in an in-memory map for Neo4j compatibility.

    Implements persistence to an in-memory dictionary for testing and development.
    Provides additional storage methods beyond the base cache.

    Attributes:
        _store: In-memory dictionary storing embedding records

    Example:
        >>> persister = DatabasePersister(telemetry=telemetry)
        >>> report = persister.persist_batch(records, context)

    """

    def __init__(self, *, telemetry: EmbeddingTelemetry | None = None) -> None:
        """Initialize database persister.

        Args:
            telemetry: Optional telemetry instance for metrics

        """
        super().__init__(telemetry=telemetry)
        self._store: dict[str, EmbeddingRecord] = {}

    def _persist(
        self,
        records: Sequence[EmbeddingRecord],
        context: PersistenceContext,
        report: PersistenceReport,
    ) -> None:
        """Persist records to in-memory store.

        Args:
            records: Embedding records to persist
            context: Persistence context (unused by this implementation)
            report: Report to update with persistence results

        """
        for record in records:
            self._store[record.id] = record
            self._remember(record)
            report.persisted += 1

    def retrieve(self, *, ids: Sequence[str] | None = None) -> list[EmbeddingRecord]:
        """Retrieve records from in-memory store.

        Args:
            ids: Optional sequence of record IDs to retrieve

        Returns:
            List of embedding records from store

        """
        if ids is None:
            return list(self._store.values())
        return [self._store[item] for item in ids if item in self._store]

    def delete(self, *, ids: Sequence[str]) -> int:
        """Delete records from both store and cache.

        Args:
            ids: Sequence of record IDs to delete

        Returns:
            Number of records successfully deleted

        """
        removed = 0
        for item_id in ids:
            if self._store.pop(item_id, None) is not None:
                removed += 1
            self._cache.pop(item_id, None)
        return removed


class DryRunPersister(EmbeddingPersister):
    """Persister that records operations without writing to storage.

    Useful for testing and validation without actual persistence.
    Tracks all operations for later inspection.

    Attributes:
        _operations: List of persistence contexts that would have been processed

    Example:
        >>> persister = DryRunPersister(telemetry=telemetry)
        >>> report = persister.persist_batch(records, context)
        >>> print(f"Would have processed {len(persister.operations)} operations")

    """

    def __init__(self, *, telemetry: EmbeddingTelemetry | None = None) -> None:
        """Initialize dry run persister.

        Args:
            telemetry: Optional telemetry instance for metrics

        """
        super().__init__(telemetry=telemetry)
        self._operations: list[PersistenceContext] = []

    def _persist(
        self,
        records: Sequence[EmbeddingRecord],
        context: PersistenceContext,
        report: PersistenceReport,
    ) -> None:
        """Record operation without actual persistence.

        Args:
            records: Embedding records (counted as skipped)
            context: Persistence context to record
            report: Report to update with skip count

        """
        self._operations.append(context)
        report.skipped += len(records)

    @property
    def operations(self) -> Sequence[PersistenceContext]:
        """Get recorded operations.

        Returns:
            Tuple of persistence contexts that would have been processed

        """
        return tuple(self._operations)


class MockPersister(EmbeddingPersister):
    """Lightweight persister for unit tests.

    Stores records in memory for test verification.
    Does not require telemetry or complex configuration.

    Attributes:
        persisted_records: List of all records that have been persisted

    Example:
        >>> persister = MockPersister()
        >>> report = persister.persist_batch(records, context)
        >>> assert len(persister.persisted_records) == len(records)

    """

    def __init__(self) -> None:
        """Initialize mock persister for testing."""
        super().__init__()
        self.persisted_records: list[EmbeddingRecord] = []

    def _persist(
        self,
        records: Sequence[EmbeddingRecord],
        context: PersistenceContext,
        report: PersistenceReport,
    ) -> None:
        """Store records in memory for test verification.

        Args:
            records: Embedding records to store
            context: Persistence context (unused)
            report: Report to update with persistence count

        """
        self.persisted_records.extend(records)
        for record in records:
            self._remember(record)
            report.persisted += 1


class HybridPersister(EmbeddingPersister):
    """Persister that delegates to other persisters based on embedding kind.

    Routes different embedding types to specialized persisters.
    Useful for mixed storage strategies (e.g., text to vector store, images to database).

    Attributes:
        _persisters: Mapping of embedding kinds to persister instances

    Example:
        >>> persisters = {"text": VectorStorePersister(router), "image": DatabasePersister()}
        >>> persister = HybridPersister(persisters, telemetry=telemetry)
        >>> report = persister.persist_batch(mixed_records, context)

    """

    def __init__(
        self,
        persisters: Mapping[str, EmbeddingPersister],
        *,
        telemetry: EmbeddingTelemetry | None = None,
    ) -> None:
        """Initialize hybrid persister.

        Args:
            persisters: Mapping of embedding kinds to persister instances
            telemetry: Optional telemetry instance for metrics

        """
        super().__init__(telemetry=telemetry)
        self._persisters = dict(persisters)

    def _persist(
        self,
        records: Sequence[EmbeddingRecord],
        context: PersistenceContext,
        report: PersistenceReport,
    ) -> None:
        """Route records to appropriate persisters by kind.

        Args:
            records: Embedding records to persist
            context: Persistence context passed to all persisters
            report: Report to update with aggregated results

        Note:
            Records without registered persisters are counted as errors.

        """
        grouped: dict[str, list[EmbeddingRecord]] = {}
        for record in records:
            grouped.setdefault(record.kind, []).append(record)
        for kind, items in grouped.items():
            persister = self._persisters.get(kind)
            if not persister:
                report.record_error(f"No persister registered for kind '{kind}'")
                continue
            child_report = persister.persist_batch(items, context)
            report.persisted += child_report.persisted
            report.skipped += child_report.skipped
            report.errors.extend(child_report.errors)
            for record in items:
                self._remember(record)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def _build_backend(
    backend: str,
    router: StorageRouter,
    *,
    telemetry: EmbeddingTelemetry | None = None,
) -> EmbeddingPersister:
    """Build a single backend persister instance.

    Args:
        backend: Backend name ("vector_store", "database", "dry_run")
        router: StorageRouter for vector store operations
        telemetry: Optional telemetry instance

    Returns:
        Configured persister instance

    Raises:
        ValueError: If backend name is not recognized

    Note:
        Private helper function for build_persister.

    """
    backend = backend.lower()
    if backend == "vector_store":
        return VectorStorePersister(router, telemetry=telemetry)
    if backend == "database":
        return DatabasePersister(telemetry=telemetry)
    if backend == "dry_run":
        return DryRunPersister(telemetry=telemetry)
    raise ValueError(f"Unknown persister backend '{backend}'")


def build_persister(
    router: StorageRouter,
    *,
    telemetry: EmbeddingTelemetry | None = None,
    settings: PersisterRuntimeSettings | Mapping[str, Any] | None = None,
) -> EmbeddingPersister:
    """Instantiate a persister based on runtime configuration.

    Factory function that creates appropriate persister instances based on
    configuration settings. Supports single backend and hybrid configurations.

    Args:
        router: StorageRouter for vector store operations
        telemetry: Optional telemetry instance for metrics collection
        settings: Runtime settings or configuration mapping

    Returns:
        Configured persister instance ready for use

    Raises:
        ValueError: If hybrid backend lacks hybrid_backends mapping

    Example:
        >>> settings = PersisterRuntimeSettings(backend="vector_store", cache_limit=512)
        >>> persister = build_persister(router, telemetry=telemetry, settings=settings)
        >>> report = persister.persist_batch(records, context)

    """
    resolved = (
        settings
        if isinstance(settings, PersisterRuntimeSettings)
        else PersisterRuntimeSettings.from_mapping(settings)
    )

    if resolved.backend.lower() == "hybrid":
        if not resolved.hybrid_backends:
            raise ValueError("Hybrid persister requires 'hybrid_backends' mapping")
        mapping: dict[str, EmbeddingPersister] = {}
        for kind, backend in resolved.hybrid_backends.items():
            mapping[kind] = _build_backend(backend, router, telemetry=telemetry)
        persister: EmbeddingPersister = HybridPersister(mapping, telemetry=telemetry)
    else:
        persister = _build_backend(resolved.backend, router, telemetry=telemetry)

    persister.configure(cache_limit=resolved.cache_limit)
    return persister


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "DatabasePersister",
    "DryRunPersister",
    "EmbeddingPersister",
    "HybridPersister",
    "MockPersister",
    "PersistenceContext",
    "PersistenceReport",
    "PersisterRuntimeSettings",
    "VectorStorePersister",
    "build_persister",
]
