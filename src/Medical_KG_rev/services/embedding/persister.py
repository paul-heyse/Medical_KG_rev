"""Embedding persistence interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Mapping, MutableMapping, Sequence

import structlog

from Medical_KG_rev.embeddings.ports import EmbeddingRecord
from Medical_KG_rev.embeddings.storage import StorageRouter

from .telemetry import EmbeddingTelemetry

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class PersistenceContext:
    """Context passed to persisters for storing embeddings."""

    tenant_id: str
    namespace: str
    model: str
    provider: str | None
    job_id: str | None = None
    correlation_id: str | None = None
    normalize: bool = False


@dataclass(slots=True)
class PersistenceReport:
    """Summary returned after persisting embeddings."""

    persisted: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def record_error(self, message: str) -> None:
        self.errors.append(message)


class EmbeddingPersister(ABC):
    """Abstract base class describing the persister contract."""

    def __init__(self, *, telemetry: EmbeddingTelemetry | None = None) -> None:
        self._telemetry = telemetry
        self._logger = logger.bind(persister=self.__class__.__name__)
        self._cache: MutableMapping[str, EmbeddingRecord] = {}
        self._recent: deque[str] = deque(maxlen=256)
        self._cache_limit = 256

    @abstractmethod
    def _persist(self, records: Sequence[EmbeddingRecord], context: PersistenceContext, report: PersistenceReport) -> None:
        """Persist records to the backing store."""

    def persist_batch(self, records: Sequence[EmbeddingRecord], context: PersistenceContext) -> PersistenceReport:
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
            self._telemetry.record_persistence(report, namespace=context.namespace, tenant_id=context.tenant_id)
        return report

    def _remember(self, record: EmbeddingRecord) -> None:
        self._cache[record.id] = record
        self._recent.appendleft(record.id)
        while len(self._cache) > self._cache_limit and self._recent:
            evicted = self._recent.pop()
            self._cache.pop(evicted, None)

    def retrieve(self, *, ids: Sequence[str] | None = None) -> list[EmbeddingRecord]:
        if ids is None:
            return list(self._cache.values())
        return [self._cache[item] for item in ids if item in self._cache]

    def delete(self, *, ids: Sequence[str]) -> int:
        removed = 0
        for item_id in ids:
            if self._cache.pop(item_id, None) is not None:
                removed += 1
        return removed

    def search(self, *, metadata: Mapping[str, object] | None = None) -> list[EmbeddingRecord]:
        if not metadata:
            return list(self._cache.values())
        results: list[EmbeddingRecord] = []
        for record in self._cache.values():
            if all(record.metadata.get(key) == value for key, value in metadata.items()):
                results.append(record)
        return results

    def debug_snapshot(self) -> Mapping[str, object]:
        return {
            "cached_records": len(self._cache),
            "recent_ids": list(self._recent),
        }

    def health_status(self) -> Mapping[str, object]:
        return {
            "persister": self.__class__.__name__,
            "cached_records": len(self._cache),
        }

    def configure(self, **kwargs: object) -> None:
        if kwargs:
            self._logger.info("persister.configure", **kwargs)


class VectorStorePersister(EmbeddingPersister):
    """Persister that writes embeddings via the shared storage router."""

    def __init__(
        self,
        storage_router: StorageRouter,
        *,
        telemetry: EmbeddingTelemetry | None = None,
    ) -> None:
        super().__init__(telemetry=telemetry)
        self._storage_router = storage_router

    def _persist(self, records: Sequence[EmbeddingRecord], context: PersistenceContext, report: PersistenceReport) -> None:
        for record in records:
            self._storage_router.persist(record)
            self._remember(record)
            report.persisted += 1


class DatabasePersister(EmbeddingPersister):
    """Persister that stores embeddings in an in-memory map for Neo4j compatibility."""

    def __init__(self, *, telemetry: EmbeddingTelemetry | None = None) -> None:
        super().__init__(telemetry=telemetry)
        self._store: Dict[str, EmbeddingRecord] = {}

    def _persist(self, records: Sequence[EmbeddingRecord], context: PersistenceContext, report: PersistenceReport) -> None:
        for record in records:
            self._store[record.id] = record
            self._remember(record)
            report.persisted += 1

    def retrieve(self, *, ids: Sequence[str] | None = None) -> list[EmbeddingRecord]:
        if ids is None:
            return list(self._store.values())
        return [self._store[item] for item in ids if item in self._store]

    def delete(self, *, ids: Sequence[str]) -> int:
        removed = 0
        for item_id in ids:
            if self._store.pop(item_id, None) is not None:
                removed += 1
            self._cache.pop(item_id, None)
        return removed


class DryRunPersister(EmbeddingPersister):
    """Persister that records operations without writing to storage."""

    def __init__(self, *, telemetry: EmbeddingTelemetry | None = None) -> None:
        super().__init__(telemetry=telemetry)
        self._operations: list[PersistenceContext] = []

    def _persist(self, records: Sequence[EmbeddingRecord], context: PersistenceContext, report: PersistenceReport) -> None:
        self._operations.append(context)
        report.skipped += len(records)

    @property
    def operations(self) -> Sequence[PersistenceContext]:
        return tuple(self._operations)


class MockPersister(EmbeddingPersister):
    """Lightweight persister for unit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.persisted_records: list[EmbeddingRecord] = []

    def _persist(self, records: Sequence[EmbeddingRecord], context: PersistenceContext, report: PersistenceReport) -> None:
        self.persisted_records.extend(records)
        for record in records:
            self._remember(record)
            report.persisted += 1


class HybridPersister(EmbeddingPersister):
    """Persister that delegates to other persisters based on embedding kind."""

    def __init__(
        self,
        persisters: Mapping[str, EmbeddingPersister],
        *,
        telemetry: EmbeddingTelemetry | None = None,
    ) -> None:
        super().__init__(telemetry=telemetry)
        self._persisters = dict(persisters)

    def _persist(self, records: Sequence[EmbeddingRecord], context: PersistenceContext, report: PersistenceReport) -> None:
        grouped: Dict[str, list[EmbeddingRecord]] = {}
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


__all__ = [
    "EmbeddingPersister",
    "VectorStorePersister",
    "DatabasePersister",
    "DryRunPersister",
    "MockPersister",
    "HybridPersister",
    "PersistenceContext",
    "PersistenceReport",
]
