"""Routing logic mapping embedding kinds to storage backends."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .ports import EmbeddingKind, EmbeddingRecord


@dataclass(slots=True)
class StorageTarget:
    name: str
    description: str
    handler: Callable[[EmbeddingRecord], None] | None = None


class StorageRouter:
    """Maps namespaces to the correct storage backend based on embedding kind."""

    def __init__(self) -> None:
        self._targets: dict[EmbeddingKind, StorageTarget] = {
            "single_vector": StorageTarget(name="faiss", description="Dense vector store (HNSW)"),
            "multi_vector": StorageTarget(name="faiss", description="Late interaction index"),
            "sparse": StorageTarget(name="opensearch_rank_features", description="Learned sparse rank_features"),
            "neural_sparse": StorageTarget(name="opensearch_neural", description="OpenSearch neural fields"),
        }
        self._buffers: dict[str, list[EmbeddingRecord]] = defaultdict(list)

    def route(self, kind: EmbeddingKind) -> StorageTarget:
        try:
            return self._targets[kind]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"No storage target registered for kind '{kind}'") from exc

    def register(self, kind: EmbeddingKind, target: StorageTarget) -> None:
        self._targets[kind] = target

    def persist(self, record: EmbeddingRecord) -> None:
        target = self.route(record.kind)
        if target.handler:
            target.handler(record)
            return
        # Default to buffering for inspection/testing when no backend handler provided.
        key = self._buffer_key(target.name, record.tenant_id)
        self._buffers[key].append(record)

    def buffered(self, name: str, tenant_id: str | None = None) -> Sequence[EmbeddingRecord]:
        return list(self._buffers.get(self._buffer_key(name, tenant_id), ()))

    def _buffer_key(self, target_name: str, tenant_id: str | None) -> str:
        tenant = tenant_id or "unknown"
        return f"{target_name}:{tenant}"
