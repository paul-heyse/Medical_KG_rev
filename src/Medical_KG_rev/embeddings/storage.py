"""Routing logic mapping embedding kinds to storage backends."""

from __future__ import annotations

from dataclasses import dataclass

from .ports import EmbeddingKind


@dataclass(slots=True)
class StorageTarget:
    name: str
    description: str


class StorageRouter:
    """Maps namespaces to the correct storage backend based on embedding kind."""

    def __init__(self) -> None:
        self._targets: dict[EmbeddingKind, StorageTarget] = {
            "single_vector": StorageTarget(name="qdrant", description="Dense vector store"),
            "multi_vector": StorageTarget(name="faiss", description="Late interaction index"),
            "sparse": StorageTarget(name="opensearch", description="Learned sparse rank_features"),
            "neural_sparse": StorageTarget(name="opensearch_neural", description="OpenSearch neural fields"),
        }

    def route(self, kind: EmbeddingKind) -> StorageTarget:
        try:
            return self._targets[kind]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"No storage target registered for kind '{kind}'") from exc

    def register(self, kind: EmbeddingKind, target: StorageTarget) -> None:
        self._targets[kind] = target
