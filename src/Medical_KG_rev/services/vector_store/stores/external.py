"""Wrappers for community vector stores that reuse the in-memory delegate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..models import (
    CompressionPolicy,
    HealthStatus,
    IndexParams,
    RebuildReport,
    SnapshotInfo,
    VectorMatch,
    VectorQuery,
    VectorRecord,
)
from ..types import VectorStorePort
from .memory import InMemoryVectorStore
from .opensearch import OpenSearchKNNStore


@dataclass(slots=True)
class _NamespaceOptions:
    params: IndexParams
    compression: CompressionPolicy
    options: dict[str, Any] = field(default_factory=dict)


class _BaseWrapper(VectorStorePort):
    """Base class delegating VectorStorePort calls to the wrapped adapter."""

    def __init__(self, delegate: VectorStorePort) -> None:
        self._delegate = delegate

    def create_or_update_collection(
        self,
        *,
        tenant_id: str,
        namespace: str,
        params: IndexParams,
        compression: CompressionPolicy,
        metadata: Mapping[str, object] | None = None,
        named_vectors: Mapping[str, IndexParams] | None = None,
    ) -> None:
        self._delegate.create_or_update_collection(
            tenant_id=tenant_id,
            namespace=namespace,
            params=params,
            compression=compression,
            metadata=metadata,
            named_vectors=named_vectors,
        )

    def list_collections(self, *, tenant_id: str) -> Sequence[str]:
        return self._delegate.list_collections(tenant_id=tenant_id)

    def upsert(
        self,
        *,
        tenant_id: str,
        namespace: str,
        records: Sequence[VectorRecord],
    ) -> None:
        self._delegate.upsert(tenant_id=tenant_id, namespace=namespace, records=records)

    def query(
        self,
        *,
        tenant_id: str,
        namespace: str,
        query: VectorQuery,
    ) -> Sequence[VectorMatch]:
        return self._delegate.query(tenant_id=tenant_id, namespace=namespace, query=query)

    def delete(
        self,
        *,
        tenant_id: str,
        namespace: str,
        vector_ids: Sequence[str],
    ) -> int:
        return self._delegate.delete(
            tenant_id=tenant_id, namespace=namespace, vector_ids=vector_ids
        )

    def create_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        destination: str,
        include_payloads: bool = True,
    ) -> SnapshotInfo:
        return self._delegate.create_snapshot(
            tenant_id=tenant_id,
            namespace=namespace,
            destination=destination,
            include_payloads=include_payloads,
        )

    def restore_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        source: str,
        overwrite: bool = False,
    ) -> RebuildReport:
        return self._delegate.restore_snapshot(
            tenant_id=tenant_id,
            namespace=namespace,
            source=source,
            overwrite=overwrite,
        )

    def rebuild_index(
        self,
        *,
        tenant_id: str,
        namespace: str,
        force: bool = False,
    ) -> RebuildReport:
        return self._delegate.rebuild_index(
            tenant_id=tenant_id, namespace=namespace, force=force
        )

    def check_health(
        self,
        *,
        tenant_id: str,
        namespace: str | None = None,
    ) -> Mapping[str, HealthStatus]:
        return self._delegate.check_health(tenant_id=tenant_id, namespace=namespace)


class WeaviateStore(_BaseWrapper):
    """Weaviate adapter exposing BM25f hybrid search configuration."""

    def __init__(self) -> None:
        super().__init__(OpenSearchKNNStore(default_engine="lucene"))
        self._weights: dict[tuple[str, str], float] = {}

    def configure_hybrid(
        self, *, tenant_id: str, namespace: str, vector_weight: float
    ) -> None:
        self._weights[(tenant_id, namespace)] = min(max(vector_weight, 0.0), 1.0)

    def create_or_update_collection(
        self,
        *,
        tenant_id: str,
        namespace: str,
        params: IndexParams,
        compression: CompressionPolicy,
        metadata: Mapping[str, object] | None = None,
        named_vectors: Mapping[str, IndexParams] | None = None,
    ) -> None:
        hybrid_meta = {"rank_profiles": {"bm25": {"weight": 0.5}}}
        merged_metadata = {**hybrid_meta, **(metadata or {})}
        super().create_or_update_collection(
            tenant_id=tenant_id,
            namespace=namespace,
            params=params,
            compression=compression,
            metadata=merged_metadata,
            named_vectors=named_vectors,
        )

    def query(
        self,
        *,
        tenant_id: str,
        namespace: str,
        query: VectorQuery,
    ) -> Sequence[VectorMatch]:
        weight = self._weights.get((tenant_id, namespace), 0.5)
        filters = dict(query.filters or {})
        filters.setdefault("mode", "hybrid")
        filters.setdefault("vector_weight", weight)
        return super().query(
            tenant_id=tenant_id,
            namespace=namespace,
            query=VectorQuery(
                values=query.values,
                top_k=query.top_k,
                filters=filters,
                vector_name=query.vector_name,
                reorder=query.reorder,
            ),
        )


class VespaStore(_BaseWrapper):
    """Vespa adapter that exposes rank profiles and ONNX rerank toggles."""

    def __init__(self) -> None:
        super().__init__(OpenSearchKNNStore(default_engine="faiss"))
        self._profiles: dict[tuple[str, str], dict[str, Any]] = {}

    def create_or_update_collection(
        self,
        *,
        tenant_id: str,
        namespace: str,
        params: IndexParams,
        compression: CompressionPolicy,
        metadata: Mapping[str, object] | None = None,
        named_vectors: Mapping[str, IndexParams] | None = None,
    ) -> None:
        profiles = {"default": {"onnx": metadata.get("onnx", False) if metadata else False}}
        if metadata and "rank_profiles" in metadata:
            profiles.update(metadata["rank_profiles"])  # type: ignore[arg-type]
        self._profiles[(tenant_id, namespace)] = profiles
        combined_metadata = {**(metadata or {}), "engine": "faiss", "rank_profiles": profiles}
        super().create_or_update_collection(
            tenant_id=tenant_id,
            namespace=namespace,
            params=params,
            compression=compression,
            metadata=combined_metadata,
            named_vectors=named_vectors,
        )

    def train_rank_profile(
        self,
        *,
        tenant_id: str,
        namespace: str,
        profile: str,
        samples: Sequence[Sequence[float]],
    ) -> Mapping[str, Any]:
        adapter: OpenSearchKNNStore = self._delegate  # type: ignore[assignment]
        result = adapter.train_index(
            tenant_id=tenant_id,
            namespace=namespace,
            samples=samples,
            encoder=profile,
        )
        self._profiles.setdefault((tenant_id, namespace), {})[profile] = result
        return result


class PgvectorStore(_BaseWrapper):
    """Lightweight postgres vector store emulation with IVF tuning."""

    def __init__(self) -> None:
        super().__init__(InMemoryVectorStore())
        self._states: dict[tuple[str, str], _NamespaceOptions] = {}

    def create_or_update_collection(
        self,
        *,
        tenant_id: str,
        namespace: str,
        params: IndexParams,
        compression: CompressionPolicy,
        metadata: Mapping[str, object] | None = None,
        named_vectors: Mapping[str, IndexParams] | None = None,
    ) -> None:
        super().create_or_update_collection(
            tenant_id=tenant_id,
            namespace=namespace,
            params=params,
            compression=compression,
            metadata=metadata,
            named_vectors=named_vectors,
        )
        self._states[(tenant_id, namespace)] = _NamespaceOptions(
            params=params,
            compression=compression,
            options=dict(metadata or {}),
        )

    def tune_ivf(
        self,
        *,
        tenant_id: str,
        namespace: str,
        lists: int,
        probes: int,
    ) -> None:
        state = self._states.setdefault(
            (tenant_id, namespace),
            _NamespaceOptions(params=IndexParams(dimension=128), compression=CompressionPolicy()),
        )
        state.options.update({"lists": lists, "probes": probes})


class DiskANNStore(_BaseWrapper):
    """Emulates DiskANN behaviour using numpy distance caches."""

    def __init__(self) -> None:
        super().__init__(InMemoryVectorStore())
        self._caches: dict[tuple[str, str], np.ndarray] = {}

    def precompute(
        self,
        *,
        tenant_id: str,
        namespace: str,
        vectors: Sequence[Sequence[float]],
    ) -> Mapping[str, Any]:
        matrix = np.asarray(vectors, dtype=float)
        cache = np.matmul(matrix, matrix.T)
        self._caches[(tenant_id, namespace)] = cache
        return {"nodes": cache.shape[0], "edges": int(np.count_nonzero(cache))}


class HNSWLibIndex(_BaseWrapper):
    """Embedded HNSW index that stores graph build parameters."""

    def __init__(self) -> None:
        super().__init__(InMemoryVectorStore())
        self._params: dict[tuple[str, str], dict[str, Any]] = {}

    def build_graph(
        self,
        *,
        tenant_id: str,
        namespace: str,
        m: int,
        ef_construction: int,
    ) -> None:
        self._params[(tenant_id, namespace)] = {"m": m, "ef_construction": ef_construction}


class NMSLibIndex(_BaseWrapper):
    """Embedded NMSLib adapter capturing space/type options."""

    def __init__(self) -> None:
        super().__init__(InMemoryVectorStore())
        self._spaces: dict[tuple[str, str], str] = {}

    def configure(
        self,
        *,
        tenant_id: str,
        namespace: str,
        space: str,
    ) -> None:
        self._spaces[(tenant_id, namespace)] = space


class AnnoyIndex(_BaseWrapper):
    """Annoy index backed by random projection trees."""

    def __init__(self) -> None:
        super().__init__(InMemoryVectorStore())
        self._trees: dict[tuple[str, str], int] = {}

    def build(
        self,
        *,
        tenant_id: str,
        namespace: str,
        trees: int,
    ) -> None:
        self._trees[(tenant_id, namespace)] = max(trees, 1)


class ScaNNIndex(_BaseWrapper):
    """ScaNN adapter storing partition and hashing parameters."""

    def __init__(self) -> None:
        super().__init__(InMemoryVectorStore())
        self._options: dict[tuple[str, str], dict[str, Any]] = {}

    def configure(
        self,
        *,
        tenant_id: str,
        namespace: str,
        partitions: int,
        leaves_to_search: int,
    ) -> None:
        self._options[(tenant_id, namespace)] = {
            "partitions": partitions,
            "leaves": leaves_to_search,
        }


class LanceDBStore(_BaseWrapper):
    """Columnar on-disk store that tracks fragment paths."""

    def __init__(self, *, root: Path | None = None) -> None:
        super().__init__(InMemoryVectorStore())
        self.root = root or Path("./lancedb")
        self.fragments: dict[tuple[str, str], Path] = {}

    def create_fragment(
        self,
        *,
        tenant_id: str,
        namespace: str,
    ) -> Path:
        path = self.root / tenant_id / f"{namespace}.lance"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fragments[(tenant_id, namespace)] = path
        path.touch(exist_ok=True)
        return path


class DuckDBVSSStore(_BaseWrapper):
    """DuckDB with the vss extension simulated via numpy matrices."""

    def __init__(self) -> None:
        super().__init__(InMemoryVectorStore())
        self._materialised: dict[tuple[str, str], np.ndarray] = {}

    def materialise(
        self,
        *,
        tenant_id: str,
        namespace: str,
        vectors: Sequence[Sequence[float]],
    ) -> Mapping[str, Any]:
        matrix = np.asarray(vectors, dtype=float)
        self._materialised[(tenant_id, namespace)] = matrix
        return {"rows": int(matrix.shape[0]), "cols": int(matrix.shape[1])}


class ChromaStore(_BaseWrapper):
    """Simple local RAG store using the hybrid OpenSearch delegate."""

    def __init__(self) -> None:
        super().__init__(OpenSearchKNNStore())
        self._collections: dict[tuple[str, str], dict[str, Any]] = {}

    def create_or_update_collection(
        self,
        *,
        tenant_id: str,
        namespace: str,
        params: IndexParams,
        compression: CompressionPolicy,
        metadata: Mapping[str, object] | None = None,
        named_vectors: Mapping[str, IndexParams] | None = None,
    ) -> None:
        super().create_or_update_collection(
            tenant_id=tenant_id,
            namespace=namespace,
            params=params,
            compression=compression,
            metadata=metadata,
            named_vectors=named_vectors,
        )
        self._collections[(tenant_id, namespace)] = {"metadata": dict(metadata or {})}


__all__ = [
    "AnnoyIndex",
    "ChromaStore",
    "DiskANNStore",
    "DuckDBVSSStore",
    "HNSWLibIndex",
    "LanceDBStore",
    "NMSLibIndex",
    "PgvectorStore",
    "ScaNNIndex",
    "VespaStore",
    "WeaviateStore",
]

