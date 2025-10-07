"""Milvus-backed vector store adapter with GPU and hybrid support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import BackendUnavailableError, NamespaceNotFoundError, VectorStoreError
from ..gpu import GPUResourceManager
from ..models import CompressionPolicy, IndexParams, VectorMatch, VectorQuery, VectorRecord
from ..types import VectorStorePort


def _collection_name(tenant_id: str, namespace: str) -> str:
    return f"{tenant_id}__{namespace}".replace("/", "_")


def _metric(metric: str) -> str:
    if metric.lower() in {"cosine", "ip", "inner_product"}:
        return "COSINE"
    if metric.lower() in {"l2", "euclidean"}:
        return "L2"
    return "COSINE"


def _index_type(params: IndexParams) -> str:
    kind = params.kind.lower()
    if kind in {"ivf_flat", "ivf"}:
        return "IVF_FLAT"
    if kind in {"ivf_pq", "pq"}:
        return "IVF_PQ"
    if kind == "hnsw":
        return "HNSW"
    if kind == "diskann":
        return "DISKANN"
    if kind in {"gpu_cagra", "cagra"}:
        return "GPU_CAGRA"
    return "IVF_FLAT"


@dataclass(slots=True)
class _CollectionState:
    params: IndexParams
    compression: CompressionPolicy
    metadata: Mapping[str, object]
    named_vectors: Mapping[str, IndexParams] | None
    records: dict[str, dict[str, Any]] = field(default_factory=dict)


class MilvusLikeClient:
    """Subset of operations required from pymilvus for easier testing."""

    def has_collection(self, name: str) -> bool:  # pragma: no cover - interface definition
        raise NotImplementedError

    def create_collection(self, name: str, schema: Mapping[str, Any], **kwargs: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def create_index(self, name: str, field_name: str, params: Mapping[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    def drop_index(self, name: str, field_name: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def insert(self, name: str, records: Sequence[Mapping[str, Any]]) -> None:  # pragma: no cover
        raise NotImplementedError

    def search(
        self,
        name: str,
        vector: Sequence[float],
        *,
        vector_name: str | None,
        top_k: int,
        metric: str,
        filters: Mapping[str, object] | None = None,
    ) -> list[tuple[str, float, Mapping[str, object]]]:  # pragma: no cover
        raise NotImplementedError

    def delete(self, name: str, ids: Sequence[str]) -> int:  # pragma: no cover
        raise NotImplementedError

    def list_collections(self) -> Sequence[str]:  # pragma: no cover
        raise NotImplementedError


class InMemoryMilvusClient(MilvusLikeClient):
    """Minimal in-memory implementation used for unit tests."""

    def __init__(self) -> None:
        self.collections: dict[str, _CollectionState] = {}

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, schema: Mapping[str, Any], **kwargs: Any) -> None:
        if name not in self.collections:
            self.collections[name] = _CollectionState(
                params=schema["params"],
                compression=schema["compression"],
                metadata=schema["metadata"],
                named_vectors=schema.get("named_vectors"),
            )

    def create_index(self, name: str, field_name: str, params: Mapping[str, Any]) -> None:
        state = self.collections[name]
        payload = dict(state.metadata)
        payload.setdefault("indexes", {})[field_name] = dict(params)
        state.metadata = payload

    def drop_index(self, name: str, field_name: str) -> None:
        state = self.collections[name]
        if "indexes" in state.metadata:
            state.metadata["indexes"].pop(field_name, None)

    def insert(self, name: str, records: Sequence[Mapping[str, Any]]) -> None:
        state = self.collections[name]
        for record in records:
            state.records[record["id"]] = dict(record)

    def search(
        self,
        name: str,
        vector: Sequence[float],
        *,
        vector_name: str | None,
        top_k: int,
        metric: str,
        filters: Mapping[str, object] | None = None,
    ) -> list[tuple[str, float, Mapping[str, object]]]:
        state = self.collections[name]
        results: list[tuple[str, float, Mapping[str, object]]] = []
        query = np.asarray(vector, dtype=np.float32)
        for record_id, payload in state.records.items():
            if filters and any(payload["metadata"].get(k) != v for k, v in filters.items()):
                continue
            if vector_name:
                target_vector = payload.get("named_vectors", {}).get(vector_name)
            else:
                target_vector = payload.get("vector")
            if target_vector is None:
                continue
            candidate = np.asarray(target_vector, dtype=np.float32)
            if metric == "COSINE":
                numerator = float(np.dot(query, candidate))
                denom = float(np.linalg.norm(query) * np.linalg.norm(candidate)) or 1e-6
                score = numerator / denom
            else:
                diff = query - candidate
                score = -float(np.dot(diff, diff))
            results.append((record_id, score, payload["metadata"]))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def delete(self, name: str, ids: Sequence[str]) -> int:
        state = self.collections[name]
        removed = 0
        for identifier in ids:
            if identifier in state.records:
                removed += 1
                del state.records[identifier]
        return removed

    def list_collections(self) -> Sequence[str]:
        return list(self.collections.keys())


class MilvusVectorStore(VectorStorePort):
    """Vector store adapter that mimics Milvus' capabilities."""

    def __init__(
        self,
        client: MilvusLikeClient | None = None,
        *,
        gpu_required: bool = False,
        batch_size: int = 512,
    ) -> None:
        self._client = client or InMemoryMilvusClient()
        self._gpu_required = gpu_required
        self._batch_size = batch_size
        self._gpu = GPUResourceManager(require_gpu=gpu_required)
        self._collections: dict[str, _CollectionState] = (
            self._client.collections if isinstance(self._client, InMemoryMilvusClient) else {}
        )

    # ------------------------------------------------------------------
    # VectorStorePort implementation
    # ------------------------------------------------------------------
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
        name = _collection_name(tenant_id, namespace)
        payload = {
            "params": params,
            "compression": compression,
            "metadata": dict(metadata or {}),
        }
        if named_vectors:
            payload["named_vectors"] = dict(named_vectors)
        gpu_enabled = False
        try:
            gpu_enabled = self._gpu.ensure() if params.use_gpu or self._gpu_required else False
        except RuntimeError as exc:
            raise BackendUnavailableError(str(exc)) from exc
        if not self._client.has_collection(name):
            self._client.create_collection(name, schema=payload)
            self._collections[name] = _CollectionState(
                params=params,
                compression=compression,
                metadata=dict(metadata or {}),
                named_vectors=dict(named_vectors or {}) or None,
            )
        else:
            state = self._collections.setdefault(
                name,
                _CollectionState(
                    params=params,
                    compression=compression,
                    metadata=dict(metadata or {}),
                    named_vectors=dict(named_vectors or {}) or None,
                ),
            )
            state.params = params
            state.compression = compression
            state.metadata = dict(metadata or {})
            state.named_vectors = dict(named_vectors or {}) or None
        index_params = {
            "index_type": _index_type(params),
            "metric_type": _metric(params.metric),
            "params": {
                key: value
                for key, value in {
                    "nlist": params.nlist,
                    "m": params.m,
                    "efConstruction": params.ef_construct,
                    "search_k": params.nprobe,
                    "gpu_id": params.gpu_id if gpu_enabled and params.gpu_id is not None else None,
                    "diskann_max_degree": params.m if params.kind == "diskann" else None,
                }.items()
                if value is not None
            },
        }
        self._client.drop_index(name, field_name="vector")
        self._client.create_index(name, field_name="vector", params=index_params)
        if named_vectors:
            for vector_name, vector_params in named_vectors.items():
                nv_params = dict(index_params)
                nv_params["index_type"] = _index_type(vector_params)
                nv_params["metric_type"] = _metric(vector_params.metric)
                self._client.drop_index(name, field_name=vector_name)
                self._client.create_index(name, field_name=vector_name, params=nv_params)

    def list_collections(self, *, tenant_id: str) -> Sequence[str]:
        prefix = f"{tenant_id}__"
        return [name.removeprefix(prefix) for name in self._client.list_collections() if name.startswith(prefix)]

    def upsert(
        self,
        *,
        tenant_id: str,
        namespace: str,
        records: Sequence[VectorRecord],
    ) -> None:
        if not records:
            return
        name = _collection_name(tenant_id, namespace)
        if not self._client.has_collection(name):
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id)
        try:
            for record in records:
                payload = {
                    "id": record.vector_id,
                    "vector": list(record.values),
                    "metadata": dict(record.metadata),
                }
                if record.named_vectors:
                    payload["named_vectors"] = {
                        name: list(values) for name, values in record.named_vectors.items()
                    }
                self._client.insert(name, [payload])
        except VectorStoreError:
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise BackendUnavailableError(str(exc)) from exc

    def query(
        self,
        *,
        tenant_id: str,
        namespace: str,
        query: VectorQuery,
    ) -> Sequence[VectorMatch]:
        name = _collection_name(tenant_id, namespace)
        if not self._client.has_collection(name):
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id)
        state = self._collections.get(name)
        metric = _metric(state.params.metric if state else "cosine")
        results = self._client.search(
            name,
            list(query.values),
            vector_name=query.vector_name,
            top_k=query.top_k,
            metric=metric,
            filters=query.filters,
        )
        matches: list[VectorMatch] = []
        for identifier, score, metadata in results:
            matches.append(VectorMatch(vector_id=identifier, score=float(score), metadata=dict(metadata)))
        return matches

    def delete(
        self,
        *,
        tenant_id: str,
        namespace: str,
        vector_ids: Sequence[str],
    ) -> int:
        name = _collection_name(tenant_id, namespace)
        if not self._client.has_collection(name):
            return 0
        return self._client.delete(name, vector_ids)


__all__ = ["MilvusVectorStore", "InMemoryMilvusClient", "MilvusLikeClient"]

