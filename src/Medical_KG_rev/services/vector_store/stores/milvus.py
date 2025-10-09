"""Milvus-backed vector store adapter with GPU and hybrid support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import time

import numpy as np

from ..errors import BackendUnavailableError, NamespaceNotFoundError, VectorStoreError
from ..gpu import GPUResourceManager
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

    def create_collection(
        self, name: str, schema: Mapping[str, Any], **kwargs: Any
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def create_index(
        self, name: str, field_name: str, params: Mapping[str, Any]
    ) -> None:  # pragma: no cover
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
        return [
            name.removeprefix(prefix)
            for name in self._client.list_collections()
            if name.startswith(prefix)
        ]

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
            matches.append(
                VectorMatch(vector_id=identifier, score=float(score), metadata=dict(metadata))
            )
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

    def create_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        destination: str,
        include_payloads: bool = True,
    ) -> SnapshotInfo:
        name = _collection_name(tenant_id, namespace)
        if not self._client.has_collection(name):
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id)
        path = self._resolve_snapshot_path(destination, tenant_id, namespace)
        state = self._collections.get(name)
        records: list[dict[str, object]] = []
        if state:
            for payload in state.records.values():
                record: dict[str, object] = {
                    "vector_id": payload["id"],
                    "metadata": dict(payload.get("metadata", {})),
                }
                if include_payloads:
                    record["values"] = list(payload.get("vector", []))
                    if payload.get("named_vectors"):
                        record["named_vectors"] = payload.get("named_vectors")
                records.append(record)
        payload = {
            "tenant_id": tenant_id,
            "namespace": namespace,
            "created_at": time.time(),
            "params": asdict(state.params) if state else {},
            "compression": asdict(state.compression) if state else {"kind": "none"},
            "collection_metadata": dict(state.metadata) if state else {},
            "named_vector_params": {
                name: asdict(params) for name, params in (state.named_vectors or {}).items()
            }
            if state
            else {},
            "records": records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True))
        stats = path.stat()
        return SnapshotInfo(
            namespace=namespace,
            path=str(path),
            size_bytes=stats.st_size,
            created_at=time.time(),
            metadata={"records": len(records), "include_payloads": include_payloads},
        )

    def restore_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        source: str,
        overwrite: bool = False,
    ) -> RebuildReport:
        payload = json.loads(Path(source).read_text())
        params_payload = payload.get("params") or {}
        compression_payload = payload.get("compression") or {"kind": "none"}
        named_params_payload = payload.get("named_vector_params") or {}
        params = IndexParams(**params_payload)
        compression = CompressionPolicy(**compression_payload)
        name = _collection_name(tenant_id, namespace)
        if overwrite:
            self._collections.pop(name, None)
            if isinstance(self._client, InMemoryMilvusClient):
                self._client.collections.pop(name, None)
        self.create_or_update_collection(
            tenant_id=tenant_id,
            namespace=namespace,
            params=params,
            compression=compression,
            metadata=payload.get("collection_metadata"),
            named_vectors={key: IndexParams(**value) for key, value in named_params_payload.items()}
            if named_params_payload
            else None,
        )
        records_payload = payload.get("records", [])
        records: list[VectorRecord] = []
        for record in records_payload:
            values = record.get("values") or []
            named_vectors = record.get("named_vectors") or None
            if not values and not named_vectors:
                continue
            records.append(
                VectorRecord(
                    vector_id=str(record["vector_id"]),
                    values=values,
                    metadata=record.get("metadata", {}),
                    named_vectors=named_vectors,
                )
            )
        if records:
            self.upsert(tenant_id=tenant_id, namespace=namespace, records=records)
        return RebuildReport(
            namespace=namespace,
            rebuilt=bool(records),
            details={"restored_records": len(records)},
        )

    def rebuild_index(
        self,
        *,
        tenant_id: str,
        namespace: str,
        force: bool = False,
    ) -> RebuildReport:
        name = _collection_name(tenant_id, namespace)
        if not self._client.has_collection(name):
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id)
        state = self._collections.get(name)
        if not state:
            return RebuildReport(namespace=namespace, rebuilt=True, details={"records": 0})
        index_params = {
            "index_type": _index_type(state.params),
            "metric_type": _metric(state.params.metric),
            "params": {
                key: value
                for key, value in state.metadata.get("indexes", {})
                .get("vector", {})
                .get("params", {})
                .items()
            }
            if state.metadata.get("indexes")
            else {},
        }
        if not index_params["params"]:
            index_params["params"] = {
                key: value
                for key, value in {
                    "nlist": state.params.nlist,
                    "m": state.params.m,
                    "efConstruction": state.params.ef_construct,
                    "search_k": state.params.nprobe,
                }.items()
                if value is not None
            }
        self._client.create_index(name, field_name="vector", params=index_params)
        if state.named_vectors:
            for vector_name, vector_params in state.named_vectors.items():
                nv_params = {
                    "index_type": _index_type(vector_params),
                    "metric_type": _metric(vector_params.metric),
                    "params": {
                        key: value
                        for key, value in {
                            "nlist": vector_params.nlist,
                            "m": vector_params.m,
                            "efConstruction": vector_params.ef_construct,
                            "search_k": vector_params.nprobe,
                        }.items()
                        if value is not None
                    },
                }
                self._client.create_index(name, field_name=vector_name, params=nv_params)
        return RebuildReport(
            namespace=namespace,
            rebuilt=True,
            details={"records": len(state.records)},
        )

    def check_health(
        self,
        *,
        tenant_id: str,
        namespace: str | None = None,
    ) -> Mapping[str, HealthStatus]:
        if namespace is not None:
            name = _collection_name(tenant_id, namespace)
            healthy = self._client.has_collection(name)
            state = self._collections.get(name)
            details = {"records": len(state.records) if state else 0}
            return {namespace: HealthStatus(name=namespace, healthy=healthy, details=details)}
        statuses: dict[str, HealthStatus] = {}
        for key in self._client.list_collections():
            if not key.startswith(f"{tenant_id}__"):
                continue
            namespace_name = key.split("__", 1)[1]
            state = self._collections.get(key)
            statuses[namespace_name] = HealthStatus(
                name=namespace_name,
                healthy=True,
                details={"records": len(state.records) if state else 0},
            )
        return statuses

    def _resolve_snapshot_path(self, destination: str, tenant_id: str, namespace: str) -> Path:
        path = Path(destination)
        if path.suffix:
            return path
        return path / f"{tenant_id}__{namespace}.milvus-snapshot.json"


__all__ = ["MilvusVectorStore", "InMemoryMilvusClient", "MilvusLikeClient"]
