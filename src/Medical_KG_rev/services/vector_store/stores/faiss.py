from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
import faiss
import numpy as np

from ..errors import DimensionMismatchError, InvalidNamespaceConfigError, NamespaceNotFoundError
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


LOGGER = logging.getLogger(__name__)


def _json_default(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, dict)):
        return value
    return str(value)


class _SQLiteStore:
    """Sidecar SQLite store for payload metadata and float vectors."""

    def __init__(self, path: Path, dimension: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._dimension = dimension
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id TEXT PRIMARY KEY,
                    internal_id INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    vector BLOB NOT NULL
                )
                """
            )

    def upsert_many(
        self,
        entries: Sequence[tuple[str, int, Mapping[str, object], np.ndarray]],
    ) -> None:
        if not entries:
            return
        payloads = [
            (
                vector_id,
                internal_id,
                json.dumps(metadata or {}, default=_json_default),
                vector.astype(np.float32).tobytes(),
            )
            for vector_id, internal_id, metadata, vector in entries
        ]
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO vectors(vector_id, internal_id, payload, vector)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(vector_id) DO UPDATE SET
                    internal_id=excluded.internal_id,
                    payload=excluded.payload,
                    vector=excluded.vector
                """,
                payloads,
            )

    def fetch_metadata(self, vector_ids: Sequence[str]) -> dict[str, Mapping[str, object]]:
        if not vector_ids:
            return {}
        placeholders = ",".join("?" for _ in vector_ids)
        cursor = self._conn.execute(
            f"SELECT vector_id, payload FROM vectors WHERE vector_id IN ({placeholders})",
            list(vector_ids),
        )
        return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    def fetch_vectors(self, vector_ids: Sequence[str]) -> dict[str, np.ndarray]:
        if not vector_ids:
            return {}
        placeholders = ",".join("?" for _ in vector_ids)
        cursor = self._conn.execute(
            f"SELECT vector_id, vector FROM vectors WHERE vector_id IN ({placeholders})",
            list(vector_ids),
        )
        return {
            row[0]: np.frombuffer(row[1], dtype=np.float32, count=self._dimension)
            for row in cursor.fetchall()
        }

    def load_id_map(self) -> tuple[dict[str, int], int]:
        cursor = self._conn.execute("SELECT vector_id, internal_id FROM vectors")
        mapping = {vector_id: internal_id for vector_id, internal_id in cursor.fetchall()}
        next_id = (max(mapping.values()) + 1) if mapping else 1
        return mapping, next_id

    def delete(self, vector_ids: Sequence[str]) -> int:
        if not vector_ids:
            return 0
        with self._conn:
            cursor = self._conn.executemany(
                "DELETE FROM vectors WHERE vector_id = ?",
                [(vector_id,) for vector_id in vector_ids],
            )
        return cursor.rowcount if cursor is not None else 0


@dataclass(slots=True)
class _PendingRecord:
    vector_id: str
    internal_id: int
    vector: np.ndarray
    raw_vector: np.ndarray
    metadata: Mapping[str, object]


@dataclass(slots=True)
class _NamespaceState:
    cpu_index: faiss.Index
    payload_store: _SQLiteStore
    dimension: int
    metric: str
    params: IndexParams
    compression: CompressionPolicy
    index_path: Path
    normalize: bool
    reorder_enabled: bool
    reorder_k: int | None
    training_threshold: int
    requires_training: bool
    vector_id_to_internal: dict[str, int]
    internal_to_vector_id: dict[int, str]
    next_internal_id: int
    training_buffer: list[np.ndarray] = field(default_factory=list)
    pending_records: list[_PendingRecord] = field(default_factory=list)
    gpu_index: faiss.Index | None = None
    gpu_resources: faiss.GpuResources | None = None

    def reset_gpu(self) -> None:
        self.gpu_index = None


class FaissVectorStore(VectorStorePort):
    """FAISS-backed vector store with persistence and compression support."""

    def __init__(self, *, base_path: str | Path | None = None) -> None:
        self._tenants: dict[str, dict[str, _NamespaceState]] = {}
        self._base_path = Path(base_path or ".vector_store/faiss").resolve()

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
        if named_vectors:
            raise InvalidNamespaceConfigError(
                namespace,
                detail="FAISS adapter does not support named vectors.",
            )

        tenant_state = self._tenants.setdefault(tenant_id, {})
        existing = tenant_state.get(namespace)
        if existing:
            if existing.dimension != params.dimension:
                raise DimensionMismatchError(
                    existing.dimension, params.dimension, namespace=namespace
                )
            existing.params = params
            existing.compression = compression
            existing.reorder_k = params.reorder_k
            existing.reorder_enabled = self._should_enable_reorder(params, compression)
            return

        storage_root = Path(params.storage_path) if params.storage_path else self._base_path
        tenant_dir = storage_root / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        index_path = tenant_dir / f"{namespace}.faiss"
        sqlite_path = tenant_dir / f"{namespace}.sqlite"

        if index_path.exists():
            cpu_index = faiss.read_index(str(index_path))
        else:
            cpu_index = self._build_index(namespace, params, compression)

        if _index_dimension(cpu_index) != params.dimension:
            raise DimensionMismatchError(
                _index_dimension(cpu_index), params.dimension, namespace=namespace
            )

        payload_store = _SQLiteStore(sqlite_path, params.dimension)
        vector_map, next_internal_id = payload_store.load_id_map()

        state = _NamespaceState(
            cpu_index=cpu_index,
            payload_store=payload_store,
            dimension=params.dimension,
            metric=params.metric,
            params=params,
            compression=compression,
            index_path=index_path,
            normalize=_should_normalize(params.metric),
            reorder_enabled=self._should_enable_reorder(params, compression),
            reorder_k=params.reorder_k,
            training_threshold=_training_threshold(params, compression),
            requires_training=not cpu_index.is_trained,
            vector_id_to_internal=vector_map,
            internal_to_vector_id={
                internal: vector_id for vector_id, internal in vector_map.items()
            },
            next_internal_id=next_internal_id,
        )

        tenant_state[namespace] = state

    def list_collections(self, *, tenant_id: str) -> Sequence[str]:
        return list(self._tenants.get(tenant_id, {}).keys())

    def upsert(
        self,
        *,
        tenant_id: str,
        namespace: str,
        records: Sequence[VectorRecord],
    ) -> None:
        state = self._get_state(tenant_id, namespace)
        if not records:
            return

        pending_entries: list[tuple[str, int, Mapping[str, object], np.ndarray]] = []

        for record in records:
            if record.named_vectors:
                raise InvalidNamespaceConfigError(
                    namespace,
                    detail="FAISS adapter does not support named vectors.",
                )

            vector = np.asarray(record.values, dtype=np.float32)
            if vector.shape[0] != state.dimension:
                raise DimensionMismatchError(state.dimension, vector.shape[0], namespace=namespace)

            raw_vector = vector.copy()
            if state.normalize:
                vector = _normalize(vector)

            internal_id = state.vector_id_to_internal.get(record.vector_id)
            if internal_id is None:
                internal_id = state.next_internal_id
                state.next_internal_id += 1
            else:
                self._remove_internal_ids(state, [internal_id])

            state.vector_id_to_internal[record.vector_id] = internal_id
            state.internal_to_vector_id[internal_id] = record.vector_id

            state.pending_records = [
                r for r in state.pending_records if r.vector_id != record.vector_id
            ]
            state.pending_records.append(
                _PendingRecord(
                    vector_id=record.vector_id,
                    internal_id=internal_id,
                    vector=vector,
                    raw_vector=raw_vector,
                    metadata=record.metadata,
                )
            )

            if state.requires_training:
                state.training_buffer.append(vector)

            pending_entries.append((record.vector_id, internal_id, record.metadata, raw_vector))

        state.payload_store.upsert_many(pending_entries)
        self._train_if_ready(state)
        self._flush_pending(state)
        self._persist_index(state)

    def query(
        self,
        *,
        tenant_id: str,
        namespace: str,
        query: VectorQuery,
    ) -> Sequence[VectorMatch]:
        state = self._tenants.get(tenant_id, {}).get(namespace)
        if not state:
            return []
        if state.requires_training and not state.cpu_index.is_trained:
            return []

        vector = np.asarray(query.values, dtype=np.float32)
        if vector.shape[0] != state.dimension:
            raise DimensionMismatchError(state.dimension, vector.shape[0], namespace=namespace)

        raw_query = vector.copy()
        if state.normalize:
            vector = _normalize(vector)

        index = self._select_index(state)
        self._apply_search_params(index, state)

        search_k, should_reorder = self._resolve_search_k(state, query.top_k, query.reorder)
        distances, ids = index.search(vector.reshape(1, -1), search_k)

        matches: list[tuple[str, float]] = []
        for distance, internal_id in zip(distances[0], ids[0]):
            if internal_id == -1:
                continue
            vector_id = state.internal_to_vector_id.get(int(internal_id))
            if not vector_id:
                continue
            matches.append((vector_id, float(distance)))

        if not matches:
            return []

        vector_ids = [match[0] for match in matches]
        metadata = state.payload_store.fetch_metadata(vector_ids)

        if should_reorder:
            stored_vectors = state.payload_store.fetch_vectors(vector_ids)
            scored = [
                (
                    vector_id,
                    _score_vectors(
                        raw_query, stored_vectors.get(vector_id), state.metric, state.normalize
                    ),
                )
                for vector_id, _ in matches
                if stored_vectors.get(vector_id) is not None
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            scored = scored[: query.top_k]
        else:
            scored = [
                (
                    vector_id,
                    _distance_to_score(distance, state.metric),
                )
                for vector_id, distance in matches[: query.top_k]
            ]

        return [
            VectorMatch(vector_id=vector_id, score=score, metadata=metadata.get(vector_id, {}))
            for vector_id, score in scored
        ]

    def delete(
        self,
        *,
        tenant_id: str,
        namespace: str,
        vector_ids: Sequence[str],
    ) -> int:
        state = self._get_state(tenant_id, namespace)
        if not vector_ids:
            return 0

        internal_ids: list[int] = []
        for vector_id in vector_ids:
            internal_id = state.vector_id_to_internal.pop(vector_id, None)
            if internal_id is None:
                continue
            state.internal_to_vector_id.pop(internal_id, None)
            internal_ids.append(internal_id)
            state.pending_records = [r for r in state.pending_records if r.vector_id != vector_id]

        removed = self._remove_internal_ids(state, internal_ids)
        if internal_ids:
            state.payload_store.delete(vector_ids)
        if removed:
            self._persist_index(state)
        return removed

    def create_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        destination: str,
        include_payloads: bool = True,
    ) -> SnapshotInfo:
        state = self._get_state(tenant_id, namespace)
        path = self._resolve_snapshot_path(destination, tenant_id, namespace)
        vector_ids = list(state.vector_id_to_internal.keys())
        vectors = state.payload_store.fetch_vectors(vector_ids)
        metadata = state.payload_store.fetch_metadata(vector_ids)
        records: list[dict[str, object]] = []
        for vector_id in vector_ids:
            record: dict[str, object] = {
                "vector_id": vector_id,
                "metadata": metadata.get(vector_id, {}),
            }
            if include_payloads and vector_id in vectors:
                record["values"] = vectors[vector_id].tolist()
            records.append(record)
        payload = {
            "tenant_id": tenant_id,
            "namespace": namespace,
            "created_at": time.time(),
            "params": asdict(state.params),
            "compression": asdict(state.compression),
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
        params = IndexParams(**payload["params"])
        compression = CompressionPolicy(**payload.get("compression", {}))
        tenant_state = self._tenants.setdefault(tenant_id, {})
        if overwrite:
            tenant_state.pop(namespace, None)
        self.create_or_update_collection(
            tenant_id=tenant_id,
            namespace=namespace,
            params=params,
            compression=compression,
            metadata=None,
            named_vectors=None,
        )
        records_payload = payload.get("records", [])
        records: list[VectorRecord] = []
        for record in records_payload:
            values = record.get("values")
            if not values:
                continue
            records.append(
                VectorRecord(
                    vector_id=str(record["vector_id"]),
                    values=values,
                    metadata=record.get("metadata", {}),
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
        state = self._get_state(tenant_id, namespace)
        vector_ids = list(state.vector_id_to_internal.keys())
        vectors = state.payload_store.fetch_vectors(vector_ids)
        metadata = state.payload_store.fetch_metadata(vector_ids)
        params = state.params
        compression = state.compression
        if force and state.index_path.exists():
            state.index_path.unlink(missing_ok=True)
        tenant_state = self._tenants.setdefault(tenant_id, {})
        tenant_state.pop(namespace, None)
        self.create_or_update_collection(
            tenant_id=tenant_id,
            namespace=namespace,
            params=params,
            compression=compression,
            metadata=None,
            named_vectors=None,
        )
        rebuilt_state = self._get_state(tenant_id, namespace)
        if not vector_ids:
            return RebuildReport(namespace=namespace, rebuilt=True, details={"vectors": 0})
        rebuilt_state.payload_store.delete(vector_ids)
        rebuilt_state.vector_id_to_internal.clear()
        rebuilt_state.internal_to_vector_id.clear()
        rebuilt_state.next_internal_id = 1
        records = [
            VectorRecord(
                vector_id=vector_id,
                values=vectors[vector_id].tolist(),
                metadata=metadata.get(vector_id, {}),
            )
            for vector_id in vector_ids
            if vector_id in vectors
        ]
        if records:
            self.upsert(tenant_id=tenant_id, namespace=namespace, records=records)
        return RebuildReport(
            namespace=namespace,
            rebuilt=True,
            details={"vectors": len(records)},
        )

    def check_health(
        self,
        *,
        tenant_id: str,
        namespace: str | None = None,
    ) -> Mapping[str, HealthStatus]:
        tenant_state = self._tenants.get(tenant_id, {})
        if namespace is not None:
            state = tenant_state.get(namespace)
            if not state:
                return {namespace: HealthStatus(name=namespace, healthy=False, details={})}
            return {
                namespace: HealthStatus(
                    name=namespace,
                    healthy=not state.requires_training or state.cpu_index.is_trained,
                    details={
                        "vectors": len(state.vector_id_to_internal),
                        "trained": state.cpu_index.is_trained,
                    },
                )
            }
        statuses: dict[str, HealthStatus] = {}
        for name, state in tenant_state.items():
            statuses[name] = HealthStatus(
                name=name,
                healthy=not state.requires_training or state.cpu_index.is_trained,
                details={
                    "vectors": len(state.vector_id_to_internal),
                    "trained": state.cpu_index.is_trained,
                },
            )
        return statuses

    def _resolve_snapshot_path(self, destination: str, tenant_id: str, namespace: str) -> Path:
        path = Path(destination)
        if path.suffix:
            return path
        return path / f"{tenant_id}__{namespace}.faiss-snapshot.json"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_state(self, tenant_id: str, namespace: str) -> _NamespaceState:
        tenant_state = self._tenants.get(tenant_id)
        if not tenant_state or namespace not in tenant_state:
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id)
        return tenant_state[namespace]

    def _build_index(
        self,
        namespace: str,
        params: IndexParams,
        compression: CompressionPolicy,
    ) -> faiss.Index:
        metric = _faiss_metric(params.metric)
        kind = params.kind.lower()

        if kind == "flat":
            factory = _flat_factory_string(compression)
            base = faiss.index_factory(params.dimension, factory, metric)
        elif kind == "ivf_flat":
            if params.nlist is None:
                raise InvalidNamespaceConfigError(
                    namespace, detail="nlist is required for IVF indexes"
                )
            factory = f"IVF{params.nlist},{_flat_factory_string(compression)}"
            base = faiss.index_factory(params.dimension, factory, metric)
        elif kind == "ivf_pq":
            if params.nlist is None:
                raise InvalidNamespaceConfigError(
                    namespace, detail="nlist is required for IVF_PQ indexes"
                )
            if compression.pq_m is None or compression.pq_nbits is None:
                raise InvalidNamespaceConfigError(
                    namespace, detail="PQ configuration requires pq_m and pq_nbits"
                )
            factory = f"IVF{params.nlist},PQ{compression.pq_m}x{compression.pq_nbits}"
            base = faiss.index_factory(params.dimension, factory, metric)
        elif kind == "opq_ivf_pq":
            if params.nlist is None:
                raise InvalidNamespaceConfigError(
                    namespace, detail="nlist is required for OPQ IVF_PQ indexes"
                )
            if compression.pq_m is None or compression.pq_nbits is None:
                raise InvalidNamespaceConfigError(
                    namespace, detail="OPQ+PQ requires pq_m and pq_nbits"
                )
            opq_m = compression.opq_m or compression.pq_m
            factory = f"OPQ{opq_m},IVF{params.nlist},PQ{compression.pq_m}x{compression.pq_nbits}"
            base = faiss.index_factory(params.dimension, factory, metric)
        elif kind == "hnsw":
            m = params.m or 32
            base = faiss.IndexHNSWFlat(params.dimension, m)
            base.hnsw.efConstruction = params.ef_construct or base.hnsw.efConstruction
            base.hnsw.efSearch = params.ef_search or base.hnsw.efSearch
            if metric == faiss.METRIC_INNER_PRODUCT:
                base.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise InvalidNamespaceConfigError(
                namespace, detail=f"Unsupported FAISS index kind '{params.kind}'"
            )

        return faiss.IndexIDMap2(base)

    def _train_if_ready(self, state: _NamespaceState) -> None:
        if not state.requires_training:
            return
        if state.cpu_index.is_trained:
            state.requires_training = False
            state.training_buffer.clear()
            return
        if len(state.training_buffer) < max(state.training_threshold, 1):
            return
        train_data = np.vstack(state.training_buffer)
        LOGGER.debug("Training FAISS index with %s vectors", train_data.shape[0])
        state.cpu_index.train(train_data)
        state.requires_training = False
        state.training_buffer.clear()

    def _flush_pending(self, state: _NamespaceState) -> None:
        if state.requires_training and not state.cpu_index.is_trained:
            return
        if not state.pending_records:
            return
        vectors = np.vstack([record.vector for record in state.pending_records]).astype(np.float32)
        ids = np.asarray([record.internal_id for record in state.pending_records], dtype=np.int64)
        state.cpu_index.add_with_ids(vectors, ids)
        state.pending_records.clear()
        state.reset_gpu()

    def _persist_index(self, state: _NamespaceState) -> None:
        state.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(state.cpu_index, str(state.index_path))

    def _remove_internal_ids(self, state: _NamespaceState, internal_ids: Sequence[int]) -> int:
        if not internal_ids:
            return 0
        selector = faiss.IDSelectorArray(np.asarray(internal_ids, dtype=np.int64))
        removed = int(state.cpu_index.remove_ids(selector))
        if removed:
            state.reset_gpu()
        return removed

    def _select_index(self, state: _NamespaceState) -> faiss.Index:
        if not state.params.use_gpu:
            return state.cpu_index
        if faiss.get_num_gpus() <= 0:
            return state.cpu_index
        if state.gpu_index is None:
            if state.gpu_resources is None:
                state.gpu_resources = faiss.StandardGpuResources()
            state.gpu_index = faiss.index_cpu_to_gpu(
                state.gpu_resources,
                state.params.gpu_id or 0,
                faiss.clone_index(state.cpu_index),
            )
        return state.gpu_index

    def _apply_search_params(self, index: faiss.Index, state: _NamespaceState) -> None:
        base = _unwrap_index(index)
        if hasattr(base, "nprobe") and state.params.nprobe:
            try:
                base.nprobe = state.params.nprobe
            except Exception:  # pragma: no cover - faiss handles compatibility
                LOGGER.debug("Failed to set nprobe on index", exc_info=True)
        if isinstance(base, faiss.IndexHNSW) and state.params.ef_search:
            base.hnsw.efSearch = state.params.ef_search

    def _resolve_search_k(
        self,
        state: _NamespaceState,
        top_k: int,
        query_reorder: bool | None,
    ) -> tuple[int, bool]:
        reorder = state.reorder_enabled
        if query_reorder is True:
            reorder = True
        elif query_reorder is False:
            reorder = False

        if reorder:
            base = state.reorder_k or max(2 * top_k, top_k)
            search_k = max(top_k, base)
        else:
            search_k = top_k
        return search_k, reorder

    def _should_enable_reorder(self, params: IndexParams, compression: CompressionPolicy) -> bool:
        if params.kind.lower() in {"ivf_pq", "opq_ivf_pq"}:
            return True
        return compression.kind in {"pq", "opq_pq"}


def _flat_factory_string(compression: CompressionPolicy) -> str:
    match compression.kind:
        case "scalar_int8":
            return "SQ8"
        case "fp16":
            return "SQfp16"
        case _:
            return "Flat"


def _should_normalize(metric: str) -> bool:
    return metric.lower() == "cosine"


def _faiss_metric(metric: str) -> int:
    normalized = metric.lower()
    if normalized in {"cosine", "ip", "dot", "inner_product"}:
        return faiss.METRIC_INNER_PRODUCT
    if normalized in {"l2", "euclidean"}:
        return faiss.METRIC_L2
    raise ValueError(f"Unsupported FAISS metric '{metric}'")


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        return vector
    return vector / norm


def _index_dimension(index: faiss.Index) -> int:
    base = _unwrap_index(index)
    return int(base.d)


def _unwrap_index(index: faiss.Index) -> faiss.Index:
    current = index
    while isinstance(current, (faiss.IndexIDMap2, faiss.IndexPreTransform)):
        if isinstance(current, faiss.IndexIDMap2):
            current = current.index
        else:
            current = current.index
    return faiss.downcast_index(current)


def _training_threshold(params: IndexParams, compression: CompressionPolicy) -> int:
    if params.train_size:
        return params.train_size
    kind = params.kind.lower()
    if kind.startswith("ivf") or compression.kind in {"scalar_int8", "fp16", "pq", "opq_pq"}:
        nlist = params.nlist or 1
        return max(params.dimension * 4, nlist * 32)
    return 0


def _distance_to_score(distance: float, metric: str) -> float:
    normalized = metric.lower()
    if normalized in {"l2", "euclidean"}:
        return -float(distance)
    return float(distance)


def _score_vectors(
    query: np.ndarray,
    stored: np.ndarray | None,
    metric: str,
    normalize: bool,
) -> float:
    if stored is None:
        return float("-inf")
    stored_vec = stored.astype(np.float32)
    if normalize:
        stored_vec = _normalize(stored_vec)
        query_vec = _normalize(query.astype(np.float32))
    else:
        query_vec = query.astype(np.float32)

    normalized = metric.lower()
    if normalized in {"l2", "euclidean"}:
        return -float(np.linalg.norm(query_vec - stored_vec))
    return float(np.dot(query_vec, stored_vec))


__all__ = ["FaissVectorStore"]
