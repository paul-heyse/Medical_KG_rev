"""Qdrant-backed implementation of :class:`VectorStorePort`."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from ..errors import HttpClient
from ..models import HttpClient
from ..types import VectorStorePort



@dataclass(slots=True)
class _NamespaceOptions:
    params: IndexParams
    compression: CompressionPolicy
    reorder_final: bool = False
    named_vectors: Mapping[str, IndexParams] | None = None


class QdrantVectorStore(VectorStorePort):
    """Concrete vector store adapter backed by Qdrant."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        *,
        default_url: str | None = None,
        prefer_grpc: bool = False,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any]
            if default_url:
                kwargs = {"url": default_url, "prefer_grpc": prefer_grpc}
            else:
                kwargs = {"host": "localhost", "port": 6333, "prefer_grpc": prefer_grpc}
            self._client = QdrantClient(**kwargs)
        self._tenant_namespaces: dict[str, set[str]] = defaultdict(set)
        self._namespace_options: dict[tuple[str, str], _NamespaceOptions] = {}

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
        self._tenant_namespaces[tenant_id].add(namespace)
        metadata = metadata or {}
        options = _NamespaceOptions(
            params=params,
            compression=compression,
            reorder_final=bool((metadata.get("search") or {}).get("reorder_final")),
            named_vectors=named_vectors,
        )
        self._namespace_options[(tenant_id, namespace)] = options

        vectors_config = self._build_vectors_config(params, named_vectors)
        hnsw_config = self._build_hnsw_config(params)
        optimizers_config = self._build_optimizers_config(metadata)
        quantization_config = self._build_quantization(compression)
        replication_factor = params.replicas

        try:
            info = self._client.get_collection(collection_name=namespace)
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                self._client.recreate_collection(
                    collection_name=namespace,
                    vectors_config=vectors_config,
                    replication_factor=replication_factor,
                    hnsw_config=hnsw_config,
                    optimizers_config=optimizers_config,
                    quantization_config=quantization_config,
                )
                return
            raise BackendUnavailableError("Failed to inspect Qdrant collection") from exc

        self._validate_existing_collection(info, tenant_id, namespace, params, named_vectors)
        self._client.update_collection(
            collection_name=namespace,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            quantization_config=quantization_config,
        )

    def list_collections(self, *, tenant_id: str) -> Sequence[str]:
        namespaces = self._tenant_namespaces.get(tenant_id)
        if not namespaces:
            return []
        return sorted(namespaces)

    def upsert(
        self,
        *,
        tenant_id: str,
        namespace: str,
        records: Sequence[VectorRecord],
    ) -> None:
        self._ensure_namespace_known(tenant_id, namespace)
        if not records:
            return
        try:
            points = [self._to_point(record) for record in records]
            self._client.upsert(collection_name=namespace, points=points)
        except UnexpectedResponse as exc:
            self._raise_for_error(exc, namespace, tenant_id)

    def query(
        self,
        *,
        tenant_id: str,
        namespace: str,
        query: VectorQuery,
    ) -> Sequence[VectorMatch]:
        self._ensure_namespace_known(tenant_id, namespace)
        options = self._namespace_options.get((tenant_id, namespace))
        search_params = self._build_search_params(options, query)
        qdrant_filter = self._build_filter(query.filters)
        qdrant_query = self._build_query_vector(query)
        try:
            matches = self._client.search(
                collection_name=namespace,
                query_vector=qdrant_query,
                limit=query.top_k,
                query_filter=qdrant_filter,
                search_params=search_params,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            self._raise_for_error(exc, namespace, tenant_id)
        results: list[VectorMatch] = []
        for match in matches:
            payload = match.payload or {}
            results.append(
                VectorMatch(
                    vector_id=str(match.id),
                    score=float(match.score),
                    metadata=payload,
                )
            )
        return results

    def delete(
        self,
        *,
        tenant_id: str,
        namespace: str,
        vector_ids: Sequence[str],
    ) -> int:
        self._ensure_namespace_known(tenant_id, namespace)
        if not vector_ids:
            return 0
        selector = qm.PointIdsList(points=list(vector_ids))
        try:
            result = self._client.delete(collection_name=namespace, points_selector=selector)
        except UnexpectedResponse as exc:
            self._raise_for_error(exc, namespace, tenant_id)
        return getattr(result, "deleted", len(vector_ids))

    def create_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        destination: str,
        include_payloads: bool = True,
    ) -> SnapshotInfo:
        self._ensure_namespace_known(tenant_id, namespace)
        path = self._resolve_snapshot_path(destination, tenant_id, namespace)
        try:
            snapshot = self._client.create_snapshot(collection_name=namespace)
        except UnexpectedResponse as exc:
            raise BackendUnavailableError("Failed to create Qdrant snapshot") from exc
        metadata = {
            "name": getattr(snapshot, "name", None),
            "size": getattr(snapshot, "size", None),
            "location": getattr(snapshot, "location", None),
            "include_payloads": include_payloads,
        }
        payload = {
            "tenant_id": tenant_id,
            "namespace": namespace,
            "created_at": time.time(),
            "snapshot": metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True))
        stats = path.stat()
        return SnapshotInfo(
            namespace=namespace,
            path=str(path),
            size_bytes=stats.st_size,
            created_at=time.time(),
            metadata=metadata,
        )

    def restore_snapshot(
        self,
        *,
        tenant_id: str,
        namespace: str,
        source: str,
        overwrite: bool = False,
    ) -> RebuildReport:
        # Qdrant snapshots are immutable; overwrite flag is informational
        self._ensure_namespace_known(tenant_id, namespace)
        payload = json.loads(Path(source).read_text())
        metadata = payload.get("snapshot", {})
        try:
            self._client.recover_snapshot(
                collection_name=namespace,
                location=metadata.get("location"),
                snapshot_name=metadata.get("name"),
            )
        except UnexpectedResponse as exc:
            raise BackendUnavailableError("Failed to restore Qdrant snapshot") from exc
        return RebuildReport(namespace=namespace, rebuilt=True, details=metadata)

    def rebuild_index(
        self,
        *,
        tenant_id: str,
        namespace: str,
        force: bool = False,
    ) -> RebuildReport:
        self._ensure_namespace_known(tenant_id, namespace)
        try:
            self._client.update_collection(
                collection_name=namespace,
                optimizer_config=qm.OptimizersConfigDiff(indexing_threshold=0 if force else None),
            )
        except UnexpectedResponse as exc:
            raise BackendUnavailableError("Failed to trigger Qdrant index rebuild") from exc
        return RebuildReport(namespace=namespace, rebuilt=True, details={"force": force})

    def check_health(
        self,
        *,
        tenant_id: str,
        namespace: str | None = None,
    ) -> Mapping[str, HealthStatus]:
        namespaces = (
            [namespace] if namespace else sorted(self._tenant_namespaces.get(tenant_id, set()))
        )
        statuses: dict[str, HealthStatus] = {}
        for ns in namespaces:
            healthy = False
            details: dict[str, object] = {}
            try:
                info = self._client.get_collection(collection_name=ns)
            except UnexpectedResponse as exc:
                details["error"] = getattr(exc, "reason_phrase", str(exc))
            else:
                healthy = True
                details["vector_count"] = getattr(info, "points_count", None)
            statuses[ns] = HealthStatus(name=ns, healthy=healthy, details=details)
        return statuses

    def _resolve_snapshot_path(self, destination: str, tenant_id: str, namespace: str) -> Path:
        path = Path(destination)
        if path.suffix:
            return path
        return path / f"{tenant_id}__{namespace}.qdrant-snapshot.json"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_namespace_known(self, tenant_id: str, namespace: str) -> None:
        if namespace not in self._tenant_namespaces.get(tenant_id, set()):
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id)

    def _build_vectors_config(
        self,
        params: IndexParams,
        named_vectors: Mapping[str, IndexParams] | None,
    ) -> qm.VectorsConfig:
        if named_vectors:
            mapping: dict[str, qm.VectorParams] = {}
            for name, vector_params in named_vectors.items():
                mapping[name] = qm.VectorParams(
                    size=vector_params.dimension,
                    distance=self._to_distance(vector_params.metric or params.metric),
                    hnsw_config=self._build_hnsw_config(vector_params),
                )
            return mapping
        return qm.VectorParams(
            size=params.dimension,
            distance=self._to_distance(params.metric),
            hnsw_config=self._build_hnsw_config(params),
        )

    def _build_hnsw_config(self, params: IndexParams) -> qm.HnswConfigDiff | None:
        if params.kind.lower() != "hnsw":
            return None
        return qm.HnswConfigDiff(
            ef_construct=params.ef_construct,
            m=params.m,
            full_scan_threshold=0,
        )

    def _build_optimizers_config(
        self, metadata: Mapping[str, object]
    ) -> qm.OptimizersConfigDiff | None:
        gpu_config = metadata.get("gpu") if metadata else None
        if not isinstance(gpu_config, Mapping):
            return None
        if not gpu_config.get("enabled"):
            return None
        indexing_threshold = int(gpu_config.get("indexing_threshold", 20_000))
        return qm.OptimizersConfigDiff(indexing_threshold=indexing_threshold)

    def _build_quantization(self, compression: CompressionPolicy) -> qm.QuantizationConfig | None:
        kind = compression.kind.lower()
        if kind == "scalar_int8":
            return qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
        if kind in {"binary", "bq"}:
            config_kwargs: dict[str, object] = {"always_ram": True}
            if hasattr(qm, "BinaryQuantizationEncoding"):
                config_kwargs["encoding"] = qm.BinaryQuantizationEncoding.ONE_BIT
            return qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(**config_kwargs))
        if kind in {"pq", "opq_pq"}:
            ratio = self._infer_pq_ratio(compression)
            return qm.ProductQuantization(
                product=qm.ProductQuantizationConfig(
                    compression=ratio,
                    always_ram=True,
                )
            )
        return None

    def _infer_pq_ratio(self, compression: CompressionPolicy) -> qm.CompressionRatio:
        mapping = {
            4: qm.CompressionRatio.X4,
            8: qm.CompressionRatio.X8,
            16: qm.CompressionRatio.X16,
            32: qm.CompressionRatio.X32,
            64: qm.CompressionRatio.X64,
        }
        if compression.pq_m and compression.pq_m in mapping:
            return mapping[compression.pq_m]
        if compression.pq_nbits and compression.pq_nbits in mapping:
            return mapping[compression.pq_nbits]
        return qm.CompressionRatio.X16

    def _validate_existing_collection(
        self,
        info: Any,
        tenant_id: str,
        namespace: str,
        params: IndexParams,
        named_vectors: Mapping[str, IndexParams] | None,
    ) -> None:
        vectors = getattr(getattr(info, "config", None), "params", None)
        if vectors is None:
            return
        vectors = getattr(vectors, "vectors", vectors)
        if hasattr(vectors, "size"):
            size = vectors.size
            if size != params.dimension:
                raise DimensionMismatchError(size, params.dimension, namespace=namespace)
            return
        if isinstance(vectors, Mapping) and named_vectors:
            for name, definition in named_vectors.items():
                existing = vectors.get(name)
                if not existing:
                    raise NamespaceNotFoundError(f"{namespace}:{name}", tenant_id=tenant_id)
                size = getattr(existing, "size", None)
                if size is not None and size != definition.dimension:
                    raise DimensionMismatchError(
                        size,
                        definition.dimension,
                        namespace=f"{namespace}:{name}",
                    )

    def _build_search_params(
        self,
        options: _NamespaceOptions | None,
        query: VectorQuery,
    ) -> qm.SearchParams | None:
        if options is None and query.reorder is None:
            return None
        reorder = (
            query.reorder
            if query.reorder is not None
            else (options.reorder_final if options else False)
        )
        quantization = qm.QuantizationSearchParams(rescore=reorder) if reorder else None
        ef_search = None
        if options and options.params.ef_search:
            ef_search = options.params.ef_search
        if quantization or ef_search:
            return qm.SearchParams(hnsw_ef=ef_search, quantization=quantization)
        return None

    def _build_filter(self, filters: Mapping[str, object] | None) -> qm.Filter | None:
        if not filters:
            return None
        must: list[qm.FieldCondition] = []
        for key, value in filters.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=list(value))))
            else:
                must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
        return qm.Filter(must=must) if must else None

    def _build_query_vector(self, query: VectorQuery) -> Any:
        if query.vector_name:
            return (query.vector_name, list(query.values))
        return list(query.values)

    def _to_point(self, record: VectorRecord) -> qm.PointStruct:
        payload = dict(record.metadata)
        if record.vector_version:
            payload["vector_version"] = record.vector_version
        vector: Any
        if record.named_vectors:
            vector = {name: list(values) for name, values in record.named_vectors.items()}
        else:
            vector = list(record.values)
        return qm.PointStruct(id=record.vector_id, vector=vector, payload=payload)

    def _raise_for_error(self, error: UnexpectedResponse, namespace: str, tenant_id: str) -> None:
        status = getattr(error, "status_code", None)
        if status == 404:
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id) from error
        if status in {413, 507}:
            raise ResourceExhaustedError(namespace) from error
        raise BackendUnavailableError(f"Qdrant error ({status})") from error

    def _to_distance(self, metric: str | None) -> qm.Distance:
        normalized = (metric or "cosine").lower()
        if normalized in {"cos", "cosine"}:
            return qm.Distance.COSINE
        if normalized in {"l2", "euclidean"}:
            return qm.Distance.EUCLID
        if normalized in {"dot", "inner_product", "ip"}:
            return qm.Distance.DOT
        return qm.Distance.COSINE
