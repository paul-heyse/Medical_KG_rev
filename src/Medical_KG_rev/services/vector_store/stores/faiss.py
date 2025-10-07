"""Adapter that wraps the lightweight FAISSIndex used in tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex

from ..errors import DimensionMismatchError, InvalidNamespaceConfigError, NamespaceNotFoundError
from ..models import CompressionPolicy, IndexParams, VectorMatch, VectorQuery, VectorRecord
from ..types import VectorStorePort


@dataclass(slots=True)
class _TenantNamespace:
    index: FAISSIndex


class FaissVectorStore(VectorStorePort):
    """Wraps FAISSIndex to conform to :class:`VectorStorePort`."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, _TenantNamespace]] = {}

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
        tenant_state = self._state.setdefault(tenant_id, {})
        existing = tenant_state.get(namespace)
        if existing and existing.index.dimension != params.dimension:
            raise DimensionMismatchError(
                existing.index.dimension, params.dimension, namespace=namespace
            )
        if named_vectors:
            raise InvalidNamespaceConfigError(
                namespace,
                detail="FAISS adapter does not support named vectors.",
            )
        if existing:
            return
        tenant_state[namespace] = _TenantNamespace(index=FAISSIndex(dimension=params.dimension))

    def list_collections(self, *, tenant_id: str) -> Sequence[str]:
        return list(self._state.get(tenant_id, {}).keys())

    def upsert(
        self,
        *,
        tenant_id: str,
        namespace: str,
        records: Sequence[VectorRecord],
    ) -> None:
        tenant_state = self._state.get(tenant_id, {})
        state = tenant_state.get(namespace)
        if not state:
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id)
        for record in records:
            state.index.add(record.vector_id, record.values, record.metadata)

    def query(
        self,
        *,
        tenant_id: str,
        namespace: str,
        query: VectorQuery,
    ) -> Sequence[VectorMatch]:
        tenant_state = self._state.get(tenant_id, {})
        state = tenant_state.get(namespace)
        if not state:
            return []
        matches = state.index.search(query.values, k=query.top_k)
        return [VectorMatch(vector_id=vector_id, score=score, metadata=metadata) for vector_id, score, metadata in matches]

    def delete(
        self,
        *,
        tenant_id: str,
        namespace: str,
        vector_ids: Sequence[str],
    ) -> int:
        tenant_state = self._state.get(tenant_id, {})
        state = tenant_state.get(namespace)
        if not state:
            return 0
        removed = 0
        remaining_vectors = []
        remaining_ids = []
        remaining_metadata = []
        for vector_id, vector, metadata in zip(
            state.index.ids, state.index.vectors, state.index.metadata, strict=False
        ):
            if vector_id in vector_ids:
                removed += 1
                continue
            remaining_ids.append(vector_id)
            remaining_vectors.append(vector)
            remaining_metadata.append(metadata)
        state.index.ids = remaining_ids
        state.index.vectors = remaining_vectors
        state.index.metadata = remaining_metadata
        return removed
