"""In-memory vector store implementation for testing purposes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from ..errors import DimensionMismatchError, NamespaceNotFoundError, ResourceExhaustedError
from ..models import CompressionPolicy, IndexParams, VectorMatch, VectorQuery, VectorRecord
from ..types import VectorStorePort

_MAX_VECTORS_PER_NAMESPACE: Final[int] = 50_000


@dataclass(slots=True)
class _NamespaceState:
    params: IndexParams
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    named_params: dict[str, IndexParams] = field(default_factory=dict)
    named_vectors: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    metadata: dict[str, Mapping[str, object]] = field(default_factory=dict)


class InMemoryVectorStore(VectorStorePort):
    """Simple numpy-backed store enforcing per-tenant isolation."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, _NamespaceState]] = {}

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
        if existing:
            if existing.params.dimension != params.dimension:
                raise DimensionMismatchError(
                    existing.params.dimension, params.dimension, namespace=namespace
                )
            if named_vectors:
                for name, vector_params in named_vectors.items():
                    existing_params = existing.named_params.get(name)
                    if existing_params and existing_params.dimension != vector_params.dimension:
                        raise DimensionMismatchError(
                            existing_params.dimension,
                            vector_params.dimension,
                            namespace=f"{namespace}:{name}",
                        )
            return
        named_params_map = dict(named_vectors or {})
        tenant_state[namespace] = _NamespaceState(
            params=params,
            named_params=named_params_map,
            named_vectors={name: {} for name in named_params_map},
        )
        if metadata:
            tenant_state[namespace].metadata["__collection__"] = dict(metadata)

    def list_collections(self, *, tenant_id: str) -> Sequence[str]:
        return list(self._state.get(tenant_id, {}).keys())

    def upsert(
        self,
        *,
        tenant_id: str,
        namespace: str,
        records: Sequence[VectorRecord],
    ) -> None:
        tenant_state = self._state.setdefault(tenant_id, {})
        state = tenant_state.get(namespace)
        if not state:
            raise NamespaceNotFoundError(namespace, tenant_id=tenant_id)
        if len(state.vectors) + len(records) > _MAX_VECTORS_PER_NAMESPACE:
            raise ResourceExhaustedError(namespace)
        for record in records:
            if record.values:
                array = np.asarray(record.values, dtype=float)
                if array.shape != (state.params.dimension,):
                    raise DimensionMismatchError(
                        state.params.dimension, array.shape[0], namespace=namespace
                    )
                state.vectors[record.vector_id] = array
            if record.named_vectors:
                for name, values in record.named_vectors.items():
                    params = state.named_params.get(name)
                    if not params:
                        raise NamespaceNotFoundError(f"{namespace}:{name}", tenant_id=tenant_id)
                    array = np.asarray(values, dtype=float)
                    if array.shape != (params.dimension,):
                        raise DimensionMismatchError(
                            params.dimension,
                            array.shape[0],
                            namespace=f"{namespace}:{name}",
                        )
                    state.named_vectors.setdefault(name, {})[record.vector_id] = array
            state.metadata[record.vector_id] = dict(record.metadata)

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
        if query.vector_name:
            named_vectors = state.named_vectors.get(query.vector_name)
            if not named_vectors:
                return []
            params = state.named_params.get(query.vector_name)
            if not params:
                return []
            query_vector = np.asarray(query.values, dtype=float)
            if query_vector.shape != (params.dimension,):
                raise DimensionMismatchError(
                    params.dimension,
                    query_vector.shape[0],
                    namespace=f"{namespace}:{query.vector_name}",
                )
            if not named_vectors:
                return []
            matrix = np.stack(list(named_vectors.values()))
            vector_ids = list(named_vectors.keys())
        else:
            query_vector = np.asarray(query.values, dtype=float)
            if query_vector.shape != (state.params.dimension,):
                raise DimensionMismatchError(
                    state.params.dimension, query_vector.shape[0], namespace=namespace
                )
            if not state.vectors:
                return []
            matrix = np.stack(list(state.vectors.values()))
            vector_ids = list(state.vectors.keys())
        scores = matrix @ query_vector
        ranked = np.argsort(scores)[::-1][: query.top_k]
        results: list[VectorMatch] = []
        for idx in ranked:
            vector_id = vector_ids[idx]
            results.append(
                VectorMatch(
                    vector_id=vector_id,
                    score=float(scores[idx]),
                    metadata=state.metadata.get(vector_id, {}),
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
        tenant_state = self._state.get(tenant_id, {})
        state = tenant_state.get(namespace)
        if not state:
            return 0
        removed = 0
        for vector_id in vector_ids:
            if vector_id in state.vectors:
                state.vectors.pop(vector_id, None)
                state.metadata.pop(vector_id, None)
                removed += 1
            for named in state.named_vectors.values():
                named.pop(vector_id, None)
        return removed
