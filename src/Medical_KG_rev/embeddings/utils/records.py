"""Helper utilities for building normalized embedding records."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest

MetadataSource = Sequence[dict[str, Any]] | dict[str, Any] | Callable[[int], dict[str, Any]] | None


def _resolve_ids(request: EmbeddingRequest, count: int) -> list[str]:
    provided = list(request.ids or [])
    if len(provided) >= count:
        return provided[:count]
    generated = [f"{request.namespace}:{index}" for index in range(len(provided), count)]
    return [*provided, *generated]


def _normalize_ids(ids: Sequence[str] | None, request: EmbeddingRequest, count: int) -> list[str]:
    if ids is None:
        return _resolve_ids(request, count)
    normalized = list(ids[:count])
    while len(normalized) < count:
        normalized.append(f"{request.namespace}:{len(normalized)}")
    return normalized


def _resolve_metadata(request: EmbeddingRequest, count: int) -> list[dict[str, Any]]:
    base = [dict(item) for item in list(request.metadata or [])[:count]]
    while len(base) < count:
        base.append({})
    return base


def _expand_extra(extra: MetadataSource, count: int) -> list[dict[str, Any]]:
    if extra is None:
        return [{} for _ in range(count)]
    if callable(extra):  # type: ignore[callable-returns-bool]
        return [dict(extra(index)) for index in range(count)]
    if isinstance(extra, dict):
        return [dict(extra) for _ in range(count)]
    values = [dict(item) for item in list(extra)[:count]]
    while len(values) < count:
        values.append({})
    return values


def _resolve_offsets(
    offsets: Sequence[Sequence[Any]] | None, count: int
) -> list[Sequence[Any] | None]:
    if offsets is None:
        return [None] * count
    values: list[Sequence[Any] | None] = list(offsets[:count])
    while len(values) < count:
        values.append(None)
    return values


def _as_float_vector(vector: Sequence[Any]) -> list[float]:
    return [float(value) for value in vector]


@dataclass(slots=True)
class RecordBuilder:
    """Factory for normalized :class:`EmbeddingRecord` instances."""

    config: EmbedderConfig
    normalized_override: bool | None = None

    @property
    def _normalized(self) -> bool:
        if self.normalized_override is not None:
            return bool(self.normalized_override)
        return bool(self.config.normalize)

    def dense(
        self,
        request: EmbeddingRequest,
        vectors: Sequence[Sequence[Any]],
        *,
        dim: int | None = None,
        offsets: Sequence[Sequence[Any]] | None = None,
        extra_metadata: MetadataSource = None,
        ids: Sequence[str] | None = None,
    ) -> list[EmbeddingRecord]:
        float_vectors = [_as_float_vector(vector) for vector in vectors]
        count = len(float_vectors)
        resolved_ids = _normalize_ids(ids, request, count)
        metadata = _resolve_metadata(request, count)
        extras = _expand_extra(extra_metadata, count)
        offset_list = _resolve_offsets(offsets, count)
        records: list[EmbeddingRecord] = []
        for chunk_id, vector, base_meta, extra_meta, offset in zip(
            resolved_ids, float_vectors, metadata, extras, offset_list, strict=False
        ):
            payload = {"provider": self.config.provider, **base_meta, **extra_meta}
            if offset is not None:
                payload["offsets"] = offset
            records.append(
                EmbeddingRecord(
                    id=chunk_id,
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id=self.config.model_id,
                    model_version=self.config.model_version,
                    kind=self.config.kind,
                    dim=dim or (len(vector) if vector else None),
                    vectors=[vector],
                    normalized=self._normalized,
                    metadata=payload,
                    correlation_id=request.correlation_id,
                )
            )
        return records

    def multi_vector(
        self,
        request: EmbeddingRequest,
        vector_groups: Sequence[Sequence[Sequence[Any]]],
        *,
        dim: int | None = None,
        extra_metadata: MetadataSource = None,
        ids: Sequence[str] | None = None,
    ) -> list[EmbeddingRecord]:
        converted = [[_as_float_vector(vector) for vector in group] for group in vector_groups]
        count = len(converted)
        resolved_ids = _normalize_ids(ids, request, count)
        metadata = _resolve_metadata(request, count)
        extras = _expand_extra(extra_metadata, count)
        records: list[EmbeddingRecord] = []
        for chunk_id, vectors, base_meta, extra_meta in zip(
            resolved_ids, converted, metadata, extras, strict=False
        ):
            payload = {"provider": self.config.provider, **base_meta, **extra_meta}
            dimension = dim or (len(vectors[0]) if vectors else self.config.dim)
            records.append(
                EmbeddingRecord(
                    id=chunk_id,
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id=self.config.model_id,
                    model_version=self.config.model_version,
                    kind=self.config.kind,
                    dim=dimension,
                    vectors=vectors,
                    normalized=self._normalized,
                    metadata=payload,
                    correlation_id=request.correlation_id,
                )
            )
        return records

    def sparse(
        self,
        request: EmbeddingRequest,
        term_weights: Sequence[dict[str, float]],
        *,
        extra_metadata: MetadataSource = None,
        dim_from_terms: bool = True,
        ids: Sequence[str] | None = None,
    ) -> list[EmbeddingRecord]:
        count = len(term_weights)
        resolved_ids = _normalize_ids(ids, request, count)
        metadata = _resolve_metadata(request, count)
        extras = _expand_extra(extra_metadata, count)
        records: list[EmbeddingRecord] = []
        for chunk_id, weights, base_meta, extra_meta in zip(
            resolved_ids, term_weights, metadata, extras, strict=False
        ):
            payload = {"provider": self.config.provider, **base_meta, **extra_meta}
            dimension = len(weights) if dim_from_terms else self.config.dim
            records.append(
                EmbeddingRecord(
                    id=chunk_id,
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id=self.config.model_id,
                    model_version=self.config.model_version,
                    kind=self.config.kind,
                    dim=dimension,
                    terms=dict(weights),
                    normalized=self._normalized,
                    metadata=payload,
                    correlation_id=request.correlation_id,
                )
            )
        return records

    def neural_sparse(
        self,
        request: EmbeddingRequest,
        vectors: Sequence[Sequence[Any]],
        *,
        field_name: str,
        dim: int,
        extra_metadata: MetadataSource = None,
        ids: Sequence[str] | None = None,
    ) -> list[EmbeddingRecord]:
        float_vectors = [_as_float_vector(vector) for vector in vectors]
        count = len(float_vectors)
        resolved_ids = _normalize_ids(ids, request, count)
        metadata = _resolve_metadata(request, count)
        extras = _expand_extra(extra_metadata, count)
        records: list[EmbeddingRecord] = []
        for chunk_id, vector, base_meta, extra_meta in zip(
            resolved_ids, float_vectors, metadata, extras, strict=False
        ):
            payload = {"provider": self.config.provider, **base_meta, **extra_meta}
            records.append(
                EmbeddingRecord(
                    id=chunk_id,
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id=self.config.model_id,
                    model_version=self.config.model_version,
                    kind=self.config.kind,
                    dim=dim,
                    neural_fields={field_name: vector},
                    vectors=[vector],
                    normalized=self._normalized,
                    metadata=payload,
                    correlation_id=request.correlation_id,
                )
            )
        return records
