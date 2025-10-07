"""Ingestion pipeline orchestration stages and helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import structlog

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.chunking import Chunk
from Medical_KG_rev.chunking.service import ChunkingOptions
from Medical_KG_rev.models.ir import Document
from Medical_KG_rev.observability.metrics import (
    observe_ingestion_stage_latency,
    observe_orchestration_stage,
    record_business_event,
    record_ingestion_document,
    record_ingestion_error,
)
from Medical_KG_rev.services.embedding import EmbeddingRequest, EmbeddingWorker
from Medical_KG_rev.services.ingestion import IngestionService
from Medical_KG_rev.services.vector_store.models import VectorRecord
from Medical_KG_rev.services.vector_store.service import VectorStoreService

from .pipeline import PipelineContext, PipelineStage, StageFailure
from .resilience import CircuitBreaker

logger = structlog.get_logger(__name__)

INGEST_CHUNKING_TOPIC = "ingest.chunking.v1"
INGEST_CHUNKS_TOPIC = "ingest.chunks.v1"
INGEST_EMBEDDINGS_TOPIC = "ingest.embeddings.v1"
INGEST_INDEXED_TOPIC = "ingest.indexed.v1"
INGEST_DLQ_TOPIC = "ingest.deadletter.v1"


def _coerce_document(payload: dict[str, Any]) -> Document:
    try:
        return Document.model_validate(payload)
    except Exception as exc:  # pragma: no cover - validation fallback
        raise StageFailure(
            "Invalid document payload",
            status=400,
            detail=str(exc),
            stage="chunking",
            error_type="validation",
        ) from exc


@dataclass(slots=True)
class ChunkingStage(PipelineStage):
    ingestion: IngestionService
    timeout_ms: int | None = 5000
    name: str = "chunking"
    circuit_breaker: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker(service="chunking-service")
    )

    def execute(self, context: PipelineContext) -> PipelineContext:
        document_payload = context.data.get("document")
        if isinstance(document_payload, Document):
            document = document_payload
        elif isinstance(document_payload, dict):
            document = _coerce_document(document_payload)
        else:
            raise StageFailure(
                "Missing document for chunking",
                status=400,
                stage=self.name,
                error_type="validation",
            )
        tenant_id = context.data.get("tenant_id") or context.tenant_id
        chunk_config = _stage_config(context.data, "chunking")
        profile_hint = (
            chunk_config.get("profile")
            or context.data.get("profile")
            or context.data.get("source")
        )
        options = _coerce_chunking_options(chunk_config.get("options"))
        try:
            with self.circuit_breaker.guard(self.name):
                run = self.ingestion.chunk_document(
                    document,
                    tenant_id=tenant_id,
                    source_hint=profile_hint,
                    options=options,
                )
        except StageFailure:
            record_ingestion_error(self.name, "circuit")
            raise
        except Exception as exc:  # pragma: no cover - service level safety
            record_ingestion_error(self.name, "failure")
            raise StageFailure(
                "Chunking stage failed",
                detail=str(exc),
                stage=self.name,
                retriable=True,
            ) from exc
        context.data["document"] = document.model_dump()
        context.data["chunks"] = [_serialise_chunk(chunk) for chunk in run.chunks]
        context.data["chunk_profile"] = run.profile
        context.data["chunk_count"] = len(run.chunks)
        context.data["chunk_duration"] = run.duration_seconds
        context.data.setdefault("metrics", {})["chunking"] = {
            "duration_ms": round(run.duration_seconds * 1000, 3),
            "chunks": len(run.chunks),
        }
        record_ingestion_document(context.pipeline_version or "default")
        observe_ingestion_stage_latency(self.name, run.duration_seconds)
        record_business_event("ingestion.chunked")
        return context


def _serialise_chunk(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "body": chunk.body,
        "granularity": chunk.granularity,
        "metadata": dict(chunk.metadata or {}),
    }


@dataclass(slots=True)
class EmbeddingStage(PipelineStage):
    worker: EmbeddingWorker
    namespaces: Sequence[str] | None = None
    models: Sequence[str] | None = None
    timeout_ms: int | None = 1000
    name: str = "embedding"
    circuit_breaker: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker(service="embedding-service")
    )

    def execute(self, context: PipelineContext) -> PipelineContext:
        chunks: Sequence[dict[str, Any]] = context.data.get("chunks") or []
        if not chunks:
            raise StageFailure(
                "No chunks available for embedding",
                status=400,
                stage=self.name,
                error_type="validation",
            )
        embed_config = _stage_config(context.data, "embedding")
        namespaces = embed_config.get("namespaces") or self.namespaces
        models = embed_config.get("models") or self.models
        texts = [chunk.get("body", "") for chunk in chunks]
        chunk_ids = [chunk.get("chunk_id", "") for chunk in chunks]
        try:
            request = EmbeddingRequest(
                tenant_id=context.tenant_id,
                chunk_ids=chunk_ids,
                texts=texts,
                namespaces=namespaces,
                models=models,
                correlation_id=context.correlation_id,
            )
            started = perf_counter()
            with self.circuit_breaker.guard(self.name):
                response = self.worker.run(request)
            duration = perf_counter() - started
        except StageFailure:
            record_ingestion_error(self.name, "circuit")
            raise
        except Exception as exc:  # pragma: no cover - service level guard
            record_ingestion_error(self.name, "failure")
            raise StageFailure(
                "Embedding stage failed",
                detail=str(exc),
                stage=self.name,
                retriable=True,
            ) from exc
        vectors = [
            {
                "id": vector.id,
                "model": vector.model,
                "namespace": vector.namespace,
                "kind": vector.kind,
                "vectors": vector.vectors,
                "terms": vector.terms,
                "dimension": vector.dimension,
                "metadata": dict(vector.metadata),
            }
            for vector in response.vectors
        ]
        context.data["embeddings"] = vectors
        context.data.setdefault("metrics", {})["embedding"] = {
            "vectors": len(vectors),
            "namespaces": sorted({vector["namespace"] for vector in vectors}),
            "duration_ms": round(duration * 1000, 3),
        }
        observe_ingestion_stage_latency(self.name, duration)
        record_business_event("ingestion.embedded")
        return context


@dataclass(slots=True)
class IndexingStage(PipelineStage):
    vector_service: VectorStoreService
    batch_size: int = 50
    timeout_ms: int | None = 2000
    name: str = "indexing"
    circuit_breaker: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker(service="vector-store")
    )

    def execute(self, context: PipelineContext) -> PipelineContext:
        embeddings: Sequence[dict[str, Any]] = context.data.get("embeddings") or []
        if not embeddings:
            raise StageFailure(
                "No embeddings provided for indexing",
                status=400,
                stage=self.name,
                error_type="validation",
            )
        index_config = _stage_config(context.data, "indexing")
        batch_size = int(index_config.get("batch_size", self.batch_size))
        grouped = defaultdict(list)
        for embedding in embeddings:
            namespace = embedding.get("namespace") or "default"
            record = _to_vector_record(embedding)
            if record is None:
                continue
            grouped[namespace].append(record)
        upserts: dict[str, int] = {}
        for namespace, records in grouped.items():
            batches = _batched(records, batch_size)
            total = 0
            for batch in batches:
                start = perf_counter()
                try:
                    with self.circuit_breaker.guard(self.name):
                        result = self.vector_service.upsert(
                            context=SecurityContext(
                                subject="ingestion-indexer",
                                tenant_id=context.tenant_id,
                                scopes={"index:write"},
                            ),
                            namespace=namespace,
                            records=batch,
                        )
                except StageFailure:
                    record_ingestion_error(self.name, "circuit")
                    raise
                except Exception as exc:  # pragma: no cover - persistence guard
                    record_ingestion_error(self.name, "failure")
                    raise StageFailure(
                        "Indexing stage failed",
                        detail=str(exc),
                        stage=self.name,
                        retriable=True,
                    ) from exc
                duration = perf_counter() - start
                observe_orchestration_stage("ingest", f"index:{namespace}", duration)
                observe_ingestion_stage_latency(self.name, duration)
                total += result.upserted
            upserts[namespace] = total
        context.data.setdefault("metrics", {})["indexing"] = {
            "namespaces": {name: count for name, count in upserts.items()},
        }
        context.data["index_result"] = upserts
        record_business_event("ingestion.indexed")
        return context


def _to_vector_record(embedding: dict[str, Any]) -> VectorRecord | None:
    vectors = embedding.get("vectors") or []
    if not vectors:
        return None
    values = list(vectors[0]) if vectors else []
    named_vectors = None
    if len(vectors) > 1:
        named_vectors = {
            f"segment_{idx}": list(vector)
            for idx, vector in enumerate(vectors[1:], start=1)
        }
    metadata = dict(embedding.get("metadata") or {})
    metadata.setdefault("model", embedding.get("model"))
    metadata.setdefault("kind", embedding.get("kind"))
    return VectorRecord(
        vector_id=str(embedding.get("id")),
        values=values,
        metadata=metadata,
        named_vectors=named_vectors,
    )


def _batched(records: Sequence[VectorRecord], size: int) -> Iterable[list[VectorRecord]]:
    for index in range(0, len(records), size):
        yield list(records[index : index + size])


def _stage_config(data: Mapping[str, Any], stage: str) -> dict[str, Any]:
    config = data.get("config")
    if isinstance(config, Mapping):
        stage_config = config.get(stage)
        if isinstance(stage_config, Mapping):
            return dict(stage_config)
    return {}


def _coerce_chunking_options(options: Any) -> ChunkingOptions | None:
    if not isinstance(options, Mapping):
        return None
    return ChunkingOptions(**{key: value for key, value in options.items() if isinstance(key, str)})


__all__ = [
    "ChunkingStage",
    "EmbeddingStage",
    "IndexingStage",
    "INGEST_CHUNKING_TOPIC",
    "INGEST_CHUNKS_TOPIC",
    "INGEST_DLQ_TOPIC",
    "INGEST_EMBEDDINGS_TOPIC",
    "INGEST_INDEXED_TOPIC",
]
