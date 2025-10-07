"""High level ingestion orchestration tying chunking, embeddings, and storage."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import structlog

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.embeddings.storage import StorageRouter
from Medical_KG_rev.embeddings.utils.batching import BatchProgress, iter_with_progress
from Medical_KG_rev.services.embedding.service import (
    EmbeddingRequest,
    EmbeddingVector,
    EmbeddingWorker,
)
from Medical_KG_rev.services.retrieval.chunking import ChunkingOptions, ChunkingService
from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex
from Medical_KG_rev.services.retrieval.opensearch_client import OpenSearchClient
from Medical_KG_rev.services.vector_store.models import VectorRecord
from Medical_KG_rev.services.vector_store.service import VectorStoreService

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class IngestionOptions:
    namespaces: Sequence[str] | None = None
    batch_size: int = 32
    retries: int = 2
    chunking: ChunkingOptions | None = None
    metadata: Mapping[str, object] | None = None
    correlation_id: str | None = None


@dataclass(slots=True)
class EmbeddingBatchMetrics:
    batches: int = 0
    total: int = 0
    duration_ms: float = 0.0


@dataclass(slots=True)
class IngestionResult:
    chunk_ids: list[str] = field(default_factory=list)
    stored: dict[str, int] = field(default_factory=dict)
    retries: int = 0
    metrics: EmbeddingBatchMetrics = field(default_factory=EmbeddingBatchMetrics)


class IngestionService:
    """Coordinates document chunking, embedding, and persistence."""

    def __init__(
        self,
        *,
        chunking: ChunkingService,
        embedding_worker: EmbeddingWorker,
        vector_store: VectorStoreService,
        opensearch: OpenSearchClient,
        storage_router: StorageRouter | None = None,
        faiss: FAISSIndex | None = None,
    ) -> None:
        self.chunking = chunking
        self.embedding_worker = embedding_worker
        self.vector_store = vector_store
        self.opensearch = opensearch
        self.faiss = faiss
        self.storage_router = storage_router or embedding_worker.storage_router
        if storage_router is not None and embedding_worker.storage_router is not storage_router:
            embedding_worker.storage_router = storage_router

    def ingest(
        self,
        *,
        tenant_id: str,
        document_id: str,
        text: str,
        context: SecurityContext,
        options: IngestionOptions | None = None,
    ) -> IngestionResult:
        opts = options or IngestionOptions()
        chunks = self.chunking.chunk(tenant_id, document_id, text, opts.chunking)
        if not chunks:
            return IngestionResult()
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.body for chunk in chunks]
        chunk_metadata: list[dict[str, Any]] = []
        for chunk in chunks:
            base_metadata: dict[str, Any] = {
                "document_id": document_id,
                "chunk_id": getattr(chunk, "chunk_id", getattr(chunk, "id", "")),
                "chunker": getattr(chunk, "chunker", "unknown"),
                "granularity": getattr(chunk, "granularity", "paragraph"),
                "text": getattr(chunk, "body", ""),
            }
            raw_meta = getattr(chunk, "metadata", getattr(chunk, "meta", {})) or {}
            base_metadata.update(dict(raw_meta))
            if "token_count" not in base_metadata and hasattr(chunk, "token_count"):
                base_metadata["token_count"] = getattr(chunk, "token_count")
            chunk_metadata.append(base_metadata)
        global_metadata = dict(opts.metadata) if opts.metadata else {}
        batch_size = max(1, opts.batch_size)
        retries = max(0, opts.retries)
        request = EmbeddingRequest(
            tenant_id=tenant_id,
            chunk_ids=chunk_ids,
            texts=texts,
            batch_size=batch_size,
            namespaces=opts.namespaces,
            correlation_id=opts.correlation_id,
            metadatas=chunk_metadata,
            actor=context.subject,
        )
        attempt = 0
        response = None
        start = time.perf_counter()
        while attempt <= retries:
            try:
                response = self.embedding_worker.run(request)
                break
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.warning(
                    "ingestion.embedding.failed",
                    tenant_id=tenant_id,
                    document_id=document_id,
                    attempt=attempt,
                    error=str(exc),
                )
                attempt += 1
                if attempt > retries:
                    raise
        duration_ms = (time.perf_counter() - start) * 1000
        result = IngestionResult(chunk_ids=chunk_ids, retries=attempt)
        result.metrics = EmbeddingBatchMetrics(
            batches=(len(texts) + batch_size - 1) // batch_size,
            total=len(texts),
            duration_ms=duration_ms,
        )
        stored_counts: dict[str, int] = {}
        progress = BatchProgress(total=len(response.vectors), callback=self._log_progress)
        for batch in iter_with_progress(response.vectors, batch_size, progress=progress):
            for item in batch:
                stored_counts.setdefault(item.kind, 0)
                stored_counts[item.kind] += self._persist_embedding(
                    item, metadata=global_metadata, context=context
                )
        result.stored = stored_counts
        logger.info(
            "ingestion.pipeline.completed",
            tenant_id=tenant_id,
            document_id=document_id,
            chunks=len(chunk_ids),
            duration_ms=duration_ms,
            retries=attempt,
            actor=context.subject,
        )
        return result

    def _persist_embedding(
        self,
        vector: EmbeddingVector,
        *,
        metadata: Mapping[str, object],
        context: SecurityContext,
    ) -> int:
        target = self.storage_router.route(vector.kind)
        payload_metadata: dict[str, Any] = {**vector.metadata}
        payload_metadata.update(dict(metadata))
        payload_metadata.pop("storage_target", None)
        if target.name == "qdrant" and vector.vectors:
            version = vector.model_version or ""
            record = VectorRecord(
                vector_id=vector.id,
                values=vector.vectors[0],
                metadata=payload_metadata,
                vector_version=f"{vector.model}:{version}" if version else vector.model,
            )
            self.vector_store.upsert(
                context=context,
                namespace=vector.namespace,
                records=[record],
            )
            return 1
        if target.name == "faiss" and self.faiss and vector.vectors:
            first = vector.vectors[0]
            self.faiss.add(
                vector.id,
                first[: self.faiss.dimension]
                if len(first) >= self.faiss.dimension
                else list(first) + [0.0] * (self.faiss.dimension - len(first)),
                metadata=payload_metadata,
            )
            return 1
        if target.name == "opensearch" and vector.terms:
            index_name = vector.namespace.replace(".", "_")
            body = dict(payload_metadata)
            body_text = str(body.get("text", ""))
            body["text"] = body_text
            body["rank_features"] = vector.terms
            self.opensearch.index(index=index_name, doc_id=vector.id, body=body)
            return 1
        if target.name == "opensearch_neural" and vector.neural_fields:
            index_name = f"neural_{vector.namespace.replace('.', '_')}"
            body = dict(payload_metadata)
            body_text = str(body.get("text", ""))
            body["text"] = body_text
            body.update(vector.neural_fields)
            self.opensearch.index(index=index_name, doc_id=vector.id, body=body)
            return 1
        return 0

    def _log_progress(self, processed: int, total: int) -> None:
        logger.info("ingestion.embedding.persist.progress", processed=processed, total=total)
