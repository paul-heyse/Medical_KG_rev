"""Dagster-aware indexing service delegating to Haystack components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import structlog

from Medical_KG_rev.adapters import get_plugin_manager
from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.runtime import StageFactory
from Medical_KG_rev.orchestration.dagster.stages import build_default_stage_factory
from Medical_KG_rev.orchestration.haystack.components import HaystackIndexWriter
from Medical_KG_rev.orchestration.stages.contracts import (
    EmbedStage,
    EmbeddingBatch,
    EmbeddingVector,
    IndexStage,
    StageContext,
)
from Medical_KG_rev.services.reranking.pipeline.cache import RerankCacheManager

from .chunking import ChunkingOptions, ChunkingService
from .faiss_index import FAISSIndex
from .opensearch_client import OpenSearchClient

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class IndexingResult:
    """Acknowledgement returned after a document has been indexed."""

    document_id: str
    chunk_ids: Sequence[str]


class IndexingService:
    """Create retrieval chunks, embed them, and persist to search indices."""

    _PIPELINE_NAME = "gateway-direct"
    _PIPELINE_VERSION = "v1"

    def __init__(
        self,
        chunking: ChunkingService,
        opensearch: OpenSearchClient | None,
        faiss: FAISSIndex | None,
        *,
        chunk_index: str = "chunks",
        rerank_cache: RerankCacheManager | None = None,
        stage_factory: StageFactory | None = None,
        embed_stage: EmbedStage | None = None,
        index_stage: IndexStage | None = None,
        adapter_manager: AdapterPluginManager | None = None,
    ) -> None:
        self.chunking = chunking
        self.opensearch = opensearch
        self.faiss = faiss
        self.chunk_index = chunk_index
        self.rerank_cache = rerank_cache

        if stage_factory is None:
            manager = adapter_manager or get_plugin_manager()
            registry = build_default_stage_factory(manager)
            stage_factory = StageFactory(registry)
        self._stage_factory = stage_factory

        self._embed_definition = StageDefinition(name="embed", type="embed")
        self._embed_stage = embed_stage or self._resolve_embed_stage()
        self._index_stage = index_stage or self._build_index_stage()

    def index_document(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
        metadata: Mapping[str, object] | None = None,
        chunk_options: ChunkingOptions | None = None,
        incremental: bool = False,
    ) -> IndexingResult:
        chunks = list(
            self.chunking.chunk(
                tenant_id,
                document_id,
                text,
                chunk_options if chunk_options is not None else ChunkingOptions(),
            )
        )
        if not chunks:
            logger.info(
                "retrieval.indexing.no_chunks",
                tenant_id=tenant_id,
                doc_id=document_id,
            )
            return IndexingResult(document_id=document_id, chunk_ids=())

        if incremental:
            existing = self._existing_chunk_ids([chunk.chunk_id for chunk in chunks])
            chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing]
            if not chunks:
                logger.info(
                    "retrieval.indexing.incremental_skipped",
                    tenant_id=tenant_id,
                    doc_id=document_id,
                )
                return IndexingResult(document_id=document_id, chunk_ids=())

        context = StageContext(
            tenant_id=tenant_id,
            doc_id=document_id,
            correlation_id=uuid4().hex,
            metadata={"request_metadata": dict(metadata or {})},
            pipeline_name=self._PIPELINE_NAME,
            pipeline_version=self._PIPELINE_VERSION,
        )

        enriched_chunks = self._annotate_chunks(chunks, metadata)
        embed_stage = self._embed_stage
        batch = embed_stage.execute(context, enriched_chunks)
        batch = self._merge_vector_metadata(batch, enriched_chunks, metadata)

        index_stage = self._index_stage
        receipt = index_stage.execute(context, batch)

        logger.info(
            "retrieval.indexing.completed",
            tenant_id=tenant_id,
            doc_id=document_id,
            chunks_indexed=receipt.chunks_indexed,
            opensearch=receipt.metadata.get("index") if hasattr(receipt, "metadata") else None,
            faiss=receipt.metadata.get("faiss_index") if hasattr(receipt, "metadata") else None,
        )

        if self.rerank_cache is not None:
            self.rerank_cache.invalidate(tenant_id, [document_id])

        return IndexingResult(
            document_id=document_id,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )

    def refresh(self) -> None:  # pragma: no cover - retained for compatibility
        return None

    def _resolve_embed_stage(self) -> EmbedStage:
        stage = self._stage_factory.resolve(self._PIPELINE_NAME, self._embed_definition)
        if not isinstance(stage, EmbedStage):  # pragma: no cover - defensive guard
            raise TypeError("Resolved embed stage does not implement EmbedStage")
        return stage

    def _build_index_stage(self) -> IndexStage:
        dense_writer = _FAISSDocumentWriter(self.faiss)
        sparse_writer = _OpenSearchDocumentWriter(self.opensearch, self.chunk_index)
        return HaystackIndexWriter(
            dense_writer=dense_writer if self.faiss is not None else None,
            sparse_writer=sparse_writer if self.opensearch is not None else None,
            opensearch_index=self.chunk_index,
            faiss_index=self.chunk_index,
        )

    def _annotate_chunks(
        self,
        chunks: Sequence[Chunk],
        request_metadata: Mapping[str, object] | None,
    ) -> list[Chunk]:
        metadata = dict(request_metadata or {})
        enriched: list[Chunk] = []
        for chunk in chunks:
            chunk_meta = {
                **chunk.meta,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "tenant_id": chunk.tenant_id,
                "text": chunk.body,
                "granularity": chunk.granularity,
            }
            chunk_meta.update(metadata)
            enriched.append(chunk.model_copy(update={"meta": chunk_meta}))
        return enriched

    def _merge_vector_metadata(
        self,
        batch: EmbeddingBatch,
        chunks: Sequence[Chunk],
        request_metadata: Mapping[str, object] | None,
    ) -> EmbeddingBatch:
        if not batch.vectors:
            return batch
        request_meta = dict(request_metadata or {})
        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        vectors: list[EmbeddingVector] = []
        for vector in batch.vectors:
            chunk = chunk_lookup.get(vector.id)
            chunk_meta = dict(chunk.meta) if chunk is not None else {}
            merged_meta = {
                "chunk_id": vector.id,
                **chunk_meta,
                **request_meta,
                **vector.metadata,
            }
            if chunk is not None:
                merged_meta.setdefault("doc_id", chunk.doc_id)
                merged_meta.setdefault("tenant_id", chunk.tenant_id)
                merged_meta.setdefault("text", chunk.body)
                merged_meta.setdefault("granularity", chunk.granularity)
                merged_meta.setdefault("chunker", chunk.chunker)
                merged_meta.setdefault("chunker_version", chunk.chunker_version)
            vectors.append(
                EmbeddingVector(
                    id=vector.id,
                    values=vector.values,
                    metadata=merged_meta,
                )
            )
        return EmbeddingBatch(vectors=tuple(vectors), model=batch.model, tenant_id=batch.tenant_id)

    def _existing_chunk_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        existing: set[str] = set()
        if self.faiss is not None:
            known = set(self.faiss.ids)
            existing.update(chunk_id for chunk_id in chunk_ids if chunk_id in known)
        if self.opensearch is not None:
            for chunk_id in chunk_ids:
                if chunk_id in existing:
                    continue
                if self.opensearch.has_document(self.chunk_index, chunk_id):
                    existing.add(chunk_id)
        if existing:
            logger.info(
                "retrieval.indexing.incremental_existing",
                chunk_ids=len(existing),
            )
        return existing


class _OpenSearchDocumentWriter:
    """Adapter that satisfies the Haystack writer contract using our client."""

    def __init__(self, client: OpenSearchClient | None, index_name: str) -> None:
        self._client = client
        self._index_name = index_name

    def run(self, *, documents: Sequence[Any]) -> dict[str, Sequence[Any]]:
        if self._client is None or not documents:
            return {"documents": list(documents)}
        payloads: list[dict[str, object]] = []
        for doc in documents:
            meta = dict(getattr(doc, "meta", {}) or {})
            payload = {
                "id": str(getattr(doc, "id", meta.get("chunk_id", uuid4().hex))),
                "chunk_id": meta.get("chunk_id"),
                "doc_id": meta.get("doc_id"),
                "granularity": meta.get("granularity"),
                "chunker": meta.get("chunker"),
                "chunker_version": meta.get("chunker_version"),
                "text": getattr(doc, "content", meta.get("text", "")),
            }
            payload.update(meta)
            payloads.append(payload)
        if payloads:
            self._client.bulk_index(self._index_name, payloads, id_field="id")
        return {"documents": list(documents)}


class _FAISSDocumentWriter:
    """Adapter that writes Haystack documents into the FAISS test index."""

    def __init__(self, index: FAISSIndex | None) -> None:
        self._index = index

    def run(self, *, documents: Sequence[Any]) -> dict[str, Sequence[Any]]:
        if self._index is None or not documents:
            return {"documents": list(documents)}
        for doc in documents:
            embedding = getattr(doc, "embedding", None)
            if embedding is None:
                continue
            vector = [float(value) for value in embedding]
            if len(vector) < self._index.dimension:
                vector.extend([0.0] * (self._index.dimension - len(vector)))
            elif len(vector) > self._index.dimension:
                vector = vector[: self._index.dimension]
            meta = dict(getattr(doc, "meta", {}) or {})
            vector_id = str(getattr(doc, "id", meta.get("chunk_id", uuid4().hex)))
            self._index.add(vector_id, vector, meta)
        return {"documents": list(documents)}


__all__ = ["IndexingResult", "IndexingService"]
