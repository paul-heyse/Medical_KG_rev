"""Indexing pipeline that orchestrates chunking, embedding, and indexing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import structlog

from Medical_KG_rev.services.embedding.service import (
    EmbeddingRequest,
    EmbeddingVector,
    EmbeddingWorker,
)

from .chunking import Chunk, ChunkingOptions, ChunkingService
from .faiss_index import FAISSIndex
from .opensearch_client import OpenSearchClient


logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class IndexingResult:
    document_id: str
    chunk_ids: Sequence[str]


class IndexingService:
    def __init__(
        self,
        chunking: ChunkingService,
        embedding_worker: EmbeddingWorker,
        opensearch: OpenSearchClient,
        faiss: FAISSIndex | None,
        chunk_index: str = "chunks",
    ) -> None:
        self.chunking = chunking
        self.embedding_worker = embedding_worker
        self.opensearch = opensearch
        self.faiss = faiss
        self.chunk_index = chunk_index

    def index_document(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
        metadata: Mapping[str, object] | None = None,
        chunk_options: ChunkingOptions | None = None,
        incremental: bool = False,
    ) -> IndexingResult:
        chunks = self.chunking.chunk(tenant_id, document_id, text, chunk_options)
        if incremental:
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            existing = self._existing_chunk_ids(chunk_ids)
            chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing]
        if not chunks:
            return IndexingResult(document_id=document_id, chunk_ids=[])
        self._index_chunks(chunks, metadata)
        self._embed_and_index(tenant_id, chunks, metadata)
        return IndexingResult(document_id=document_id, chunk_ids=[chunk.chunk_id for chunk in chunks])

    def _index_chunks(self, chunks: Sequence[Chunk], metadata: Mapping[str, object] | None) -> None:
        documents = []
        for chunk in chunks:
            doc = {
                "id": chunk.chunk_id,
                "text": chunk.body,
                "doc_id": chunk.doc_id,
                "granularity": chunk.granularity,
                "chunker": chunk.chunker,
                **chunk.meta,
            }
            if metadata:
                doc.update(metadata)
            documents.append(doc)
        self.opensearch.bulk_index(self.chunk_index, documents, id_field="id")

    def _embed_and_index(
        self,
        tenant_id: str,
        chunks: Sequence[Chunk],
        metadata: Mapping[str, object] | None,
    ) -> None:
        if self.faiss is None:
            return
        request = EmbeddingRequest(
            tenant_id=tenant_id,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            texts=[chunk.body for chunk in chunks],
            normalize=True,
        )
        response = self.embedding_worker.run(request)
        dense_vectors = [vector for vector in response.vectors if vector.values]
        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        for vector in dense_vectors:
            chunk = chunk_lookup.get(vector.id)
            if chunk is None:
                continue
            normalized = self._normalize_vector_payload(vector)
            if normalized is None:
                continue
            metadata = self._merge_metadata(chunk, vector, metadata)
            self.faiss.add(vector.id, normalized, metadata)

    def refresh(self) -> None:
        """Placeholder for compatibility with production implementation."""
        return None

    def _existing_chunk_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        if self.faiss is not None:
            known = set(self.faiss.ids)
            return {chunk_id for chunk_id in chunk_ids if chunk_id in known}
        logger.info(
            "retrieval.indexing.incremental_fallback",
            reason="faiss_unavailable",
            chunk_ids=len(chunk_ids),
        )
        existing: set[str] = set()
        for chunk_id in chunk_ids:
            if self.opensearch.has_document(self.chunk_index, chunk_id):
                existing.add(chunk_id)
        return existing

    def _normalize_vector_payload(self, vector: EmbeddingVector) -> list[float] | None:
        values = vector.values
        if values is None or self.faiss is None:
            return None
        floats = [float(value) for value in values]
        if len(floats) < self.faiss.dimension:
            floats.extend([0.0] * (self.faiss.dimension - len(floats)))
        elif len(floats) > self.faiss.dimension:
            floats = floats[: self.faiss.dimension]
        return floats

    def _merge_metadata(
        self,
        chunk: Chunk,
        vector: EmbeddingVector,
        request_metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        metadata = dict(chunk.meta)
        metadata.setdefault("text", chunk.body)
        metadata.setdefault("granularity", chunk.granularity)
        metadata.setdefault("chunker", chunk.chunker)
        if request_metadata:
            metadata.update(request_metadata)
        metadata.update(vector.metadata)
        metadata.setdefault("model", vector.model)
        metadata.setdefault("namespace", vector.namespace)
        metadata.setdefault("vector_kind", vector.kind)
        return metadata
