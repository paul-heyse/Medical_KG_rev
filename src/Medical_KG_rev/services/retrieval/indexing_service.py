"""Indexing pipeline that orchestrates chunking, embedding, and indexing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from Medical_KG_rev.services.embedding.service import EmbeddingRequest, EmbeddingWorker
from Medical_KG_rev.services.reranking.pipeline.cache import RerankCacheManager

from .chunking import Chunk, ChunkingOptions, ChunkingService
from .faiss_index import FAISSIndex
from .opensearch_client import OpenSearchClient


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
        rerank_cache: RerankCacheManager | None = None,
    ) -> None:
        self.chunking = chunking
        self.embedding_worker = embedding_worker
        self.opensearch = opensearch
        self.faiss = faiss
        self.chunk_index = chunk_index
        self.rerank_cache = rerank_cache

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
        if incremental and self.faiss is not None:
            existing = set(self.faiss.ids)
            chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing]
        if not chunks:
            return IndexingResult(document_id=document_id, chunk_ids=[])
        self._index_chunks(chunks, metadata)
        self._embed_and_index(tenant_id, chunks)
        return IndexingResult(document_id=document_id, chunk_ids=[chunk.chunk_id for chunk in chunks])

    def _index_chunks(self, chunks: Sequence[Chunk], metadata: Mapping[str, object] | None) -> None:
        documents = []
        for chunk in chunks:
            chunk_meta = getattr(chunk, "meta", None) or getattr(chunk, "metadata", {})
            doc = {
                "id": chunk.chunk_id,
                "text": chunk.body,
                "doc_id": chunk.doc_id,
                "granularity": chunk.granularity,
                "chunker": chunk.chunker,
            }
            doc.update(chunk_meta)
            if metadata:
                doc.update(metadata)
            documents.append(doc)
        self.opensearch.bulk_index(self.chunk_index, documents, id_field="id")

    def _embed_and_index(self, tenant_id: str, chunks: Sequence[Chunk]) -> None:
        if self.faiss is None:
            return
        request = EmbeddingRequest(
            tenant_id=tenant_id,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            texts=[chunk.body for chunk in chunks],
            normalize=True,
            metadatas=chunk_metadata,
        )
        response = self.embedding_worker.run(request)
        dense_vectors = [
            vector
            for vector in response.vectors
            if vector.kind in {"single_vector", "multi_vector"}
        ]
        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        for vector in dense_vectors:
            chunk = chunk_lookup.get(vector.id)
            if chunk is None:
                continue
            payload = getattr(vector, "vectors", None)
            if payload:
                base_values = payload[0]
            else:
                base_values = getattr(vector, "values", None)
            if base_values is None:
                continue
            values = [float(value) for value in base_values]
            if len(values) < self.faiss.dimension:
                values = values + [0.0] * (self.faiss.dimension - len(values))
            elif len(values) > self.faiss.dimension:
                values = values[: self.faiss.dimension]
            metadata = dict(chunk.meta)
            metadata.setdefault("text", chunk.body)
            metadata.setdefault("granularity", chunk.granularity)
            metadata.setdefault("chunker", chunk.chunker)
            self.faiss.add(vector.id, values, metadata)

    def refresh(self) -> None:
        """Placeholder for compatibility with production implementation."""
        return None
