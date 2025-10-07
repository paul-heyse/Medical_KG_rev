"""Indexing pipeline that orchestrates chunking, embedding, and indexing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from Medical_KG_rev.services.embedding.service import EmbeddingRequest, EmbeddingWorker

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
        faiss: FAISSIndex,
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
            chunks = [chunk for chunk in chunks if chunk.chunk_id not in self.faiss.ids]
        if not chunks:
            return IndexingResult(document_id=document_id, chunk_ids=[])
        self._index_chunks(chunks, metadata)
        self._embed_and_index(tenant_id, chunks)
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

    def _embed_and_index(self, tenant_id: str, chunks: Sequence[Chunk]) -> None:
        request = EmbeddingRequest(
            tenant_id=tenant_id,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            texts=[chunk.body for chunk in chunks],
            normalize=True,
        )
        response = self.embedding_worker.run(request)
        dense_vectors = [vector for vector in response.vectors if vector.kind == "dense"]
        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        for vector in dense_vectors:
            chunk = chunk_lookup.get(vector.id)
            if chunk is None:
                continue
            metadata = dict(chunk.meta)
            metadata.setdefault("text", chunk.body)
            metadata.setdefault("granularity", chunk.granularity)
            metadata.setdefault("chunker", chunk.chunker)
            self.faiss.add(vector.id, vector.values[: self.faiss.dimension], metadata)

    def refresh(self) -> None:
        """Placeholder for compatibility with production implementation."""
        return None
