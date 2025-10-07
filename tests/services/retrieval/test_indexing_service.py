from __future__ import annotations

from Medical_KG_rev.services.retrieval.chunking import ChunkingService
from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex
from Medical_KG_rev.services.retrieval.indexing_service import IndexingService
from Medical_KG_rev.services.retrieval.opensearch_client import OpenSearchClient
from Medical_KG_rev.services.embedding.service import EmbeddingVector


class _StubEmbeddingWorker:
    def run(self, request):
        vectors = []
        for chunk_id in request.chunk_ids:
            vectors.append(
                EmbeddingVector(
                    id=chunk_id,
                    model="qwen-3",
                    namespace="default",
                    kind="single_vector",
                    vectors=[[1.0, 0.0, 0.0, 0.0]],
                    terms=None,
                    dimension=4,
                    metadata={"foo": "bar", "vector_id": chunk_id},
                )
            )
        return type("Response", (), {"vectors": vectors})()


def test_index_document_creates_chunks_and_vectors():
    chunking = ChunkingService()
    worker = _StubEmbeddingWorker()
    opensearch = OpenSearchClient()
    faiss = FAISSIndex(dimension=4)
    service = IndexingService(chunking, worker, opensearch, faiss)

    text = "Introduction\nThis is a test document.\n\nResults\nSome findings."
    result = service.index_document("tenant", "doc-1", text)

    assert result.document_id == "doc-1"
    assert len(result.chunk_ids) > 0
    assert len(faiss.ids) == len(result.chunk_ids)


def test_incremental_indexing_skips_existing_chunks():
    chunking = ChunkingService()
    worker = _StubEmbeddingWorker()
    opensearch = OpenSearchClient()
    faiss = FAISSIndex(dimension=4)
    service = IndexingService(chunking, worker, opensearch, faiss)

    text = "Section\nContent"
    first = service.index_document("tenant", "doc-1", text)
    second = service.index_document("tenant", "doc-1", text, incremental=True)

    assert second.chunk_ids == []


def test_incremental_indexing_without_faiss_falls_back_to_text_index():
    chunking = ChunkingService()
    worker = _StubEmbeddingWorker()
    opensearch = OpenSearchClient()
    service = IndexingService(chunking, worker, opensearch, faiss=None)

    text = "Section\nContent"
    service.index_document("tenant", "doc-1", text)
    second = service.index_document("tenant", "doc-1", text, incremental=True)

    assert second.chunk_ids == []


def test_metadata_merges_embedding_and_chunk_payloads():
    chunking = ChunkingService()
    worker = _StubEmbeddingWorker()
    opensearch = OpenSearchClient()
    faiss = FAISSIndex(dimension=4)
    service = IndexingService(chunking, worker, opensearch, faiss)

    text = "Intro\nBody"
    service.index_document("tenant", "doc-1", text, metadata={"source": "pdf"})

    assert faiss.metadata
    stored_metadata = faiss.metadata[0]
    assert stored_metadata["foo"] == "bar"
    assert stored_metadata["source"] == "pdf"
    assert stored_metadata["vector_kind"] == "single_vector"
