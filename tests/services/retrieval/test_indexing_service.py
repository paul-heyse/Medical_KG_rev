from __future__ import annotations

from dataclasses import dataclass

from Medical_KG_rev.services.retrieval.chunking import ChunkingService
from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex
from Medical_KG_rev.services.retrieval.indexing_service import IndexingService
from Medical_KG_rev.services.retrieval.opensearch_client import OpenSearchClient


@dataclass
class _Vector:
    id: str
    model: str = "qwen-3"
    kind: str = "dense"
    values: list[float] | None = None
    dimension: int = 4


class _StubEmbeddingWorker:
    def run(self, request):
        vectors = []
        for chunk_id in request.chunk_ids:
            vectors.append(_Vector(id=chunk_id, values=[1.0, 0.0, 0.0, 0.0]))
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
