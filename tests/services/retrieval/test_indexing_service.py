import pytest

from Medical_KG_rev.services.retrieval.chunking import ChunkingOptions, ChunkingService
from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex
from Medical_KG_rev.services.retrieval.indexing_service import IndexingService
from Medical_KG_rev.services.retrieval.opensearch_client import OpenSearchClient


def _build_service(faiss: FAISSIndex | None = None) -> tuple[IndexingService, OpenSearchClient, FAISSIndex | None]:
    chunking = ChunkingService()
    opensearch = OpenSearchClient()
    service = IndexingService(chunking, opensearch, faiss)
    return service, opensearch, faiss


def test_index_document_creates_chunks_and_vectors() -> None:
    faiss = FAISSIndex(dimension=4)
    service, opensearch, index = _build_service(faiss)

    text = "Introduction\nThis is a test document.\n\nResults\nSome findings."
    result = service.index_document("tenant", "doc-1", text)

    assert result.document_id == "doc-1"
    assert len(result.chunk_ids) > 0
    assert index is not None and len(index.ids) == len(result.chunk_ids)
    for chunk_id in result.chunk_ids:
        assert opensearch.has_document("chunks", chunk_id)


def test_incremental_indexing_skips_existing_chunks() -> None:
    faiss = FAISSIndex(dimension=4)
    service, _opensearch, _ = _build_service(faiss)

    text = "Section\nContent"
    first = service.index_document("tenant", "doc-1", text)
    assert first.chunk_ids

    second = service.index_document("tenant", "doc-1", text, incremental=True)
    assert list(second.chunk_ids) == []


def test_incremental_indexing_without_faiss_falls_back_to_text_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, opensearch, _ = _build_service(faiss=None)

    calls: list[str] = []
    original_has_document = opensearch.has_document

    def tracking_has_document(index: str, doc_id: str) -> bool:
        calls.append(doc_id)
        return original_has_document(index, doc_id)

    monkeypatch.setattr(opensearch, "has_document", tracking_has_document)

    text = "Section\nContent"
    result = service.index_document("tenant", "doc-1", text)
    assert result.chunk_ids

    second = service.index_document("tenant", "doc-1", text, incremental=True)
    assert list(second.chunk_ids) == []
    assert calls  # open search was consulted for incremental detection


def test_metadata_merges_embedding_and_chunk_payloads() -> None:
    faiss = FAISSIndex(dimension=4)
    service, opensearch, index = _build_service(faiss)

    text = "Intro\nBody"
    options = ChunkingOptions(metadata={"source": "pdf"})
    service.index_document("tenant", "doc-1", text, metadata={"pipeline": "gateway"}, chunk_options=options)

    assert index is not None
    assert index.metadata
    stored_metadata = index.metadata[0]
    assert stored_metadata["source"] == "pdf"
    assert stored_metadata["pipeline"] == "gateway"
    assert stored_metadata["chunker"].startswith("haystack")

    stored_doc = opensearch.get("chunks", index.ids[0])
    assert stored_doc["text"]
