from __future__ import annotations

from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex
from Medical_KG_rev.services.retrieval.opensearch_client import OpenSearchClient
from Medical_KG_rev.services.retrieval.retrieval_service import RetrievalService


def _setup_clients():
    opensearch = OpenSearchClient()
    opensearch.index("chunks", "1", {"text": "headache nausea", "document_id": "doc-1"})
    opensearch.index("chunks", "2", {"text": "migraine treatment", "document_id": "doc-2"})
    faiss = FAISSIndex(dimension=4)
    faiss.add("1", [1.0, 0.0, 0.0, 0.0], {"text": "headache nausea", "document_id": "doc-1"})
    faiss.add("2", [0.0, 1.0, 0.0, 0.0], {"text": "migraine treatment", "document_id": "doc-2"})
    return opensearch, faiss


def test_rrf_fusion_combines_results():
    opensearch, faiss = _setup_clients()
    service = RetrievalService(opensearch, faiss)

    results = service.search("chunks", "headache treatment", k=2)

    assert len(results) == 2
    assert all(result.retrieval_score > 0 for result in results)


def test_rerank_adds_scores():
    opensearch, faiss = _setup_clients()
    service = RetrievalService(opensearch, faiss)

    results = service.search("chunks", "headache", rerank=True)

    assert any(result.rerank_score is not None for result in results)


def test_granularity_weights_and_metadata():
    opensearch = OpenSearchClient()
    opensearch.index(
        "chunks",
        "doc-1:semantic:paragraph:0",
        {"text": "paragraph result", "granularity": "paragraph"},
    )
    opensearch.index(
        "chunks",
        "doc-1:sliding:window:1",
        {"text": "window result", "granularity": "window"},
    )
    faiss = FAISSIndex(dimension=4)
    service = RetrievalService(opensearch, faiss)

    results = service.search("chunks", "result", k=2)

    assert all(result.granularity for result in results)
    assert results[0].metadata.get("granularity") in {"paragraph", "window"}


def test_granularity_filtering():
    opensearch = OpenSearchClient()
    opensearch.index(
        "chunks",
        "doc-1:semantic:paragraph:0",
        {"text": "paragraph result", "granularity": "paragraph"},
    )
    opensearch.index(
        "chunks",
        "doc-1:sliding:window:1",
        {"text": "window result", "granularity": "window"},
    )
    faiss = FAISSIndex(dimension=4)
    service = RetrievalService(opensearch, faiss)

    results = service.search("chunks", "result", k=2, filters={"granularity": "paragraph"})

    assert results
    assert all(result.granularity == "paragraph" for result in results)


def test_neighbor_merging_for_windows():
    opensearch = OpenSearchClient()
    opensearch.index(
        "chunks",
        "doc-1:sliding_window:window:1",
        {"text": "first window", "granularity": "window"},
    )
    opensearch.index(
        "chunks",
        "doc-1:sliding_window:window:2",
        {"text": "second window", "granularity": "window"},
    )
    faiss = FAISSIndex(dimension=4)
    service = RetrievalService(opensearch, faiss)

    results = service.search("chunks", "window", k=2)

    merged = [result for result in results if result.metadata.get("merged_ids")]
    assert merged
