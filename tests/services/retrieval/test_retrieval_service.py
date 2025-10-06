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
