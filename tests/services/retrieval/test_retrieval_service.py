from __future__ import annotations

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.config import RerankingSettings
from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex
from Medical_KG_rev.services.retrieval.opensearch_client import OpenSearchClient
from Medical_KG_rev.services.retrieval.rerank_policy import TenantRerankPolicy
from Medical_KG_rev.services.retrieval.retrieval_service import RetrievalService
from Medical_KG_rev.services.reranking.errors import RerankingError


def _setup_clients():
    opensearch = OpenSearchClient()
    opensearch.index(
        "chunks",
        "1",
        {
            "text": "headache nausea",
            "document_id": "doc-1",
            "metadata": {"chunking_profile": "pmc-imrad"},
        },
    )
    opensearch.index(
        "chunks",
        "2",
        {
            "text": "migraine treatment",
            "document_id": "doc-2",
            "metadata": {"chunking_profile": "ctgov-registry"},
        },
    )
    faiss = FAISSIndex(dimension=4)
    faiss.add(
        "1",
        [1.0, 0.0, 0.0, 0.0],
        {"text": "headache nausea", "document_id": "doc-1", "chunking_profile": "pmc-imrad"},
    )
    faiss.add(
        "2",
        [0.0, 1.0, 0.0, 0.0],
        {"text": "migraine treatment", "document_id": "doc-2", "chunking_profile": "ctgov-registry"},
    )
    return opensearch, faiss


def _policy(**overrides: bool) -> TenantRerankPolicy:
    return TenantRerankPolicy(default_enabled=False, tenant_defaults=overrides, experiment_ratio=0.0)


def _service(opensearch: OpenSearchClient, faiss: FAISSIndex, **policy_overrides: bool) -> RetrievalService:
    return RetrievalService(
        opensearch,
        faiss,
        reranking_settings=RerankingSettings(),
        rerank_policy=_policy(**policy_overrides),
    )


def test_rrf_fusion_combines_results():
    opensearch, faiss = _setup_clients()
    service = _service(opensearch, faiss)

    results = service.search("chunks", "headache treatment", k=2)

    assert len(results) == 2
    assert all(result.retrieval_score > 0 for result in results)
    assert all(result.metadata.get("reranking", {}).get("applied") is False for result in results)


def test_rerank_adds_scores():
    opensearch, faiss = _setup_clients()
    service = _service(opensearch, faiss)

    results = service.search("chunks", "headache", rerank=True)

    assert any(result.rerank_score is not None for result in results)
    assert all("model" in result.metadata.get("reranking", {}) for result in results)


def test_explain_mode_includes_stage_metrics():
    opensearch, faiss = _setup_clients()
    service = _service(opensearch, faiss)

    results = service.search("chunks", "headache", rerank=True, explain=True)

    assert results[0].metadata.get("pipeline_metrics")
    assert results[0].metadata.get("timing")


def test_tenant_default_enables_reranking():
    opensearch, faiss = _setup_clients()
    service = _service(opensearch, faiss, oncology=True)

    results = service.search(
        "chunks",
        "headache",
        context=SecurityContext(subject="user", tenant_id="oncology", scopes={"*"}),
    )

    assert any(result.rerank_score is not None for result in results)
    metadata = results[0].metadata.get("reranking")
    assert metadata and metadata["applied"] is True
    assert metadata["cohort"].startswith("tenant:oncology")


def test_explicit_override_disables_reranking():
    opensearch, faiss = _setup_clients()
    service = _service(opensearch, faiss, oncology=True)

    results = service.search(
        "chunks",
        "headache",
        rerank=False,
        context=SecurityContext(subject="user", tenant_id="oncology", scopes={"*"}),
    )

    assert all(result.rerank_score is None for result in results)
    metadata = results[0].metadata.get("reranking")
    assert metadata and metadata["applied"] is False
    assert metadata["reason"] == "request"


def test_rerank_model_override_sets_metadata():
    opensearch, faiss = _setup_clients()
    service = _service(opensearch, faiss)

    results = service.search(
        "chunks",
        "headache",
        rerank=True,
        rerank_model="ms-marco-minilm-l12-v2",
    )

    metadata = results[0].metadata.get("reranking")
    assert metadata is not None
    assert metadata["model"]["key"] == "ms-marco-minilm-l12-v2"
    assert metadata["model"]["model_id"].endswith("MiniLM-L-12-v2")


def test_unknown_model_falls_back_to_default():
    opensearch, faiss = _setup_clients()
    service = _service(opensearch, faiss)

    results = service.search(
        "chunks",
        "headache",
        rerank=True,
        rerank_model="does-not-exist",
    )

    metadata = results[0].metadata.get("reranking")
    assert metadata is not None
    assert metadata["model"]["key"] == "bge-reranker-base"
    assert "warnings" in metadata
    assert "model_fallback" in metadata["warnings"]


def test_chunking_profile_filter_limits_results():
    opensearch, _ = _setup_clients()
    service = _service(opensearch, None)

    results = service.search(
        "chunks",
        "migraine",
        filters={"chunking_profile": "ctgov-registry"},
        rerank=False,
    )

    assert results
    assert all(
        result.metadata.get("chunking_profile") == "ctgov-registry"
        for result in results
    )


def test_rerank_fallback_records_error():
    opensearch, faiss = _setup_clients()

    class FailingEngine:
        def rerank(self, *args, **kwargs):  # noqa: D401 - signature proxy
            raise RerankingError("boom", status=500)

    service = RetrievalService(
        opensearch,
        faiss,
        reranking_engine=FailingEngine(),
        reranking_settings=RerankingSettings(),
        rerank_policy=_policy(),
    )

    results = service.search("chunks", "headache", rerank=True)

    assert all(result.rerank_score is None for result in results)
    metadata = results[0].metadata.get("reranking")
    assert metadata["fallback"] == "fusion"
    assert metadata["error"] == "RerankingError"
