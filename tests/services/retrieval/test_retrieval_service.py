from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

if "Medical_KG_rev.config" not in sys.modules:
    sys.modules["Medical_KG_rev.config"] = SimpleNamespace()


class RerankingSettings(SimpleNamespace):
    def __init__(
        self,
        *,
        fusion: Any | None = None,
        model: Any | None = None,
        pipeline: Any | None = None,
        cache_ttl: int = 3600,
        circuit_breaker_failures: int = 5,
        circuit_breaker_reset: float = 30.0,
    ) -> None:
        model = model or SimpleNamespace(model=None, reranker_id=None, batch_size=32)
        pipeline = pipeline or SimpleNamespace(
            retrieve_candidates=100,
            rerank_candidates=50,
            return_top_k=10,
        )
        super().__init__(
            fusion=fusion,
            model=model,
            pipeline=pipeline,
            cache_ttl=cache_ttl,
            circuit_breaker_failures=circuit_breaker_failures,
            circuit_breaker_reset=circuit_breaker_reset,
        )


sys.modules["Medical_KG_rev.config"].RerankingSettings = RerankingSettings

import pytest

pytest.importorskip("yaml")

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.services.retrieval.opensearch_client import OpenSearchClient
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
            "metadata": {
                "section_label": "Introduction",
                "chunking_profile": "pmc-intro",
            },
        },
    )
    opensearch.index(
        "chunks",
        "2",
        {
            "text": "migraine treatment",
            "document_id": "doc-2",
            "metadata": {
                "section_label": "Results",
                "chunking_profile": "ctgov-registry",
            },
        },
    )
    faiss = None
    return opensearch, faiss


def _policy(**overrides: bool) -> TenantRerankPolicy:
    return _DummyRerankPolicy(default_enabled=False, tenant_defaults=overrides)


@dataclass(slots=True)
class _Decision:
    enabled: bool
    cohort: str
    reason: str

    def as_metadata(self) -> dict[str, object]:
        return {"enabled": self.enabled, "cohort": self.cohort, "reason": self.reason}


class _DummyRerankPolicy:
    def __init__(self, *, default_enabled: bool = False, tenant_defaults: Mapping[str, bool] | None = None) -> None:
        self.default_enabled = bool(default_enabled)
        self.tenant_defaults = dict(tenant_defaults or {})

    def decide(self, tenant_id: str, query: str, explicit: bool | None) -> _Decision:
        if explicit is not None:
            return _Decision(bool(explicit), "override", "request")
        if tenant_id in self.tenant_defaults:
            enabled = bool(self.tenant_defaults[tenant_id])
            cohort = f"tenant:{tenant_id}:{'on' if enabled else 'off'}"
            return _Decision(enabled, cohort, "tenant-config")
        if self.default_enabled:
            return _Decision(True, "default:on", "global-config")
        return _Decision(False, "default:off", "global-config")


def _service(opensearch: OpenSearchClient, faiss: Any | None, **policy_overrides: bool) -> RetrievalService:
    return RetrievalService(
        opensearch,
        faiss,
        reranking_settings=None,
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


def test_clinical_boosting_prioritises_matching_section():
    client = OpenSearchClient()
    client.index(
        "chunks",
        "eligibility-chunk",
        {
            "text": "Eligibility criteria include adults and exclusion criteria",
            "document_id": "doc-eligibility",
            "metadata": {
                "section_label": "Eligibility Criteria",
                "intent_hint": "eligibility",
            },
        },
    )
    client.index(
        "chunks",
        "results-chunk",
        {
            "text": "Results indicate improved survival and eligibility mentioned",
            "document_id": "doc-results",
            "metadata": {
                "section_label": "Results",
                "intent_hint": "results",
            },
        },
    )

    service = RetrievalService(
        client,
        None,
        reranking_settings=None,
        rerank_policy=_policy(),
    )

    results = service.search(
        "chunks",
        "eligibility criteria for pembrolizumab",
        rerank=False,
        k=2,
    )

    assert results
    assert results[0].id == "eligibility-chunk"
    boosting = results[0].metadata.get("boosting", {})
    summary = boosting.get("clinical_summary")
    assert summary and summary["applied"] is True
    assert summary["intents"]
    assert boosting.get("clinical_details")
