from __future__ import annotations

import pytest

from Medical_KG_rev.services.reranking.features import FeaturePipeline
from Medical_KG_rev.services.reranking.ltr import (
    LambdaMARTModel,
    OpenSearchLTRReranker,
    VespaRankProfile,
    VespaRankProfileReranker,
)
from Medical_KG_rev.services.reranking.models import QueryDocumentPair


def _pair(doc_id: str, **metadata: float) -> QueryDocumentPair:
    return QueryDocumentPair(
        tenant_id="tenant",
        doc_id=doc_id,
        query="hypertension treatment",
        text="hypertension treatment reduces blood pressure",
        metadata=metadata,
    )


def test_feature_pipeline_extracts_expected_features() -> None:
    pipeline = FeaturePipeline.default()
    pair = _pair("doc-1", bm25_score=12.0, splade_score=0.3, dense_score=0.6, recency_days=7)
    features = pipeline.extract(pair)
    assert "lexical_semantic_interaction" in features
    assert features["recency"] == pytest.approx(1.0 / (1.0 + 7.0))


def test_opensearch_ltr_scores_and_builds_sltr_query() -> None:
    model = LambdaMARTModel(
        coefficients={"bm25_score": 0.05, "dense_score": 0.1},
        intercept=0.2,
        name="lambda-mart",
        version="v2",
    )
    reranker = OpenSearchLTRReranker(model=model)
    pairs = [
        _pair("doc-1", bm25_score=12.0, splade_score=0.4, dense_score=0.7, recency_days=3),
        _pair("doc-2", bm25_score=5.0, splade_score=0.1, dense_score=0.2, recency_days=40),
    ]
    response = reranker.score_pairs(pairs, explain=True)
    assert len(response.results) == 2
    top = response.results[0]
    assert top.metadata["model"] == "lambda-mart"
    query = reranker.build_sltr_query("hypertension", doc_ids=[pair.doc_id for pair in pairs])
    model_info = query["rescore"]["query"]["rescore_query"]["sltr"]["model"]["stored"]
    assert model_info["name"] == reranker.feature_set


def test_training_pipeline_builds_dataset() -> None:
    pipeline = OpenSearchLTRReranker.training_pipeline(
        label_getter=lambda pair: 1.0 if pair.doc_id == "doc-1" else 0.0
    )
    dataset = pipeline.build_dataset([
        _pair("doc-1", bm25_score=10.0, dense_score=0.5, splade_score=0.3),
        _pair("doc-2", bm25_score=6.0, dense_score=0.1, splade_score=0.2),
    ])
    assert dataset.feature_order
    assert len(dataset.features) == 2


def test_vespa_rank_profile_reranker_returns_profile_metadata() -> None:
    profile = VespaRankProfile(name="clinical_rank", first_phase="nativeRank")
    reranker = VespaRankProfileReranker(profile=profile)
    reranker.with_second_phase("rerank-phase")
    response = reranker.score_pairs(
        [
            _pair("doc-1", bm25_score=8.0, dense_score=0.4, splade_score=0.3),
            _pair("doc-2", bm25_score=2.0, dense_score=0.1, splade_score=0.05),
        ],
        explain=True,
    )
    assert response.results[0].metadata["profile"]["name"] == "clinical_rank"
    package = reranker.build_deployment_package()
    assert package["rank_profiles"][0]["name"] == "clinical_rank"
