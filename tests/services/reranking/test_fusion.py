from __future__ import annotations

from Medical_KG_rev.services.reranking import FusionService, FusionSettings, FusionStrategy, ScoredDocument


def _doc(doc_id: str, score: float, strategy: str) -> ScoredDocument:
    return ScoredDocument(
        doc_id=doc_id,
        content=f"content-{doc_id}",
        tenant_id="tenant",
        source=strategy,
        strategy_scores={strategy: score},
        metadata={"strategy": strategy},
        score=score,
    )


def test_rrf_fusion_merges_and_orders():
    service = FusionService()
    ranked_lists = {
        "bm25": [_doc("a", 1.0, "bm25"), _doc("b", 0.8, "bm25")],
        "dense": [_doc("b", 0.9, "dense"), _doc("c", 0.7, "dense")],
    }

    fused = service.fuse(ranked_lists)
    assert fused.documents[0].doc_id == "a"
    assert fused.metrics["strategy_count"] == 2


def test_weighted_fusion_respects_weights():
    settings = FusionSettings(strategy=FusionStrategy.WEIGHTED, weights={"bm25": 0.8, "dense": 0.2})
    service = FusionService(settings)
    ranked_lists = {
        "bm25": [_doc("x", 0.9, "bm25")],
        "dense": [_doc("x", 0.1, "dense"), _doc("y", 0.8, "dense")],
    }

    fused = service.fuse(ranked_lists)
    assert fused.documents[0].doc_id == "x"
    assert fused.metrics["weights"]["bm25"] > fused.metrics["weights"]["dense"]
