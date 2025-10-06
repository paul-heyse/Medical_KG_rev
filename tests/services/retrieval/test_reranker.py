from Medical_KG_rev.services.retrieval.reranker import CrossEncoderReranker


def test_reranker_returns_scores_and_metrics():
    reranker = CrossEncoderReranker()
    candidates = [
        {"id": "1", "text": "hypertension treatment"},
        {"id": "2", "text": "diabetes management"},
    ]
    ranked, metrics = reranker.rerank("hypertension", candidates, top_k=2)
    assert len(ranked) == 2
    assert metrics["evaluated"] == 2
    assert ranked[0]["id"] in {"1", "2"}
    assert "model" in metrics
