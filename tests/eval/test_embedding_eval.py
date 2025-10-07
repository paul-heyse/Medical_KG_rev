from Medical_KG_rev.eval.embedding_eval import ABTestResult, EmbeddingEvaluator, EvaluationDataset


def test_embedding_evaluator_metrics() -> None:
    dataset = EvaluationDataset(
        name="toy",
        queries={"q1": ["heart rate"]},
        relevant={"q1": {"doc-1"}},
    )

    def _retrieve(namespace: str, query: str, k: int):  # noqa: D401 - simple stub
        return [{"_id": "doc-1"}, {"_id": "doc-2"}]

    evaluator = EmbeddingEvaluator(dataset, _retrieve)
    leaderboard = evaluator.evaluate("dense", k=2)
    assert leaderboard.metrics[0].metric == "recall@k"
    ab = evaluator.ab_test("dense", "dense", k=2)
    assert isinstance(ab, ABTestResult)
