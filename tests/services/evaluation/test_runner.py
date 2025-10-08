from collections import defaultdict

from Medical_KG_rev.services.evaluation import (
    EvaluationConfig,
    EvaluationRunner,
    build_test_set,
)


def _test_dataset():
    return build_test_set(
        "demo",
        [
            {
                "query_id": "Q1",
                "query_text": "alpha",
                "query_type": "exact_term",
                "relevant_docs": [{"doc_id": "D1", "grade": 3}],
            },
            {
                "query_id": "Q2",
                "query_text": "beta",
                "query_type": "paraphrase",
                "relevant_docs": [{"doc_id": "D2", "grade": 2}],
            },
        ],
        version="v1",
    )


def test_evaluation_runner_computes_metrics() -> None:
    runner = EvaluationRunner(bootstrap_samples=10, random_seed=1)
    dataset = _test_dataset()
    calls: defaultdict[str, int] = defaultdict(int)

    def retrieve(record):
        calls[record.query_id] += 1
        return [f"D{record.query_id[-1]}", "DX"]

    result = runner.evaluate(dataset, retrieve, config=EvaluationConfig(top_k=2))
    assert result.dataset == "demo"
    assert result.metrics["recall@5"].mean == 1.0
    assert "exact_term" in result.per_query_type
    assert result.latency.mean >= 0.0
    assert result.cache_hit is False

    cached = runner.evaluate(dataset, retrieve, config=EvaluationConfig(top_k=2))
    assert cached.cache_hit is True
    assert calls["Q1"] == 1  # second call uses cache

    uncached = runner.evaluate(dataset, retrieve, config=EvaluationConfig(top_k=3), use_cache=False)
    assert uncached.cache_hit is False
    assert calls["Q1"] == 2
