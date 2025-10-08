import pytest

from Medical_KG_rev.services.evaluation.metrics import (
    average_precision,
    evaluate_ranking,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_precision_and_mrr() -> None:
    relevances = [3, 0, 2, 1, 0]
    assert recall_at_k(relevances, total_relevant=3, k=1) == 1 / 3
    assert precision_at_k(relevances, k=3) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(relevances) == 1.0


def test_average_precision() -> None:
    relevances = [1, 0, 1, 0, 1]
    assert average_precision(relevances) == pytest.approx((1 + (2 / 3) + (3 / 5)) / 3)


def test_ndcg_matches_reference() -> None:
    relevances = [3, 2, 0, 1]
    score = ndcg_at_k(relevances, k=4)
    assert 0 <= score <= 1
    assert score == pytest.approx(1.0)


def test_evaluate_ranking_returns_expected_metrics() -> None:
    judgments = {"A": 3.0, "B": 2.0, "C": 1.0}
    retrieved = ["A", "C", "D", "B"]
    metrics = evaluate_ranking(retrieved, judgments).metrics
    assert metrics["recall@5"] == 1.0
    assert metrics["ndcg@5"] == pytest.approx(metrics["ndcg@5"])
    assert metrics["mrr"] == 1.0
    assert metrics["map"] == pytest.approx((1 + (2 / 3) + (3 / 4)) / 3)


def test_invalid_k_raises_value_error() -> None:
    with pytest.raises(ValueError):
        recall_at_k([1, 0, 0], total_relevant=1, k=0)
