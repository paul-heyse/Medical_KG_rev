from Medical_KG_rev.services.reranking.evaluation.harness import (
    EvaluationResult,
    RerankerEvaluator,
)


def test_tradeoff_and_leaderboard():
    evaluator = RerankerEvaluator(ground_truth={})
    results = [
        EvaluationResult("a", 0.8, 0.7, 0.6, 10, 20, 30),
        EvaluationResult("b", 0.85, 0.72, 0.61, 12, 25, 35),
    ]
    tradeoff = evaluator.build_tradeoff_curve(results)
    assert tradeoff[0][0] <= tradeoff[1][0]

    leaderboard = evaluator.leaderboard(results)
    assert leaderboard[0].reranker_id == "b"


def test_ab_testing_delta():
    evaluator = RerankerEvaluator(ground_truth={})
    baseline = EvaluationResult("baseline", 0.7, 0.6, 0.5, 10, 15, 20)
    challenger = EvaluationResult("challenger", 0.8, 0.65, 0.55, 11, 18, 25)
    delta = evaluator.ab_test(baseline, challenger)
    assert delta["ndcg_delta"] == 0.1
    assert delta["latency_delta"] == 3
