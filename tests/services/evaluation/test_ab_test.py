import pytest

from Medical_KG_rev.services.evaluation.ab_test import ABTestRunner


def test_ab_test_runner_detects_difference() -> None:
    runner = ABTestRunner(alpha=0.05)
    metrics_a = [0.45, 0.5, 0.48, 0.46]
    metrics_b = [0.72, 0.7, 0.71, 0.73]
    outcome = runner.run(
        variant_a="fusion", variant_b="fusion+rerank", metrics_a=metrics_a, metrics_b=metrics_b
    )
    assert outcome.mean_difference > 0
    assert 0 <= outcome.p_value <= 1
    assert outcome.t_statistic != 0


def test_ab_test_runner_validates_lengths() -> None:
    runner = ABTestRunner()
    with pytest.raises(ValueError):
        runner.run(variant_a="A", variant_b="B", metrics_a=[0.1], metrics_b=[0.2, 0.3])
