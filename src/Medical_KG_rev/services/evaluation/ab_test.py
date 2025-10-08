"""A/B testing utilities for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import betainc, sqrt
from statistics import mean, stdev
from typing import Sequence


@dataclass(slots=True)
class ABTestResult:
    variant_a: str
    variant_b: str
    mean_difference: float
    t_statistic: float
    p_value: float
    significant: bool


class ABTestRunner:
    """Runs paired experiments comparing nDCG@10 values across configurations."""

    def __init__(self, *, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def run(
        self,
        *,
        variant_a: str,
        variant_b: str,
        metrics_a: Sequence[float],
        metrics_b: Sequence[float],
    ) -> ABTestResult:
        if len(metrics_a) != len(metrics_b):
            raise ValueError("Metric sequences must be the same length for paired t-test")
        if not metrics_a:
            return ABTestResult(
                variant_a=variant_a,
                variant_b=variant_b,
                mean_difference=0.0,
                t_statistic=0.0,
                p_value=1.0,
                significant=False,
            )
        differences = [b - a for a, b in zip(metrics_a, metrics_b)]
        mean_diff = mean(differences)
        std_diff = stdev(differences) if len(differences) > 1 else 0.0
        if std_diff == 0.0:
            p_value = 1.0
            t_statistic = 0.0
        else:
            n = len(differences)
            standard_error = std_diff / sqrt(n)
            t_statistic = mean_diff / standard_error if standard_error else 0.0
            p_value = _two_tailed_p_value(t_statistic, n - 1)
        return ABTestResult(
            variant_a=variant_a,
            variant_b=variant_b,
            mean_difference=mean_diff,
            t_statistic=t_statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
        )


def _two_tailed_p_value(t_stat: float, degrees: int) -> float:
    if degrees <= 0:
        return 1.0
    x = degrees / (degrees + t_stat * t_stat)
    # betainc returns regularised incomplete beta function
    cdf = 0.5 * betainc(degrees / 2.0, 0.5, x)
    if t_stat > 0:
        cdf = 1.0 - cdf
    return min(1.0, max(0.0, 2.0 * min(cdf, 1.0 - cdf)))


__all__ = ["ABTestResult", "ABTestRunner"]
