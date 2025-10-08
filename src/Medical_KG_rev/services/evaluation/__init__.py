"""Evaluation services for retrieval quality measurement."""

from .metrics import (
    average_precision,
    evaluate_ranking,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .runner import EvaluationConfig, EvaluationResult, EvaluationRunner, MetricSummary
from .test_sets import (
    QueryJudgment,
    QueryType,
    TestSet,
    TestSetManager,
    build_test_set,
    cohens_kappa,
)
from .ab_test import ABTestResult, ABTestRunner
from .ci import enforce_recall_threshold

__all__ = [
    "ABTestResult",
    "ABTestRunner",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationRunner",
    "MetricSummary",
    "QueryJudgment",
    "QueryType",
    "TestSet",
    "TestSetManager",
    "average_precision",
    "cohens_kappa",
    "build_test_set",
    "enforce_recall_threshold",
    "evaluate_ranking",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
