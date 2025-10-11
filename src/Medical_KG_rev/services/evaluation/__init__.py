"""Evaluation services for retrieval quality measurement."""

from .ab_test import ABTestResult, ABTestRunner
from .ci import enforce_recall_threshold
from .metrics import HttpClient
from .runner import EvaluationConfig, EvaluationResult, EvaluationRunner, MetricSummary
from .test_sets import (
    QueryJudgment,
    QueryType,
    TestSet,
    TestSetManager,
    build_test_set,
    cohens_kappa,
)

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
    "build_test_set",
    "cohens_kappa",
    "enforce_recall_threshold",
    "evaluate_ranking",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
