"""Evaluation utilities including embedding and retrieval metrics."""

from .ab_testing import ABTestOutcome, ABTestRunner
from .embedding_eval import (
    EmbeddingEvaluator,
    EvaluationDataset,
    MetricResult,
    NamespaceLeaderboard,
)
from .ground_truth import GroundTruthManager, GroundTruthRecord
from .harness import EvalHarness, EvaluationReport
from .metrics import average_precision, mean_reciprocal_rank, ndcg_at_k, recall_at_k

__all__ = [
    "ABTestOutcome",
    "ABTestRunner",
    "EmbeddingEvaluator",
    "EvaluationDataset",
    "EvalHarness",
    "EvaluationReport",
    "GroundTruthManager",
    "GroundTruthRecord",
    "MetricResult",
    "NamespaceLeaderboard",
    "average_precision",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
]
