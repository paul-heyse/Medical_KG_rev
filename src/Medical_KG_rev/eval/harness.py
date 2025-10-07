"""Evaluation harness orchestrating metric computation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import json

from .ground_truth import GroundTruthManager, GroundTruthRecord
from .metrics import evaluate_query


Evaluator = Callable[[str, Mapping[str, Any]], Sequence[str]]


@dataclass
class EvaluationReport:
    name: str
    metrics: Mapping[str, float]
    per_query: Mapping[str, Mapping[str, float]]

    def to_markdown(self) -> str:
        lines = [f"# Evaluation Report: {self.name}"]
        lines.append("## Aggregate Metrics")
        for metric, value in sorted(self.metrics.items()):
            lines.append(f"- **{metric}**: {value:.4f}")
        lines.append("\n## Per-Query Metrics")
        for query_id, metrics in self.per_query.items():
            lines.append(f"- **{query_id}**: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return "\n".join(lines)

    def to_json(self) -> str:
        payload = {
            "name": self.name,
            "metrics": dict(self.metrics),
            "per_query": {key: dict(value) for key, value in self.per_query.items()},
        }
        return json.dumps(payload, indent=2)


class EvalHarness:
    """Runs evaluations against retrieval pipelines."""

    def __init__(self, ground_truth: GroundTruthManager) -> None:
        self.ground_truth = ground_truth

    def run(
        self,
        dataset_name: str,
        retrieve: Callable[[GroundTruthRecord], Sequence[str]],
    ) -> EvaluationReport:
        per_query: dict[str, Mapping[str, float]] = {}
        aggregates: dict[str, float] = {}
        for record in self.ground_truth.queries(dataset_name):
            retrieved_ids = list(retrieve(record))
            metrics = evaluate_query(retrieved_ids, record.judgments)
            per_query[record.query_id] = metrics
            for key, value in metrics.items():
                aggregates.setdefault(key, 0.0)
                aggregates[key] += value
        total = max(len(per_query), 1)
        averaged = {key: value / total for key, value in aggregates.items()}
        return EvaluationReport(name=dataset_name, metrics=averaged, per_query=per_query)

    def write_report(self, report: EvaluationReport, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{report.name}.json").write_text(report.to_json(), encoding="utf-8")
        (output / f"{report.name}.md").write_text(report.to_markdown(), encoding="utf-8")


__all__ = ["EvalHarness", "EvaluationReport"]
