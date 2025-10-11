"""Evaluation harness orchestrating metric computation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from datetime import time as time_of_day
from datetime import timedelta
from pathlib import Path
from threading import Event, Thread
from typing import Any
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
            lines.append(
                f"- **{query_id}**: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )
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

    def compare_configs(
        self,
        dataset_name: str,
        variants: Mapping[str, Callable[[GroundTruthRecord], Sequence[str]]],
    ) -> Mapping[str, EvaluationReport]:
        """Evaluate multiple retrieval strategies for side-by-side comparison."""
        results: dict[str, EvaluationReport] = {}
        for name, retrieve in variants.items():
            results[name] = self.run(dataset_name, retrieve)
        return results

    def track_history(
        self,
        report: EvaluationReport,
        history_path: str | Path,
    ) -> None:
        """Append evaluation metrics to a JSONL history for regression detection."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "name": report.name,
            "metrics": report.metrics,
        }
        path = Path(history_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def evaluate_stages(
        self,
        dataset_name: str,
        *,
        chunking: Callable[[GroundTruthRecord], float] | None = None,
        embedding: Callable[[GroundTruthRecord], float] | None = None,
        retrieval: Callable[[GroundTruthRecord], float] | None = None,
        fusion: Callable[[GroundTruthRecord], float] | None = None,
        rerank: Callable[[GroundTruthRecord], float] | None = None,
    ) -> StageEvaluationResult:
        """Compute per-stage metrics using provided evaluators."""
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for record in self.ground_truth.queries(dataset_name):
            for name, fn in (
                ("chunking_f1", chunking),
                ("embedding_correlation", embedding),
                ("retrieval_recall", retrieval),
                ("fusion_gain", fusion),
                ("rerank_gain", rerank),
            ):
                if fn is None:
                    continue
                value = float(fn(record))
                totals.setdefault(name, 0.0)
                totals[name] += value
                counts[name] = counts.get(name, 0) + 1
        averages = {key: totals[key] / max(counts.get(key, 1), 1) for key in totals}
        return StageEvaluationResult(dataset=dataset_name, **averages)

    def schedule_nightly(
        self,
        dataset_name: str,
        retrieve: Callable[[GroundTruthRecord], Sequence[str]],
        output_dir: str | Path,
        *,
        hour_utc: int = 2,
    ) -> NightlyEvaluationRunner:
        """Schedule nightly evaluation runs at the specified UTC hour."""
        runner = NightlyEvaluationRunner(
            harness=self,
            dataset=dataset_name,
            retrieve=retrieve,
            output_dir=output_dir,
            hour_utc=hour_utc,
        )
        runner.start()
        return runner


@dataclass
class StageEvaluationResult:
    dataset: str
    chunking_f1: float | None = None
    embedding_correlation: float | None = None
    retrieval_recall: float | None = None
    fusion_gain: float | None = None
    rerank_gain: float | None = None


class NightlyEvaluationRunner:
    """Background thread that executes nightly evaluation runs."""

    def __init__(
        self,
        *,
        harness: EvalHarness,
        dataset: str,
        retrieve: Callable[[GroundTruthRecord], Sequence[str]],
        output_dir: str | Path,
        hour_utc: int,
    ) -> None:
        self.harness = harness
        self.dataset = dataset
        self.retrieve = retrieve
        self.output_dir = Path(output_dir)
        self.hour_utc = hour_utc
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)

    def _seconds_until_run(self) -> float:
        now = datetime.utcnow()
        target_time = datetime.combine(now.date(), time_of_day(self.hour_utc, 0))
        if now >= target_time:
            target_time += timedelta(days=1)
        return max((target_time - now).total_seconds(), 0.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            wait_seconds = self._seconds_until_run()
            self._stop.wait(wait_seconds)
            if self._stop.is_set():
                break
            report = self.harness.run(self.dataset, self.retrieve)
            self.harness.write_report(report, self.output_dir)
            self.harness.track_history(
                report,
                self.output_dir / f"{self.dataset}-history.jsonl",
            )


__all__ = [
    "EvalHarness",
    "EvaluationReport",
    "NightlyEvaluationRunner",
    "StageEvaluationResult",
]
