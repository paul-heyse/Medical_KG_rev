"""Evaluation runner orchestrating retrieval benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from random import Random
from statistics import mean, median, stdev
from time import perf_counter

from prometheus_client import Gauge  # type: ignore

from .metrics import evaluate_ranking
from .test_sets import QueryJudgment, QueryType, TestSet


EVALUATION_RECALL = Gauge(
    "medicalkg_retrieval_recall_at_k",
    "Recall@K observed during evaluation runs",
    labelnames=("dataset", "k", "config"),
)
EVALUATION_NDCG = Gauge(
    "medicalkg_retrieval_ndcg_at_k",
    "nDCG@K observed during evaluation runs",
    labelnames=("dataset", "k", "config"),
)
EVALUATION_MRR = Gauge(
    "medicalkg_retrieval_mrr",
    "MRR observed during evaluation runs",
    labelnames=("dataset", "config"),
)


@dataclass(slots=True, frozen=True)
class MetricSummary:
    mean: float
    median: float
    std: float
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass(slots=True, frozen=True)
class EvaluationConfig:
    """Serializable configuration describing an evaluation run."""

    top_k: int = 10
    components: Sequence[str] | None = None
    rerank: bool | None = None

    def to_json(self) -> str:
        payload = {
            "top_k": self.top_k,
            "components": list(self.components) if self.components else None,
            "rerank": self.rerank,
        }
        return json.dumps(payload, sort_keys=True)


@dataclass(slots=True)
class EvaluationResult:
    dataset: str
    test_set_version: str
    metrics: Mapping[str, MetricSummary]
    latency: MetricSummary
    per_query: Mapping[str, Mapping[str, float]]
    per_query_type: Mapping[str, Mapping[str, float]]
    cache_key: str
    cache_hit: bool
    config: EvaluationConfig


class EvaluationRunner:
    """Executes retrieval evaluation runs for a given test set."""

    def __init__(
        self,
        *,
        bootstrap_samples: int = 500,
        random_seed: int | None = None,
    ) -> None:
        self.bootstrap_samples = bootstrap_samples
        self._random = Random(random_seed)
        self._cache: dict[str, EvaluationResult] = {}

    def _serialise_config(self, config: EvaluationConfig, test_set: TestSet) -> str:
        payload = {
            "config": json.loads(config.to_json()),
            "test_set": {"name": test_set.name, "version": test_set.version},
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def evaluate(
        self,
        test_set: TestSet,
        retrieval_fn: Callable[[QueryJudgment], Sequence[str]],
        *,
        config: EvaluationConfig | None = None,
        use_cache: bool = True,
    ) -> EvaluationResult:
        if config is None:
            config = EvaluationConfig()
        cache_key = self._serialise_config(config, test_set)
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            return EvaluationResult(
                dataset=cached.dataset,
                test_set_version=cached.test_set_version,
                metrics=cached.metrics,
                latency=cached.latency,
                per_query=cached.per_query,
                per_query_type=cached.per_query_type,
                cache_key=cache_key,
                cache_hit=True,
                config=cached.config,
            )

        per_query: dict[str, Mapping[str, float]] = {}
        per_query_type_values: dict[QueryType, list[Mapping[str, float]]] = defaultdict(list)
        latencies: list[float] = []
        for record in test_set.queries:
            started = perf_counter()
            doc_ids = list(retrieval_fn(record))
            latencies.append((perf_counter() - started) * 1000.0)
            metrics = evaluate_ranking(doc_ids, record.as_relevance_mapping())
            per_query[record.query_id] = metrics.metrics
            per_query_type_values[record.query_type].append(metrics.metrics)

        metrics_summary = self._summarise_metrics(per_query.values())
        latency_summary = self._summarise_latencies(latencies)
        per_query_type_summary = {
            key.value: {metric: mean_metric(values, metric) for metric in metrics_summary}
            for key, values in per_query_type_values.items()
        }

        result = EvaluationResult(
            dataset=test_set.name,
            test_set_version=test_set.version,
            metrics=metrics_summary,
            latency=latency_summary,
            per_query=per_query,
            per_query_type=per_query_type_summary,
            cache_key=cache_key,
            cache_hit=False,
            config=config,
        )
        self._cache[cache_key] = result
        self._record_metrics(result)
        return result

    def _summarise_metrics(self, values: Sequence[Mapping[str, float]]) -> dict[str, MetricSummary]:
        aggregated: dict[str, list[float]] = defaultdict(list)
        for payload in values:
            for metric, value in payload.items():
                aggregated[metric].append(float(value))
        summary: dict[str, MetricSummary] = {}
        for metric, scores in aggregated.items():
            summary[metric] = MetricSummary(
                mean=_mean(scores),
                median=median(scores) if scores else 0.0,
                std=_std(scores),
                ci_low=None,
                ci_high=None,
            )
            if metric in {"recall@10", "ndcg@10"}:
                ci_low, ci_high = self._bootstrap(scores)
                summary[metric] = MetricSummary(
                    mean=summary[metric].mean,
                    median=summary[metric].median,
                    std=summary[metric].std,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
        return summary

    def _summarise_latencies(self, latencies: Sequence[float]) -> MetricSummary:
        return MetricSummary(
            mean=_mean(latencies),
            median=median(latencies) if latencies else 0.0,
            std=_std(latencies),
        )

    def _bootstrap(self, values: Sequence[float]) -> tuple[float | None, float | None]:
        if len(values) < 2 or self.bootstrap_samples <= 0:
            return (None, None)
        samples: list[float] = []
        for _ in range(self.bootstrap_samples):
            draw = [self._random.choice(values) for _ in values]
            samples.append(_mean(draw))
        samples.sort()
        lower_index = int(0.025 * len(samples))
        upper_index = int(0.975 * len(samples))
        return (samples[lower_index], samples[min(upper_index, len(samples) - 1)])

    def _record_metrics(self, result: EvaluationResult) -> None:
        config_hash = result.cache_key[:8]
        for metric, summary in result.metrics.items():
            if metric.startswith("recall@"):
                k = metric.split("@", 1)[1]
                EVALUATION_RECALL.labels(result.dataset, k, config_hash).set(summary.mean)
            elif metric.startswith("ndcg@"):
                k = metric.split("@", 1)[1]
                EVALUATION_NDCG.labels(result.dataset, k, config_hash).set(summary.mean)
        if "mrr" in result.metrics:
            EVALUATION_MRR.labels(result.dataset, config_hash).set(result.metrics["mrr"].mean)


def _mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0
+
+
+def mean_metric(values: Sequence[Mapping[str, float]], metric: str) -> float:
+    collected = [payload.get(metric, 0.0) for payload in values]
+    return mean(collected) if collected else 0.0
+
+
+__all__ = [
+    "EvaluationConfig",
+    "EvaluationResult",
+    "EvaluationRunner",
+    "MetricSummary",
+    "mean_metric",
+]
