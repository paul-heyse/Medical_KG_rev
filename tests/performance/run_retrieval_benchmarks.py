#!/usr/bin/env python3
"""Run hybrid retrieval benchmarks and emit summary statistics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import httpx

from Medical_KG_rev.services.evaluation.runner import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationRunner,
)
from Medical_KG_rev.services.evaluation.test_sets import TestSet, TestSetManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000", help="Gateway base URL")
    parser.add_argument("--tenant-id", default="eval", help="Tenant identifier used for requests")
    parser.add_argument("--token", default=None, help="Bearer token for authenticated deployments")
    parser.add_argument(
        "--test-set", default="test_set_v1", help="Name of the evaluation dataset to load"
    )
    parser.add_argument(
        "--dataset-root", default=None, help="Filesystem path containing test set YAML files"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of documents to request per query"
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Enable reranking during the benchmark run"
    )
    parser.add_argument("--rerank-model", default=None, help="Optional reranker model override")
    parser.add_argument(
        "--metrics-endpoint",
        default=None,
        help="Prometheus metrics endpoint (defaults to <base-url>/metrics)",
    )
    parser.add_argument(
        "--output", default=None, help="Optional JSON file to write benchmark results to"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP timeout in seconds for retrieval requests"
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Bootstrap samples for EvaluationRunner (0 disables)",
    )
    parser.add_argument(
        "--markdown-output", default=None, help="Optional Markdown file for human-readable summary"
    )
    return parser.parse_args()


class ResponseCollector:
    """Captures latency and component timing observations from retrieval responses."""

    def __init__(self) -> None:
        self.component_timings: dict[str, list[float]] = defaultdict(list)
        self.stage_timings: dict[str, list[float]] = defaultdict(list)
        self.cache_hits: list[float] = []
        self.errors: list[Mapping[str, Any]] = []
        self.requests: int = 0

    def record(self, response: Mapping[str, Any]) -> None:
        meta = response.get("meta") or {}
        documents = (response.get("data") or {}).get("documents") or []
        rerank_meta = meta.get("rerank") or {}
        stage_timings_ms = rerank_meta.get("stage_timings_ms") or {}
        components = rerank_meta.get("components") or {}
        component_timings_ms = (
            components.get("timings_ms") if isinstance(components, Mapping) else {}
        )

        for name, value in stage_timings_ms.items():
            try:
                self.stage_timings[name].append(float(value))
            except (TypeError, ValueError):
                continue

        if isinstance(component_timings_ms, Mapping):
            for name, value in component_timings_ms.items():
                try:
                    self.component_timings[name].append(float(value))
                except (TypeError, ValueError):
                    continue

        # Fallback to document metadata if component timings are embedded there.
        if not component_timings_ms:
            for doc in documents:
                metadata = doc.get("metadata") or {}
                comp = metadata.get("components") or {}
                timings = comp.get("timings_ms") if isinstance(comp, Mapping) else {}
                if not timings:
                    continue
                for name, value in timings.items():
                    try:
                        self.component_timings[name].append(float(value))
                    except (TypeError, ValueError):
                        continue

        cache_meta = rerank_meta.get("cache") if isinstance(rerank_meta, Mapping) else None
        if isinstance(cache_meta, Mapping) and "hit" in cache_meta:
            self.cache_hits.append(1.0 if cache_meta.get("hit") else 0.0)

        if meta.get("errors"):
            for error in meta["errors"]:
                if isinstance(error, Mapping):
                    self.errors.append(error)

        self.requests += 1


def percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def summarise(values: Iterable[float]) -> Mapping[str, float]:
    ordered = [float(v) for v in values]
    if not ordered:
        return {}
    return {
        "count": float(len(ordered)),
        "mean": mean(ordered),
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "max": max(ordered),
    }


def scrape_prometheus(
    endpoint: str, headers: Mapping[str, str] | None, metric_names: Sequence[str]
) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {name: [] for name in metric_names}
    try:
        response = httpx.get(endpoint, headers=headers, timeout=10.0)
        response.raise_for_status()
    except Exception:
        return metrics
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        for metric in metric_names:
            if not line.startswith(metric):
                continue
            try:
                value = float(line.split()[-1])
            except ValueError:
                continue
            metrics[metric].append(value)
    return metrics


def build_retrieval_fn(
    client: httpx.Client,
    base_url: str,
    tenant_id: str,
    collector: ResponseCollector,
    top_k: int,
    rerank: bool,
    rerank_model: str | None,
) -> Callable[[Any], list[str]]:
    def _call(record: Any) -> list[str]:
        payload: dict[str, Any] = {
            "tenant_id": tenant_id,
            "query": getattr(record, "query_text", getattr(record, "text", "")),
            "top_k": top_k,
            "filters": {},
            "rerank": rerank,
            "rerank_model": rerank_model,
            "query_intent": getattr(record, "query_type", None),
            "table_only": getattr(record, "query_type", "") == "tabular",
            "explain": True,
        }
        started = perf_counter()
        response = client.post(f"{base_url}/v1/retrieve", json=payload)
        duration = perf_counter() - started
        response.raise_for_status()
        data = response.json()
        collector.record(data)
        documents = (data.get("data") or {}).get("documents") or []
        doc_ids = [str(doc.get("id")) for doc in documents]
        # Attach measured latency so callers can inspect via collector.stage_timings.
        collector.stage_timings["http_roundtrip"].append(duration * 1000.0)
        return doc_ids

    return _call


def load_test_set(name: str, root: str | None) -> TestSet:
    manager = TestSetManager(root=Path(root) if root else None)
    return manager.load(name)


def run_benchmark(
    args: argparse.Namespace,
) -> tuple[EvaluationResult, ResponseCollector, dict[str, list[float]]]:
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    collector = ResponseCollector()
    test_set = load_test_set(args.test_set, args.dataset_root)
    client = httpx.Client(timeout=args.timeout, headers=headers)
    retrieval_fn = build_retrieval_fn(
        client,
        args.base_url,
        args.tenant_id,
        collector,
        top_k=args.top_k,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
    )
    runner = EvaluationRunner(bootstrap_samples=args.bootstrap_samples or 0)
    config = EvaluationConfig(top_k=args.top_k, rerank=args.rerank)
    result = runner.evaluate(test_set, retrieval_fn, config=config, use_cache=False)
    client.close()
    metrics_endpoint = args.metrics_endpoint or f"{args.base_url.rstrip('/')}/metrics"
    prom_metrics = scrape_prometheus(
        metrics_endpoint,
        headers if args.token else None,
        [
            "gpu_utilization_percent",
            "reranking_gpu_utilization_percent",
            "reranking_cache_hit_rate",
        ],
    )
    return result, collector, prom_metrics


def build_summary(
    evaluation: EvaluationResult,
    collector: ResponseCollector,
    prom_metrics: Mapping[str, list[float]],
) -> dict[str, Any]:
    summary = {
        "dataset": evaluation.dataset,
        "test_set_version": evaluation.test_set_version,
        "metrics": {
            metric: {
                "mean": data.mean,
                "median": data.median,
                "std": data.std,
                "ci_low": data.ci_low,
                "ci_high": data.ci_high,
            }
            for metric, data in evaluation.metrics.items()
        },
        "latency_ms": {
            "mean": evaluation.latency.mean,
            "median": evaluation.latency.median,
            "std": evaluation.latency.std,
            "p95": percentile(collector.stage_timings.get("http_roundtrip", []), 95),
        },
        "per_component_latency_ms": {
            component: summarise(values)
            for component, values in collector.component_timings.items()
        },
        "pipeline_stage_latency_ms": {
            stage: summarise(values)
            for stage, values in collector.stage_timings.items()
            if stage != "http_roundtrip"
        },
        "cache_hit_rate": sum(collector.cache_hits) / len(collector.cache_hits)
        if collector.cache_hits
        else None,
        "errors": collector.errors,
        "requests": collector.requests,
        "prometheus_samples": {
            metric: summarise(values) for metric, values in prom_metrics.items() if values
        },
    }
    return summary


def format_markdown(summary: Mapping[str, Any]) -> str:
    def fmt(value: Any, precision: int = 3) -> str:
        if value is None:
            return "—"
        if isinstance(value, (int, float)):
            return f"{float(value):.{precision}f}"
        return str(value)

    lines: list[str] = []
    lines.append("# Retrieval Benchmark Summary")
    lines.append("")
    lines.append(f"- **Dataset**: {summary.get('dataset', 'unknown')}")
    lines.append(f"- **Test Set Version**: {summary.get('test_set_version', 'unknown')}")
    lines.append(f"- **Requests**: {summary.get('requests', 0)}")
    cache_hit_rate = summary.get("cache_hit_rate")
    if cache_hit_rate is not None:
        lines.append(f"- **Cache Hit Rate**: {fmt(cache_hit_rate, precision=2)}")
    lines.append("")

    metrics = summary.get("metrics", {})
    if metrics:
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Mean | Median | Std | CI Low | CI High |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for name, stats in metrics.items():
            lines.append(
                f"| {name} | {fmt(stats.get('mean'))} | {fmt(stats.get('median'))} | "
                f"{fmt(stats.get('std'))} | {fmt(stats.get('ci_low'))} | {fmt(stats.get('ci_high'))} |"
            )
        lines.append("")

    latency = summary.get("latency_ms", {})
    if latency:
        lines.append("## Latency (ms)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for name, value in latency.items():
            lines.append(f"| {name} | {fmt(value)} |")
        lines.append("")

    def render_section(title: str, payload: Mapping[str, Mapping[str, float]] | None) -> None:
        if not payload:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Name | Count | Mean | P50 | P95 | Max |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for name, stats in sorted(payload.items()):
            lines.append(
                f"| {name} | {fmt(stats.get('count'), 0)} | {fmt(stats.get('mean'))} | "
                f"{fmt(stats.get('p50'))} | {fmt(stats.get('p95'))} | {fmt(stats.get('max'))} |"
            )
        lines.append("")

    render_section("Per-Component Latency (ms)", summary.get("per_component_latency_ms"))
    render_section("Pipeline Stage Latency (ms)", summary.get("pipeline_stage_latency_ms"))

    prometheus = summary.get("prometheus_samples", {})
    if prometheus:
        lines.append("## Prometheus Samples")
        lines.append("")
        lines.append("| Metric | Count | Mean | P50 | P95 | Max |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for name, stats in sorted(prometheus.items()):
            lines.append(
                f"| {name} | {fmt(stats.get('count'), 0)} | {fmt(stats.get('mean'))} | "
                f"{fmt(stats.get('p50'))} | {fmt(stats.get('p95'))} | {fmt(stats.get('max'))} |"
            )
        lines.append("")

    errors = summary.get("errors") or []
    if errors:
        lines.append("## Errors")
        lines.append("")
        for error in errors:
            lines.append(f"- {error}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    evaluation, collector, prom_metrics = run_benchmark(args)
    summary = build_summary(evaluation, collector, prom_metrics)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(format_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
