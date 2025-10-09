"""Prometheus metrics for pipeline state operations."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

_STAGE_DURATION = Histogram(
    "pipeline_stage_duration_seconds",
    "Duration of pipeline stages",
    labelnames=("pipeline", "stage", "stage_type"),
)
_STAGE_ATTEMPTS = Counter(
    "pipeline_stage_attempts_total",
    "Number of attempts taken by a stage",
    labelnames=("pipeline", "stage", "stage_type"),
)
_STAGE_FAILURES = Counter(
    "pipeline_stage_failures_total",
    "Number of failed stage executions",
    labelnames=("pipeline", "stage", "stage_type"),
)


def record_stage_metrics(
    *,
    pipeline: str | None,
    stage: str,
    stage_type: str,
    attempts: int | None,
    duration_ms: int | None,
    error: str | None,
) -> None:
    """Push orchestration metrics into Prometheus registries."""
    pipeline_label = pipeline or "unknown"
    labels = dict(pipeline=pipeline_label, stage=stage, stage_type=stage_type)
    if attempts:
        _STAGE_ATTEMPTS.labels(**labels).inc(attempts)
    if duration_ms is not None:
        _STAGE_DURATION.labels(**labels).observe(duration_ms / 1000)
    if error:
        _STAGE_FAILURES.labels(**labels).inc()
