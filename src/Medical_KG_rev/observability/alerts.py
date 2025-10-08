"""Lightweight alerting helpers for orchestration events."""

from __future__ import annotations

from dataclasses import dataclass

import structlog


logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class AlertThresholds:
    latency_ms: float = 200.0
    error_rate_threshold: float = 0.05
    dlq_threshold: int = 50
    mineru_timeout_seconds: float = 300.0
    pdf_processing_sla_seconds: float = 240.0


class AlertManager:
    """Best-effort alert dispatcher writing to logs for now."""

    def __init__(self, thresholds: AlertThresholds | None = None) -> None:
        self.thresholds = thresholds or AlertThresholds()

    def latency_breach(self, stage: str, duration_ms: float) -> None:
        if duration_ms > self.thresholds.latency_ms:
            logger.warning(
                "alerts.latency_breach",
                stage=stage,
                duration_ms=round(duration_ms, 2),
                threshold_ms=self.thresholds.latency_ms,
            )

    def error_observed(self, stage: str, error_type: str) -> None:
        logger.error(
            "alerts.stage_error",
            stage=stage,
            error_type=error_type,
        )

    def download_failure_rate(self, tenant_id: str, failure_rate: float, window: float) -> None:
        if failure_rate >= self.thresholds.error_rate_threshold:
            logger.error(
                "alerts.pdf_download.failure_rate",
                tenant_id=tenant_id,
                failure_rate=round(failure_rate, 4),
                window_seconds=window,
            )

    def mineru_timeout(self, tenant_id: str, document_id: str, duration: float) -> None:
        if duration >= self.thresholds.mineru_timeout_seconds:
            logger.error(
                "alerts.mineru.timeout",
                tenant_id=tenant_id,
                document_id=document_id,
                duration_seconds=round(duration, 2),
                threshold_seconds=self.thresholds.mineru_timeout_seconds,
            )

    def processing_sla_breach(self, stage: str, duration: float) -> None:
        if duration >= self.thresholds.pdf_processing_sla_seconds:
            logger.error(
                "alerts.pdf_processing.sla_breach",
                stage=stage,
                duration_seconds=round(duration, 2),
                threshold_seconds=self.thresholds.pdf_processing_sla_seconds,
            )

    def circuit_state_changed(self, service: str, state: str) -> None:
        if state == "open":
            logger.error("alerts.circuit_open", service=service)
        else:
            logger.info("alerts.circuit_state", service=service, state=state)

    def dlq_depth(self, depth: int) -> None:
        if depth > self.thresholds.dlq_threshold:
            logger.error(
                "alerts.dlq_depth",
                depth=depth,
                threshold=self.thresholds.dlq_threshold,
            )

    def pipeline_backlog(self, queue: str, depth: int) -> None:
        if depth > self.thresholds.dlq_threshold:
            logger.error(
                "alerts.pdf_pipeline.backlog",
                queue=queue,
                depth=depth,
                threshold=self.thresholds.dlq_threshold,
            )


_ALERT_MANAGER = AlertManager()


def get_alert_manager() -> AlertManager:
    return _ALERT_MANAGER


__all__ = ["AlertManager", "AlertThresholds", "get_alert_manager"]
