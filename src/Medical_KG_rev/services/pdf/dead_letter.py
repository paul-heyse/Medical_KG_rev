from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from Medical_KG_rev.observability.alerts import get_alert_manager
from Medical_KG_rev.observability.metrics import (
    record_dead_letter_event,
    set_dead_letter_queue_depth,
)
from Medical_KG_rev.orchestration.kafka import KafkaClient

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class PdfDeadLetterQueue:
    """Publish unrecoverable PDF processing events to a Kafka-backed DLQ."""

    kafka: KafkaClient
    topic: str = "pdf.deadletter.v1"

    def __post_init__(self) -> None:
        self.kafka.create_topics([self.topic])
        self._alerts = get_alert_manager()

    def publish(
        self,
        *,
        job_id: str | None,
        tenant_id: str,
        document_id: str,
        stage: str,
        reason: str,
        payload: dict[str, Any] | None = None,
        attempts: int | None = None,
    ) -> None:
        message = {
            "job_id": job_id,
            "tenant_id": tenant_id,
            "document_id": document_id,
            "stage": stage,
            "reason": reason,
            "payload": payload or {},
            "attempts": attempts or 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        key = job_id or f"{tenant_id}:{document_id}"
        self.kafka.publish(self.topic, value=message, key=key)
        record_dead_letter_event(self.topic, stage)
        depth = self.kafka.pending(self.topic)
        set_dead_letter_queue_depth(self.topic, depth)
        self._alerts.dlq_depth(depth)
        self._alerts.pipeline_backlog(self.topic, depth)
        logger.error(
            "pdf.dlq.published",
            topic=self.topic,
            job_id=job_id,
            tenant_id=tenant_id,
            document_id=document_id,
            stage=stage,
            reason=reason,
            depth=depth,
        )

    def depth(self) -> int:
        return self.kafka.pending(self.topic)


__all__ = ["PdfDeadLetterQueue"]
