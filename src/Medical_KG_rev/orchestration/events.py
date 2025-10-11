"""Placeholder event emitting utilities for orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

try:  # pragma: no cover - optional dependency
    from cloudevents.http import CloudEvent  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - fallback implementation
    raise ImportError("cloudevents is required for orchestration event emission") from exc

from Medical_KG_rev.orchestration.kafka import KafkaClient
from Medical_KG_rev.orchestration.stages.contracts import StageContext


def _current_time_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class CloudEventFactory:
    """Create minimal CloudEvents for orchestration stages."""

    source_prefix: str = "medical-kg/orchestration"

    def stage_event(self, ctx: StageContext, stage: str, event_type: str, **data: Any) -> CloudEvent:
        attributes = {
            "id": uuid4().hex,
            "type": event_type,
            "source": f"{self.source_prefix}/{stage}",
            "subject": ctx.doc_id or ctx.correlation_id or "unknown",
            "time": _current_time_iso(),
            "datacontenttype": "application/json",
        }
        payload = {
            "tenant_id": ctx.tenant_id,
            "correlation_id": ctx.correlation_id,
            "pipeline": ctx.pipeline_name,
            **data,
        }
        return CloudEvent(attributes, payload)

    def stage_started(self, ctx: StageContext, stage: str, *, attempt: int) -> CloudEvent:
        return self.stage_event(ctx, stage, "stage.started", attempt=attempt)

    def stage_completed(
        self,
        ctx: StageContext,
        stage: str,
        *,
        attempt: int,
        duration_ms: int,
        output_count: int,
    ) -> CloudEvent:
        return self.stage_event(
            ctx,
            stage,
            "stage.completed",
            attempt=attempt,
            duration_ms=duration_ms,
            output_count=output_count,
        )

    def stage_failed(
        self,
        ctx: StageContext,
        stage: str,
        *,
        attempt: int,
        error: str,
    ) -> CloudEvent:
        return self.stage_event(ctx, stage, "stage.failed", attempt=attempt, error=error)


class StageEventEmitter:
    """Publish stage lifecycle events."""

    def __init__(
        self,
        kafka: KafkaClient,
        *,
        topic: str = "orchestration.events.v1",
        factory: CloudEventFactory | None = None,
    ) -> None:
        self._kafka = kafka
        self._topic = topic
        self._factory = factory or CloudEventFactory()
        self._kafka.create_topics([topic])

    def _publish(self, event: CloudEvent) -> None:
        self._kafka.publish(self._topic, event.to_dict())

    def emit_started(self, ctx: StageContext, stage: str, *, attempt: int) -> None:
        self._publish(self._factory.stage_started(ctx, stage, attempt=attempt))

    def emit_completed(
        self,
        ctx: StageContext,
        stage: str,
        *,
        attempt: int,
        duration_ms: int,
        output_count: int,
    ) -> None:
        event = self._factory.stage_completed(
            ctx,
            stage,
            attempt=attempt,
            duration_ms=duration_ms,
            output_count=output_count,
        )
        self._publish(event)

    def emit_failed(self, ctx: StageContext, stage: str, *, attempt: int, error: str) -> None:
        self._publish(self._factory.stage_failed(ctx, stage, attempt=attempt, error=error))


__all__ = ["CloudEvent", "CloudEventFactory", "StageEventEmitter"]
