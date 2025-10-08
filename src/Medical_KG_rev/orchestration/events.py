"""CloudEvents helpers for orchestration stage lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from Medical_KG_rev.orchestration.kafka import KafkaClient
from Medical_KG_rev.orchestration.stages.contracts import StageContext

try:  # pragma: no cover - optional dependency for structured CloudEvents
    from cloudevents.http import CloudEvent  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    class CloudEvent(dict):  # type: ignore[override]
        """Fallback CloudEvent implementation using a simple mapping."""

        def __init__(self, attributes: dict[str, Any], data: dict[str, Any]) -> None:
            super().__init__({**attributes, "data": data})

        @property
        def data(self) -> dict[str, Any]:
            return self["data"]

        def get_data(self) -> dict[str, Any]:  # noqa: D401 - mirror cloudevents API
            return self.data

        def __getitem__(self, item: str) -> Any:  # type: ignore[override]
            return super().__getitem__(item)

        def to_dict(self) -> dict[str, Any]:  # noqa: D401 - convenience helper
            return dict(self)


def _to_message(event: CloudEvent) -> dict[str, Any]:
    if hasattr(event, "to_dict"):
        payload = event.to_dict()  # type: ignore[call-arg]
    else:  # pragma: no cover - cloudevents fallback path
        payload = {**event}
    attributes = {key: value for key, value in payload.items() if key != "data"}
    data = payload.get("data", {})
    return {"attributes": attributes, "data": data}


@dataclass(slots=True)
class CloudEventFactory:
    """Create CloudEvents for stage lifecycle transitions."""

    source_prefix: str = "medical-kg/orchestration"

    def _base_attributes(self, event_type: str, ctx: StageContext, stage: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        subject = ctx.doc_id or ctx.correlation_id or "unknown"
        return {
            "id": uuid4().hex,
            "type": event_type,
            "source": f"{self.source_prefix}/{ctx.pipeline_name or 'unknown'}/{stage}",
            "subject": subject,
            "time": now,
            "datacontenttype": "application/json",
        }

    def stage_started(self, ctx: StageContext, stage: str, attempt: int) -> CloudEvent:
        attributes = self._base_attributes("stage.started", ctx, stage)
        data = {
            "pipeline": ctx.pipeline_name,
            "pipeline_version": ctx.pipeline_version,
            "stage": stage,
            "tenant_id": ctx.tenant_id,
            "attempt": attempt,
        }
        return CloudEvent(attributes, data)

    def stage_completed(
        self,
        ctx: StageContext,
        stage: str,
        *,
        duration_ms: int,
        output_count: int,
        attempt: int,
    ) -> CloudEvent:
        attributes = self._base_attributes("stage.completed", ctx, stage)
        data = {
            "pipeline": ctx.pipeline_name,
            "pipeline_version": ctx.pipeline_version,
            "stage": stage,
            "tenant_id": ctx.tenant_id,
            "attempt": attempt,
            "duration_ms": duration_ms,
            "output_count": output_count,
        }
        return CloudEvent(attributes, data)

    def stage_failed(
        self,
        ctx: StageContext,
        stage: str,
        *,
        error: str,
        attempt: int,
    ) -> CloudEvent:
        attributes = self._base_attributes("stage.failed", ctx, stage)
        data = {
            "pipeline": ctx.pipeline_name,
            "pipeline_version": ctx.pipeline_version,
            "stage": stage,
            "tenant_id": ctx.tenant_id,
            "attempt": attempt,
            "error": error,
        }
        return CloudEvent(attributes, data)

    def stage_retrying(
        self,
        ctx: StageContext,
        stage: str,
        *,
        attempt: int,
        backoff_ms: int,
        reason: str,
    ) -> CloudEvent:
        attributes = self._base_attributes("stage.retrying", ctx, stage)
        data = {
            "pipeline": ctx.pipeline_name,
            "pipeline_version": ctx.pipeline_version,
            "stage": stage,
            "tenant_id": ctx.tenant_id,
            "attempt": attempt,
            "backoff_ms": backoff_ms,
            "reason": reason,
        }
        return CloudEvent(attributes, data)


class StageEventEmitter:
    """Publish stage lifecycle CloudEvents to the orchestration topic."""

    def __init__(
        self,
        kafka: KafkaClient,
        *,
        factory: CloudEventFactory | None = None,
        topic: str = "orchestration.events.v1",
    ) -> None:
        self._kafka = kafka
        self._topic = topic
        self._factory = factory or CloudEventFactory()
        self._kafka.create_topics([topic])

    @property
    def topic(self) -> str:
        return self._topic

    def emit_started(self, ctx: StageContext, stage: str, *, attempt: int) -> None:
        event = self._factory.stage_started(ctx, stage, attempt)
        self._publish(event)

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
            duration_ms=duration_ms,
            output_count=output_count,
            attempt=attempt,
        )
        self._publish(event)

    def emit_failed(self, ctx: StageContext, stage: str, *, attempt: int, error: str) -> None:
        event = self._factory.stage_failed(ctx, stage, error=error, attempt=attempt)
        self._publish(event)

    def emit_retrying(
        self,
        ctx: StageContext,
        stage: str,
        *,
        attempt: int,
        backoff_ms: int,
        reason: str,
    ) -> None:
        event = self._factory.stage_retrying(
            ctx,
            stage,
            attempt=attempt,
            backoff_ms=backoff_ms,
            reason=reason,
        )
        self._publish(event)

    def _publish(self, event: CloudEvent) -> None:
        message = _to_message(event)
        self._kafka.publish(self._topic, message)


__all__ = ["CloudEventFactory", "StageEventEmitter", "CloudEvent"]
