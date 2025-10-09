"""CloudEvents helpers for orchestration stage lifecycle.

This module provides CloudEvents-based event publishing for orchestration
stage lifecycle management. It implements event factories and emitters for
tracking stage transitions, failures, and retries.

The module supports:
- CloudEvents specification compliance
- Stage lifecycle event publishing
- Kafka-based event distribution
- Optional dependency handling for cloudevents library

Thread Safety:
    Thread-safe for concurrent event publishing.

Performance:
    O(1) event creation and publishing operations.

Example:
    >>> factory = CloudEventFactory()
    >>> emitter = StageEventEmitter(kafka_client)
    >>> emitter.emit_started(ctx, "processing", attempt=1)

"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from Medical_KG_rev.orchestration.kafka import KafkaClient
from Medical_KG_rev.orchestration.stages.contracts import StageContext

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

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

        def get_data(self) -> dict[str, Any]:
            """Get event data payload."""
            return self.data

        def __getitem__(self, item: str) -> Any:  # type: ignore[override]
            return super().__getitem__(item)

        def to_dict(self) -> dict[str, Any]:
            """Convert event to dictionary representation."""
            return dict(self)


# ==============================================================================
# STAGE CONTEXT DATA MODELS
# ==============================================================================


# ==============================================================================
# STAGE IMPLEMENTATIONS
# ==============================================================================

def _to_message(event: CloudEvent) -> dict[str, Any]:
    """Convert CloudEvent to Kafka message format.

    Args:
        event: CloudEvent instance to convert.

    Returns:
        Kafka message with attributes and data separated.

    """
    payload = event.to_dict() if hasattr(event, "to_dict") else {**event}  # type: ignore[call-arg]
    attributes = {key: value for key, value in payload.items() if key != "data"}
    data = payload.get("data", {})
    return {"attributes": attributes, "data": data}


@dataclass(slots=True)
class CloudEventFactory:
    """Create CloudEvents for stage lifecycle transitions.

    Factory for generating CloudEvents compliant with the CloudEvents
    specification for orchestration stage lifecycle events.

    Attributes:
        source_prefix: Prefix for event source identifiers.

    """

    source_prefix: str = "medical-kg/orchestration"

    def _base_attributes(self, event_type: str, ctx: StageContext, stage: str) -> dict[str, Any]:
        """Generate base CloudEvent attributes.

        Args:
            event_type: Type of event being created.
            ctx: Stage context containing metadata.
            stage: Stage name where event occurred.

        Returns:
            Base attributes for CloudEvent.

        """
        now = datetime.now(timezone.UTC).isoformat()
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
        """Create CloudEvent for stage start.

        Args:
            ctx: Stage context with pipeline metadata.
            stage: Name of the stage being started.
            attempt: Attempt number for this stage.

        Returns:
            CloudEvent representing stage start.

        """
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
        state_snapshot: str | None = None,
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
        if state_snapshot is not None:
            data["state_snapshot"] = state_snapshot
        return CloudEvent(attributes, data)

    def stage_failed(
        self,
        ctx: StageContext,
        stage: str,
        *,
        error: str,
        attempt: int,
        state_snapshot: str | None = None,
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
        if state_snapshot is not None:
            data["state_snapshot"] = state_snapshot
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
    """Publish stage lifecycle CloudEvents to the orchestration topic.

    Event emitter that publishes CloudEvents for stage lifecycle transitions
    to a Kafka topic for downstream processing and observability.

    Attributes:
        _kafka: Kafka client for publishing events.
        _topic: Kafka topic name for events.
        _factory: CloudEvent factory for creating events.

    """

    def __init__(
        self,
        kafka: KafkaClient,
        *,
        factory: CloudEventFactory | None = None,
        topic: str = "orchestration.events.v1",
    ) -> None:
        """Initialize event emitter.

        Args:
            kafka: Kafka client for publishing events.
            factory: Optional CloudEvent factory.
            topic: Kafka topic name for events.

        """
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
        state_snapshot: str | None = None,
    ) -> None:
        event = self._factory.stage_completed(
            ctx,
            stage,
            duration_ms=duration_ms,
            output_count=output_count,
            attempt=attempt,
            state_snapshot=state_snapshot,
        )
        self._publish(event)

    def emit_failed(
        self,
        ctx: StageContext,
        stage: str,
        *,
        attempt: int,
        error: str,
        state_snapshot: str | None = None,
    ) -> None:
        event = self._factory.stage_failed(
            ctx,
            stage,
            error=error,
            attempt=attempt,
            state_snapshot=state_snapshot,
        )
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


# ==============================================================================
# PLUGIN REGISTRATION
# ==============================================================================


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = ["CloudEvent", "CloudEventFactory", "StageEventEmitter"]
