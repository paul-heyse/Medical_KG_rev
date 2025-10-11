"""CloudEvents emission utilities for chunking lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict
from uuid import uuid4

try:  # pragma: no cover - optional dependency
    from cloudevents.http import CloudEvent  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - fallback implementation
    raise ImportError("cloudevents is required for chunking event emission") from exc

from Medical_KG_rev.orchestration.kafka import KafkaClient


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _base_attributes(event_type: str, subject: str, correlation_id: str | None) -> Dict[str, Any]:
    event_id = uuid4().hex
    return {
        "specversion": "1.0",
        "type": event_type,
        "source": "services.chunking",
        "subject": subject,
        "time": _now_iso(),
        "id": event_id,
        "datacontenttype": "application/json",
        "correlationid": correlation_id or event_id,
    }


def _to_message(event: CloudEvent) -> Dict[str, Any]:
    if hasattr(event, "to_dict"):
        payload = event.to_dict()
    else:  # pragma: no cover - fallback mapping
        payload = dict(event)
    return payload


@dataclass(slots=True)
class ChunkingEventEmitter:
    """Publish chunking lifecycle events to Kafka."""

    kafka: KafkaClient = field(default_factory=KafkaClient)
    topic: str = "chunking.events.v1"

    def __post_init__(self) -> None:
        self.kafka.create_topics([self.topic])

    def emit_started(
        self,
        *,
        tenant_id: str,
        document_id: str,
        profile: str,
        correlation_id: str | None,
        source: str | None = None,
    ) -> CloudEvent:
        subject = f"tenant:{tenant_id}:document:{document_id}"
        attributes = _base_attributes("com.medical-kg.chunking.started", subject, correlation_id)
        data = {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "profile": profile,
            "source": source or "unknown",
            "correlation_id": attributes["correlationid"],
        }
        event = CloudEvent(attributes, data)
        self.kafka.publish(self.topic, value=_to_message(event), key=attributes["correlationid"])
        return event

    def emit_completed(
        self,
        *,
        tenant_id: str,
        document_id: str,
        profile: str,
        correlation_id: str | None,
        duration_ms: float,
        chunks: int,
    ) -> CloudEvent:
        subject = f"tenant:{tenant_id}:document:{document_id}"
        attributes = _base_attributes("com.medical-kg.chunking.completed", subject, correlation_id)
        data = {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "profile": profile,
            "duration_ms": max(duration_ms, 0.0),
            "chunks": max(chunks, 0),
            "correlation_id": attributes["correlationid"],
        }
        event = CloudEvent(attributes, data)
        self.kafka.publish(self.topic, value=_to_message(event), key=attributes["correlationid"])
        return event

    def emit_failed(
        self,
        *,
        tenant_id: str,
        document_id: str,
        profile: str,
        correlation_id: str | None,
        error_type: str,
        message: str,
    ) -> CloudEvent:
        subject = f"tenant:{tenant_id}:document:{document_id}"
        attributes = _base_attributes("com.medical-kg.chunking.failed", subject, correlation_id)
        data = {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "profile": profile,
            "error_type": error_type,
            "message": message,
            "correlation_id": attributes["correlationid"],
        }
        event = CloudEvent(attributes, data)
        self.kafka.publish(self.topic, value=_to_message(event), key=attributes["correlationid"])
        return event


__all__ = ["ChunkingEventEmitter", "CloudEvent"]
