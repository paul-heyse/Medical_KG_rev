"""CloudEvents emission utilities for chunking lifecycle."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

try:  # pragma: no cover - optional dependency for structured CloudEvents
    from cloudevents.http import CloudEvent  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback used in unit tests
    class CloudEvent(dict):  # type: ignore[override]
        """Minimal CloudEvent shim when the official dependency is unavailable."""

        def __init__(self, attributes: dict[str, Any], data: dict[str, Any]):
            super().__init__(attributes)
            self.data = data

        def keys(self):
            return super().keys()

try:  # pragma: no cover - orchestration Kafka client optional in CI
    from Medical_KG_rev.orchestration.kafka import KafkaClient
except Exception:  # pragma: no cover - lightweight fallback
    class KafkaClient:  # type: ignore[override]
        """In-memory Kafka stub used when the orchestration stack is unavailable."""

        def __init__(self) -> None:
            self._topics: dict[str, list[dict[str, Any]]] = {}

        def create_topics(self, topics):
            for topic in topics:
                self._topics.setdefault(topic, [])

        def publish(self, topic: str, value: dict[str, Any], *, key=None, headers=None):
            self._topics.setdefault(topic, []).append(value)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _base_attributes(event_type: str, *, subject: str, correlation_id: str | None) -> dict[str, Any]:
    event_id = uuid.uuid4().hex
    return {
        "specversion": "1.0",
        "type": event_type,
        "source": "services.chunking",
        "subject": subject,
        "time": _now(),
        "id": event_id,
        "datacontenttype": "application/json",
        "correlationid": correlation_id or event_id,
    }


def _to_message(event: CloudEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {key: event.get(key) for key in event.keys()}  # type: ignore[arg-type]
    payload["data"] = event.data
    return payload


@dataclass(slots=True)
class ChunkingEventEmitter:
    """Publishes chunking lifecycle CloudEvents to Kafka."""

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
        attributes = _base_attributes(
            "com.medical-kg.chunking.started",
            subject=subject,
            correlation_id=correlation_id,
        )
        data = {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "profile": profile,
            "source": source or "unknown",
            "correlation_id": attributes["correlationid"],
        }
        event = CloudEvent(attributes, data)
        self.kafka.publish(
            self.topic,
            value=_to_message(event),
            key=attributes["correlationid"],
            headers={"content-type": "application/cloudevents+json"},
        )
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
        average_tokens: float | None,
        average_chars: float | None,
    ) -> CloudEvent:
        subject = f"tenant:{tenant_id}:document:{document_id}"
        attributes = _base_attributes(
            "com.medical-kg.chunking.completed",
            subject=subject,
            correlation_id=correlation_id,
        )
        data = {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "profile": profile,
            "duration_ms": max(duration_ms, 0.0),
            "chunks": max(chunks, 0),
            "average_token_count": max(average_tokens or 0.0, 0.0),
            "average_char_count": max(average_chars or 0.0, 0.0),
            "correlation_id": attributes["correlationid"],
        }
        event = CloudEvent(attributes, data)
        self.kafka.publish(
            self.topic,
            value=_to_message(event),
            key=attributes["correlationid"],
            headers={"content-type": "application/cloudevents+json"},
        )
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
        attributes = _base_attributes(
            "com.medical-kg.chunking.failed",
            subject=subject,
            correlation_id=correlation_id,
        )
        data = {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "profile": profile,
            "error_type": error_type,
            "message": message,
            "correlation_id": attributes["correlationid"],
        }
        event = CloudEvent(attributes, data)
        self.kafka.publish(
            self.topic,
            value=_to_message(event),
            key=attributes["correlationid"],
            headers={"content-type": "application/cloudevents+json"},
        )
        return event

    def emit_mineru_gate_waiting(
        self,
        *,
        tenant_id: str,
        job_id: str,
        document_id: str,
        reason: str,
    ) -> CloudEvent:
        subject = f"tenant:{tenant_id}:job:{job_id}"
        attributes = _base_attributes(
            "com.medical-kg.mineru.gate.waiting",
            subject=subject,
            correlation_id=None,
        )
        data = {
            "tenant_id": tenant_id,
            "job_id": job_id,
            "document_id": document_id,
            "reason": reason,
            "correlation_id": attributes["correlationid"],
        }
        event = CloudEvent(attributes, data)
        self.kafka.publish(
            self.topic,
            value=_to_message(event),
            key=attributes["correlationid"],
            headers={"content-type": "application/cloudevents+json"},
        )
        return event

    def emit_postpdf_start_triggered(
        self,
        *,
        job_id: str,
        triggered_by: str,
    ) -> CloudEvent:
        subject = f"job:{job_id}"
        attributes = _base_attributes(
            "com.medical-kg.postpdf.start.triggered",
            subject=subject,
            correlation_id=None,
        )
        data = {
            "job_id": job_id,
            "triggered_by": triggered_by,
            "correlation_id": attributes["correlationid"],
        }
        event = CloudEvent(attributes, data)
        self.kafka.publish(
            self.topic,
            value=_to_message(event),
            key=attributes["correlationid"],
            headers={"content-type": "application/cloudevents+json"},
        )
        return event


__all__ = ["ChunkingEventEmitter"]
