"""CloudEvents emission utilities for embedding lifecycle."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

try:  # pragma: no cover - optional dependency for structured CloudEvents
    from cloudevents.http import CloudEvent  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback used in tests without dependency
    class CloudEvent(dict):  # type: ignore[override]
        """Minimal CloudEvent fallback supporting the mapping interface."""

        def __init__(self, attributes: dict[str, Any], data: dict[str, Any]):
            super().__init__(attributes)
            self.data = data

        def keys(self):
            return super().keys()

try:  # pragma: no cover - orchestration client optional in unit tests
    from Medical_KG_rev.orchestration.kafka import KafkaClient
except Exception:  # pragma: no cover - fallback for lightweight envs
    class KafkaClient:  # type: ignore[override]
        """Minimal in-memory Kafka facade used when orchestration stack unavailable."""

        def __init__(self) -> None:
            self._topics: dict[str, list[dict[str, Any]]] = {}

        def create_topics(self, topics):
            for topic in topics:
                self._topics.setdefault(topic, [])

        def publish(self, topic: str, value: dict[str, Any], *, key=None, headers=None):
            self._topics.setdefault(topic, []).append(value)



def _now() -> str:
    return datetime.now(UTC).isoformat()


def _base_attributes(event_type: str, *, namespace: str, correlation_id: str | None) -> dict[str, Any]:
    event_id = str(uuid.uuid4())
    return {
        "specversion": "1.0",
        "type": event_type,
        "source": "services.embedding.worker",
        "subject": namespace,
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
class EmbeddingEventEmitter:
    """Publishes embedding lifecycle CloudEvents to Kafka."""

    kafka: KafkaClient
    topic: str = "embedding.events.v1"

    def __post_init__(self) -> None:
        self.kafka.create_topics([self.topic])

    def emit_started(
        self,
        *,
        tenant_id: str,
        namespace: str,
        provider: str,
        batch_size: int,
        correlation_id: str | None,
    ) -> CloudEvent:
        attributes = _base_attributes(
            "com.medical-kg.embedding.started",
            namespace=namespace,
            correlation_id=correlation_id,
        )
        data = {
            "tenant_id": tenant_id,
            "namespace": namespace,
            "provider": provider,
            "batch_size": batch_size,
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
        namespace: str,
        provider: str,
        correlation_id: str | None,
        duration_ms: float,
        generated: int,
        cache_hits: int,
        cache_misses: int,
    ) -> CloudEvent:
        attributes = _base_attributes(
            "com.medical-kg.embedding.completed",
            namespace=namespace,
            correlation_id=correlation_id,
        )
        data = {
            "tenant_id": tenant_id,
            "namespace": namespace,
            "provider": provider,
            "duration_ms": duration_ms,
            "embeddings_generated": generated,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
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
        namespace: str,
        provider: str,
        correlation_id: str | None,
        error_type: str,
        message: str,
    ) -> CloudEvent:
        attributes = _base_attributes(
            "com.medical-kg.embedding.failed",
            namespace=namespace,
            correlation_id=correlation_id,
        )
        data = {
            "tenant_id": tenant_id,
            "namespace": namespace,
            "provider": provider,
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


__all__ = ["EmbeddingEventEmitter"]

