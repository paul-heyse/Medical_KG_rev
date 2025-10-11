"""Events for embedding service operations."""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from Medical_KG_rev.utils.fallbacks import fallback_unavailable

try:
    from cloudevents.http import CloudEvent  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    fallback_unavailable("Embedding CloudEvent dependency", exc)

try:
    from Medical_KG_rev.orchestration.kafka import KafkaClient
except Exception as exc:  # pragma: no cover - fallback for lightweight envs
    fallback_unavailable("Embedding event Kafka client", exc)

logger = structlog.get_logger(__name__)


class EmbeddingEventEmitter:
    """Event emitter for embedding operations."""

    def __init__(self, kafka_client: KafkaClient | None = None) -> None:
        """Initialize the embedding event emitter."""
        self.kafka_client = kafka_client or KafkaClient()
        self.logger = logger

    def emit_embedding_started(
        self,
        namespace: str,
        model: str,
        text_count: int,
        tenant_id: str | None = None,
    ) -> None:
        """Emit embedding started event."""
        event_data = {
            "namespace": namespace,
            "model": model,
            "text_count": text_count,
            "tenant_id": tenant_id,
            "timestamp": time.time(),
            "event_type": "embedding_started",
        }

        self._publish_event("embedding.events", event_data)
        self.logger.info(
            "embedding.started",
            namespace=namespace,
            model=model,
            text_count=text_count,
            tenant_id=tenant_id,
        )

    def emit_embedding_completed(
        self,
        namespace: str,
        model: str,
        text_count: int,
        embedding_count: int,
        duration_ms: float,
        tenant_id: str | None = None,
    ) -> None:
        """Emit embedding completed event."""
        event_data = {
            "namespace": namespace,
            "model": model,
            "text_count": text_count,
            "embedding_count": embedding_count,
            "duration_ms": duration_ms,
            "tenant_id": tenant_id,
            "timestamp": time.time(),
            "event_type": "embedding_completed",
        }

        self._publish_event("embedding.events", event_data)
        self.logger.info(
            "embedding.completed",
            namespace=namespace,
            model=model,
            text_count=text_count,
            embedding_count=embedding_count,
            duration_ms=duration_ms,
            tenant_id=tenant_id,
        )

    def emit_embedding_failed(
        self,
        namespace: str,
        model: str,
        error: str,
        tenant_id: str | None = None,
    ) -> None:
        """Emit embedding failed event."""
        event_data = {
            "namespace": namespace,
            "model": model,
            "error": error,
            "tenant_id": tenant_id,
            "timestamp": time.time(),
            "event_type": "embedding_failed",
        }

        self._publish_event("embedding.events", event_data)
        self.logger.error(
            "embedding.failed",
            namespace=namespace,
            model=model,
            error=error,
            tenant_id=tenant_id,
        )

    def emit_namespace_created(
        self,
        namespace: str,
        config: dict[str, Any],
        tenant_id: str | None = None,
    ) -> None:
        """Emit namespace created event."""
        event_data = {
            "namespace": namespace,
            "config": config,
            "tenant_id": tenant_id,
            "timestamp": time.time(),
            "event_type": "namespace_created",
        }

        self._publish_event("embedding.events", event_data)
        self.logger.info(
            "namespace.created",
            namespace=namespace,
            config=config,
            tenant_id=tenant_id,
        )

    def emit_namespace_deleted(
        self,
        namespace: str,
        tenant_id: str | None = None,
    ) -> None:
        """Emit namespace deleted event."""
        event_data = {
            "namespace": namespace,
            "tenant_id": tenant_id,
            "timestamp": time.time(),
            "event_type": "namespace_deleted",
        }

        self._publish_event("embedding.events", event_data)
        self.logger.info(
            "namespace.deleted",
            namespace=namespace,
            tenant_id=tenant_id,
        )

    def emit_model_loaded(
        self,
        model: str,
        namespace: str,
        load_time_ms: float,
        tenant_id: str | None = None,
    ) -> None:
        """Emit model loaded event."""
        event_data = {
            "model": model,
            "namespace": namespace,
            "load_time_ms": load_time_ms,
            "tenant_id": tenant_id,
            "timestamp": time.time(),
            "event_type": "model_loaded",
        }

        self._publish_event("embedding.events", event_data)
        self.logger.info(
            "model.loaded",
            model=model,
            namespace=namespace,
            load_time_ms=load_time_ms,
            tenant_id=tenant_id,
        )

    def emit_model_unloaded(
        self,
        model: str,
        namespace: str,
        tenant_id: str | None = None,
    ) -> None:
        """Emit model unloaded event."""
        event_data = {
            "model": model,
            "namespace": namespace,
            "tenant_id": tenant_id,
            "timestamp": time.time(),
            "event_type": "model_unloaded",
        }

        self._publish_event("embedding.events", event_data)
        self.logger.info(
            "model.unloaded",
            model=model,
            namespace=namespace,
            tenant_id=tenant_id,
        )

    def emit_embedding_cache_hit(
        self,
        namespace: str,
        model: str,
        cache_key: str,
        tenant_id: str | None = None,
    ) -> None:
        """Emit embedding cache hit event."""
        event_data = {
            "namespace": namespace,
            "model": model,
            "cache_key": cache_key,
            "tenant_id": tenant_id,
            "timestamp": time.time(),
            "event_type": "embedding_cache_hit",
        }

        self._publish_event("embedding.events", event_data)
        self.logger.debug(
            "embedding.cache.hit",
            namespace=namespace,
            model=model,
            cache_key=cache_key,
            tenant_id=tenant_id,
        )

    def emit_embedding_cache_miss(
        self,
        namespace: str,
        model: str,
        cache_key: str,
        tenant_id: str | None = None,
    ) -> None:
        """Emit embedding cache miss event."""
        event_data = {
            "namespace": namespace,
            "model": model,
            "cache_key": cache_key,
            "tenant_id": tenant_id,
            "timestamp": time.time(),
            "event_type": "embedding_cache_miss",
        }

        self._publish_event("embedding.events", event_data)
        self.logger.debug(
            "embedding.cache.miss",
            namespace=namespace,
            model=model,
            cache_key=cache_key,
            tenant_id=tenant_id,
        )

    def _publish_event(self, topic: str, event_data: dict[str, Any]) -> None:
        """Publish event to Kafka topic."""
        try:
            if CloudEvent:
                # Create CloudEvent
                event = CloudEvent(
                    type=event_data["event_type"],
                    source="embedding-service",
                    data=event_data,
                )
                self.kafka_client.publish(topic, event)
            else:
                # Fallback to simple dict
                self.kafka_client.publish(topic, event_data)
        except Exception as exc:
            self.logger.warning(f"Failed to publish event to {topic}: {exc}")

    def health_check(self) -> dict[str, Any]:
        """Check event emitter health."""
        return {
            "emitter": "embedding",
            "status": "healthy",
            "kafka_client": "available" if self.kafka_client else "unavailable",
            "cloudevents": "available" if CloudEvent else "unavailable",
        }


# Global event emitter instance
_embedding_event_emitter: EmbeddingEventEmitter | None = None


def get_embedding_event_emitter() -> EmbeddingEventEmitter:
    """Get the global embedding event emitter instance."""
    global _embedding_event_emitter

    if _embedding_event_emitter is None:
        _embedding_event_emitter = EmbeddingEventEmitter()

    return _embedding_event_emitter


def create_embedding_event_emitter(kafka_client: KafkaClient | None = None) -> EmbeddingEventEmitter:
    """Create a new embedding event emitter instance."""
    return EmbeddingEventEmitter(kafka_client)
