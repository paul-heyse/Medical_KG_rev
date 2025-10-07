"""Background workers consuming orchestration topics."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from random import random
from typing import Any

import structlog

from ..gateway.models import JobEvent
from ..gateway.sse.manager import EventStreamManager
from ..observability.metrics import (
    record_dead_letter_event,
    set_dead_letter_queue_depth,
    set_orchestration_queue_depth,
)
from ..services.embedding import EmbeddingWorker as EmbeddingServiceWorker
from ..services.ingestion import IngestionService
from ..services.vector_store.service import VectorStoreService
from ..utils.logging import bind_correlation_id, get_correlation_id, reset_correlation_id
from .ingestion_pipeline import (
    ChunkingStage,
    EmbeddingStage,
    IndexingStage,
    INGEST_CHUNKING_TOPIC,
    INGEST_CHUNKS_TOPIC,
    INGEST_DLQ_TOPIC,
    INGEST_EMBEDDINGS_TOPIC,
    INGEST_INDEXED_TOPIC,
)
from .kafka import KafkaClient, KafkaMessage
from .ledger import JobLedger, JobLedgerEntry
from .pipeline import PipelineContext, PipelineExecutor, StageFailure, ensure_correlation_id
from .profiles import ProfileDetector, apply_profile_overrides
from .orchestrator import (
    DEAD_LETTER_TOPIC,
    INGEST_REQUESTS_TOPIC,
    INGEST_RESULTS_TOPIC,
    MAPPING_EVENTS_TOPIC,
    OrchestrationError,
    Orchestrator,
)

try:  # pragma: no cover - optional telemetry
    from opentelemetry.context import attach as otel_attach, detach as otel_detach
    from opentelemetry.propagate import get_global_textmap
except Exception:  # pragma: no cover - telemetry optional
    otel_attach = otel_detach = None
    _TRACE_PROPAGATOR = None
else:  # pragma: no cover - telemetry optional
    _TRACE_PROPAGATOR = get_global_textmap()


logger = structlog.get_logger(__name__)


@dataclass
class WorkerMetrics:
    processed: int = 0
    failed: int = 0
    retries: int = 0


@dataclass
class WorkerBase:
    name: str
    kafka: KafkaClient
    ledger: JobLedger
    events: EventStreamManager
    batch_size: int = 10
    metrics: WorkerMetrics = field(default_factory=WorkerMetrics)
    _stopped: bool = field(default=False, init=False)

    def shutdown(self) -> None:
        self._stopped = True

    def health(self) -> dict[str, object]:
        return {
            "name": self.name,
            "stopped": self._stopped,
            "metrics": self.metrics.__dict__.copy(),
        }

    def run_once(self) -> None:
        if self._stopped:
            return
        for message in self.kafka.consume(self.topic, max_messages=self.batch_size):
            try:
                self.process_message(message)
                self.metrics.processed += 1
            except Exception:  # pragma: no cover - worker level safety
                self.metrics.failed += 1

    # Properties implemented by subclasses
    @property
    def topic(self) -> str:  # pragma: no cover - abstract property
        raise NotImplementedError

    def process_message(self, message: KafkaMessage) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(slots=True)
class RetryPolicy:
    """Retry policy with exponential backoff and jitter."""

    max_attempts: int = 5
    base_delay_seconds: float = 1.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.1

    def delay(self, attempt: int) -> float:
        attempt = max(attempt, 1)
        delay = self.base_delay_seconds * math.pow(self.multiplier, attempt - 1)
        jitter = delay * self.jitter_ratio * (random() - 0.5) * 2
        return max(delay + jitter, self.base_delay_seconds * 0.5)


@dataclass
class PipelineWorkerBase(WorkerBase):
    """Base class for ingestion pipeline workers handling retries and metrics."""

    stage: object
    pipeline_name: str
    operation: str = "ingest"
    output_topic: str | None = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    profile_detector: ProfileDetector | None = None

    def __post_init__(self) -> None:
        executor_stage = self.stage
        if not hasattr(executor_stage, "execute"):
            raise TypeError("stage must implement execute(context)")
        self._executor = PipelineExecutor(
            [executor_stage],
            operation=self.operation,
            pipeline=self.pipeline_name,
        )

    @property
    def topic(self) -> str:  # pragma: no cover - abstract property
        raise NotImplementedError

    def run_once(self) -> None:
        if self._stopped:
            return
        try:
            depth = self.kafka.pending(self.topic)
        except Exception:  # pragma: no cover - best effort metrics
            depth = None
        else:
            set_orchestration_queue_depth(self.stage.name, depth)
        super().run_once()

    def process_message(self, message: KafkaMessage) -> None:
        payload = dict(message.value)
        job_id = payload.get("job_id")
        tenant_id = payload.get("tenant_id")
        if not job_id or not tenant_id:
            return
        otel_token = None
        if _TRACE_PROPAGATOR and otel_attach and message.headers:
            try:  # pragma: no cover - optional telemetry
                otel_ctx = _TRACE_PROPAGATOR.extract(message.headers)
                otel_token = otel_attach(otel_ctx)
            except Exception:
                otel_token = None
        correlation = message.headers.get("x-correlation-id") if message.headers else None
        if correlation or payload.get("correlation_id"):
            correlation_id = correlation or str(payload.get("correlation_id"))
            token = bind_correlation_id(correlation_id)
        else:
            correlation_id, token = ensure_correlation_id(get_correlation_id())
        try:
            payload = self._apply_profile(payload, job_id)
            entry = self.ledger.get(job_id)
            if not entry:
                entry = self.ledger.mark_processing(job_id, stage=self.stage.name)
            else:
                self.ledger.mark_processing(job_id, stage=self.stage.name)
            context = PipelineContext(
                tenant_id=tenant_id,
                operation=self.operation,
                data=payload,
                correlation_id=correlation_id,
                pipeline_version=str(payload.get("pipeline_version"))
                if payload.get("pipeline_version")
                else None,
            )
            result = self._executor.run(context)
        except StageFailure as failure:
            self._handle_stage_failure(job_id, message, failure)
        except Exception as exc:  # pragma: no cover - guardrail
            failure = StageFailure(
                "Worker failure",
                detail=str(exc),
                stage=self.stage.name,
                retriable=False,
            )
            self._handle_stage_failure(job_id, message, failure)
        else:
            self._handle_success(job_id, result, message, correlation_id)
        finally:
            reset_correlation_id(token)
            if otel_token and otel_detach:  # pragma: no cover - optional telemetry
                try:
                    otel_detach(otel_token)
                except Exception:
                    pass

    def _handle_success(
        self,
        job_id: str,
        context: PipelineContext,
        message: KafkaMessage,
        correlation_id: str | None,
    ) -> None:
        metadata = {
            "correlation_id": correlation_id,
            "pipeline_version": context.pipeline_version,
            "last_stage": self.stage.name,
        }
        metrics = context.data.get("metrics", {}).get(self.stage.name)
        if metrics:
            metadata.setdefault("stage_metrics", {})[self.stage.name] = metrics
        self.ledger.update_metadata(job_id, metadata)
        self.events.publish(
            JobEvent(
                job_id=job_id,
                type="jobs.progress",
                payload={"stage": self.stage.name, "pipeline": self.pipeline_name},
            )
        )
        if self.output_topic:
            payload = dict(context.data)
            payload.setdefault("job_id", job_id)
            payload.setdefault("tenant_id", context.tenant_id)
            headers = {"x-correlation-id": correlation_id or context.correlation_id}
            if _TRACE_PROPAGATOR:  # pragma: no cover - optional telemetry
                carrier: dict[str, str] = {}
                try:
                    _TRACE_PROPAGATOR.inject(carrier)
                except Exception:
                    carrier = {}
                headers.update({key.lower(): value for key, value in carrier.items()})
            self.kafka.publish(
                self.output_topic,
                payload,
                key=job_id,
                headers=headers,
                attempts=0,
            )

    def _handle_stage_failure(
        self,
        job_id: str,
        message: KafkaMessage,
        failure: StageFailure,
    ) -> None:
        attempts = message.attempts + 1
        if failure.retriable and attempts <= self.retry_policy.max_attempts:
            delay = self.retry_policy.delay(attempts)
            available_at = time.time() + delay
            self.kafka.publish(
                self.topic,
                dict(message.value),
                key=message.key,
                headers=message.headers or {},
                available_at=available_at,
                attempts=attempts,
            )
            self.metrics.retries += 1
            return
        problem = failure.problem
        metadata = {
            "stage": self.stage.name,
            "error": problem.to_response(),
        }
        self.ledger.mark_failed(job_id, stage=self.stage.name, reason=problem.title, metadata=metadata)
        self.kafka.publish(
            INGEST_DLQ_TOPIC,
            {**message.value, "error": problem.to_response()},
            key=message.key,
            headers=message.headers or {},
        )
        record_dead_letter_event(self.stage.name, failure.error_type)
        try:  # pragma: no cover - metrics best effort
            depth = self.kafka.pending(INGEST_DLQ_TOPIC)
        except Exception:
            depth = None
        else:
            set_dead_letter_queue_depth(depth)
        self.events.publish(
            JobEvent(
                job_id=job_id,
                type="jobs.failed",
                payload={"stage": self.stage.name, "detail": problem.to_response()},
            )
        )

    def _apply_profile(self, payload: dict[str, Any], job_id: str | None) -> dict[str, Any]:
        if not self.profile_detector:
            return payload
        if payload.get("_profile_applied"):
            return payload
        metadata: dict[str, Any] = {}
        meta_payload = payload.get("metadata")
        if isinstance(meta_payload, Mapping):
            metadata.update({k: v for k, v in meta_payload.items() if isinstance(k, str)})
        document_payload = payload.get("document")
        if isinstance(document_payload, Mapping):
            metadata.setdefault("source", document_payload.get("source"))
            for key, value in document_payload.items():
                if isinstance(key, str) and key not in metadata:
                    metadata[key] = value
        explicit = payload.get("profile")
        try:
            profile = self.profile_detector.detect(
                explicit=str(explicit) if explicit else None,
                metadata=metadata,
            )
        except KeyError as exc:
            raise StageFailure(
                "Unknown profile specified",
                status=400,
                stage=self.stage.name,
                error_type="validation",
                detail=str(exc),
            ) from exc
        updated = apply_profile_overrides(payload, profile)
        updated["_profile_applied"] = True
        pipeline = profile.ingestion_definition(self.profile_detector.manager.config)
        version = f"{pipeline.name}:{self.profile_detector.manager.config.version}"
        updated.setdefault("pipeline_version", version)
        logger.info(
            "orchestration.profile.applied",
            job_id=job_id,
            stage=self.stage.name,
            profile=profile.name,
            pipeline=pipeline.name,
        )
        return updated

class ChunkingWorker(PipelineWorkerBase):
    def __init__(
        self,
        kafka: KafkaClient,
        ledger: JobLedger,
        events: EventStreamManager,
        *,
        ingestion_service: IngestionService,
        profile_detector: ProfileDetector | None = None,
        name: str = "chunking-worker",
        batch_size: int = 5,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        stage = ChunkingStage(ingestion=ingestion_service)
        super().__init__(
            name=name,
            kafka=kafka,
            ledger=ledger,
            events=events,
            batch_size=batch_size,
            stage=stage,
            pipeline_name="ingest-chunking",
            output_topic=INGEST_CHUNKS_TOPIC,
            retry_policy=retry_policy or RetryPolicy(),
            profile_detector=profile_detector,
        )

    @property
    def topic(self) -> str:
        return INGEST_CHUNKING_TOPIC


class EmbeddingPipelineWorker(PipelineWorkerBase):
    def __init__(
        self,
        kafka: KafkaClient,
        ledger: JobLedger,
        events: EventStreamManager,
        *,
        embedding_worker: EmbeddingServiceWorker,
        namespaces: list[str] | None = None,
        models: list[str] | None = None,
        name: str = "embedding-worker",
        batch_size: int = 5,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        stage = EmbeddingStage(worker=embedding_worker, namespaces=namespaces, models=models)
        super().__init__(
            name=name,
            kafka=kafka,
            ledger=ledger,
            events=events,
            batch_size=batch_size,
            stage=stage,
            pipeline_name="ingest-embedding",
            output_topic=INGEST_EMBEDDINGS_TOPIC,
            retry_policy=retry_policy or RetryPolicy(),
        )

    @property
    def topic(self) -> str:
        return INGEST_CHUNKS_TOPIC


class IndexingWorker(PipelineWorkerBase):
    def __init__(
        self,
        kafka: KafkaClient,
        ledger: JobLedger,
        events: EventStreamManager,
        *,
        vector_service: VectorStoreService,
        batch_size: int = 5,
        name: str = "indexing-worker",
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        stage = IndexingStage(vector_service=vector_service)
        super().__init__(
            name=name,
            kafka=kafka,
            ledger=ledger,
            events=events,
            batch_size=batch_size,
            stage=stage,
            pipeline_name="ingest-indexing",
            output_topic=INGEST_INDEXED_TOPIC,
            retry_policy=retry_policy or RetryPolicy(),
        )

    @property
    def topic(self) -> str:
        return INGEST_EMBEDDINGS_TOPIC

    def _handle_success(
        self,
        job_id: str,
        context: PipelineContext,
        message: KafkaMessage,
        correlation_id: str | None,
    ) -> None:
        super()._handle_success(job_id, context, message, correlation_id)
        self.ledger.mark_completed(
            job_id,
            metadata={"index_result": context.data.get("index_result")},
        )
        self.events.publish(
            JobEvent(
                job_id=job_id,
                type="jobs.completed",
                payload={"stage": self.stage.name, "pipeline": self.pipeline_name},
            )
        )


class IngestWorker(WorkerBase):
    orchestrator: Orchestrator

    def __init__(
        self,
        orchestrator: Orchestrator,
        kafka: KafkaClient,
        ledger: JobLedger,
        events: EventStreamManager,
        *,
        name: str = "ingest-worker",
        batch_size: int = 10,
    ) -> None:
        super().__init__(
            name=name, kafka=kafka, ledger=ledger, events=events, batch_size=batch_size
        )
        self.orchestrator = orchestrator

    @property
    def topic(self) -> str:
        return INGEST_REQUESTS_TOPIC

    def process_message(self, message: KafkaMessage) -> None:
        job_id = message.value.get("job_id")  # type: ignore[assignment]
        if not job_id:
            return
        try:
            result = self.orchestrator.execute_pipeline(job_id, message.value)
            self.kafka.publish(
                INGEST_RESULTS_TOPIC, {"job_id": job_id, "result": result}, key=job_id
            )
        except OrchestrationError as exc:
            self.metrics.failed += 1
            self.events.publish(
                JobEvent(job_id=job_id, type="jobs.failed", payload={"reason": str(exc)})
            )
        except Exception as exc:  # pragma: no cover - guardrail
            self.metrics.failed += 1
            self.kafka.publish(
                DEAD_LETTER_TOPIC,
                {"job_id": job_id, "reason": str(exc)},
                key=job_id,
            )
            self.events.publish(
                JobEvent(job_id=job_id, type="jobs.failed", payload={"reason": str(exc)})
            )


class MappingWorker(WorkerBase):
    def __init__(
        self,
        kafka: KafkaClient,
        ledger: JobLedger,
        events: EventStreamManager,
        *,
        name: str = "mapping-worker",
        batch_size: int = 10,
    ) -> None:
        super().__init__(
            name=name, kafka=kafka, ledger=ledger, events=events, batch_size=batch_size
        )

    @property
    def topic(self) -> str:
        return MAPPING_EVENTS_TOPIC

    def process_message(self, message: KafkaMessage) -> None:
        job_id = message.value.get("job_id")  # type: ignore[assignment]
        if not job_id:
            return
        entry = self.ledger.get(job_id)
        if not entry:
            return
        self.ledger.update_metadata(job_id, {"mapping": True, "stage": "mapping"})
        self.events.publish(
            JobEvent(job_id=job_id, type="jobs.progress", payload={"stage": "mapping"})
        )


__all__ = [
    "ChunkingWorker",
    "EmbeddingPipelineWorker",
    "IndexingWorker",
    "IngestWorker",
    "MappingWorker",
    "PipelineWorkerBase",
    "RetryPolicy",
    "WorkerBase",
    "WorkerMetrics",
]
