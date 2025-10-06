"""Background workers consuming orchestration topics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..gateway.models import JobEvent
from ..gateway.sse.manager import EventStreamManager
from .kafka import KafkaClient, KafkaMessage
from .ledger import JobLedger
from .orchestrator import (
    DEAD_LETTER_TOPIC,
    INGEST_REQUESTS_TOPIC,
    INGEST_RESULTS_TOPIC,
    MAPPING_EVENTS_TOPIC,
    OrchestrationError,
    Orchestrator,
)


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

    def health(self) -> Dict[str, object]:
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
        super().__init__(name=name, kafka=kafka, ledger=ledger, events=events, batch_size=batch_size)
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
            self.kafka.publish(INGEST_RESULTS_TOPIC, {"job_id": job_id, "result": result}, key=job_id)
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
        super().__init__(name=name, kafka=kafka, ledger=ledger, events=events, batch_size=batch_size)

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


__all__ = ["WorkerBase", "IngestWorker", "MappingWorker", "WorkerMetrics"]
