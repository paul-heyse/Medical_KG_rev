"""High level orchestration for ingestion pipelines."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from ..gateway.models import JobEvent
from ..gateway.sse.manager import EventStreamManager
from .kafka import KafkaClient
from .ledger import JobLedger, JobLedgerEntry

INGEST_REQUESTS_TOPIC = "ingest.requests.v1"
INGEST_RESULTS_TOPIC = "ingest.results.v1"
MAPPING_EVENTS_TOPIC = "mapping.events.v1"
DEAD_LETTER_TOPIC = "ingest.deadletter.v1"


class OrchestrationError(RuntimeError):
    pass


@dataclass
class PipelineStage:
    name: str
    handler: Callable[[JobLedgerEntry, dict[str, object]], dict[str, object]]
    emits_mapping_event: bool = False


@dataclass
class Pipeline:
    name: str
    stages: list[PipelineStage]


PRIORITY_HEADERS = {"low": "0", "normal": "1", "high": "2"}
BASE_BACKOFF_SECONDS = 1.0
MAX_RETRIES = 3


class Orchestrator:
    """Coordinates ingestion pipelines backed by Kafka topics."""

    def __init__(
        self,
        kafka: KafkaClient,
        ledger: JobLedger,
        events: EventStreamManager,
    ) -> None:
        self.kafka = kafka
        self.ledger = ledger
        self.events = events
        self.pipelines: dict[str, Pipeline] = {}
        self._register_default_pipelines()

    # ------------------------------------------------------------------
    # Pipeline registration
    # ------------------------------------------------------------------
    def _register_default_pipelines(self) -> None:
        self.register_pipeline(
            Pipeline(
                name="auto",
                stages=[
                    PipelineStage("metadata", self._handle_metadata),
                    PipelineStage("chunk", self._handle_chunking),
                    PipelineStage("embed", self._handle_embedding),
                    PipelineStage("index", self._handle_indexing),
                ],
            )
        )
        self.register_pipeline(
            Pipeline(
                name="two-phase",
                stages=[
                    PipelineStage("metadata", self._handle_metadata),
                    PipelineStage("pdf-fetch", self._handle_pdf_fetch),
                    PipelineStage("mineru", self._handle_mineru_parse),
                    PipelineStage("postpdf", self._handle_post_pdf, emits_mapping_event=True),
                ],
            )
        )
        self.register_pipeline(
            Pipeline(
                name="adapter-chain",
                stages=[
                    PipelineStage("openalex", self._handle_adapter_chain),
                    PipelineStage("mineru", self._handle_mineru_parse),
                    PipelineStage("postpdf", self._handle_post_pdf, emits_mapping_event=True),
                ],
            )
        )

    def register_pipeline(self, pipeline: Pipeline) -> None:
        self.pipelines[pipeline.name] = pipeline

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------
    def submit_job(
        self,
        *,
        tenant_id: str,
        dataset: str,
        item: dict[str, object],
        priority: str = "normal",
        metadata: dict[str, object] | None = None,
    ) -> JobLedgerEntry:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        doc_key = f"{dataset}:{item.get('id', uuid.uuid4().hex)}"
        pipeline_name = self._resolve_pipeline(item)
        ledger_entry = self.ledger.idempotent_create(
            job_id=job_id,
            doc_key=doc_key,
            tenant_id=tenant_id,
            pipeline=pipeline_name,
            metadata={"dataset": dataset, "item": item, **(metadata or {})},
        )
        if ledger_entry.job_id != job_id:
            # Existing job, do not enqueue again.
            ledger_entry.metadata.setdefault("duplicate", True)
            return ledger_entry
        headers = {"x-priority": PRIORITY_HEADERS.get(priority, "1")}
        payload = {
            "job_id": job_id,
            "tenant_id": tenant_id,
            "dataset": dataset,
            "pipeline": pipeline_name,
        }
        self.kafka.publish(
            INGEST_REQUESTS_TOPIC,
            payload,
            key=job_id,
            headers=headers,
        )
        self.events.publish(
            JobEvent(job_id=job_id, type="jobs.started", payload={"pipeline": pipeline_name})
        )
        return self.ledger.get(job_id) or ledger_entry

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute_pipeline(self, job_id: str, payload: dict[str, object]) -> dict[str, object]:
        entry = self.ledger.get(job_id)
        if not entry:
            raise OrchestrationError(f"Job {job_id} not found in ledger")
        pipeline = self.pipelines.get(entry.pipeline or "")
        if not pipeline:
            raise OrchestrationError(f"Pipeline {entry.pipeline} not registered")
        context: dict[str, object] = dict(payload)
        for stage in pipeline.stages:
            try:
                self.ledger.mark_processing(job_id, stage=stage.name)
                self.events.publish(
                    JobEvent(job_id=job_id, type="jobs.progress", payload={"stage": stage.name})
                )
                context = stage.handler(entry, context)
                if stage.emits_mapping_event:
                    self.kafka.publish(
                        MAPPING_EVENTS_TOPIC, {"job_id": job_id, "stage": stage.name}
                    )
            except Exception as exc:  # pragma: no cover - defensive
                self._handle_stage_failure(job_id, stage.name, exc, context)
                raise OrchestrationError(str(exc))
        self.ledger.mark_completed(job_id, metadata={"result": context})
        self.events.publish(
            JobEvent(job_id=job_id, type="jobs.completed", payload={"pipeline": entry.pipeline})
        )
        return context

    def cancel_job(self, job_id: str, reason: str | None = None) -> JobLedgerEntry | None:
        entry = self.ledger.get(job_id)
        if not entry or entry.is_terminal():
            return entry
        self.kafka.discard(INGEST_REQUESTS_TOPIC, key=job_id)
        updated = self.ledger.mark_cancelled(job_id, reason=reason)
        self.events.publish(
            JobEvent(job_id=job_id, type="jobs.failed", payload={"reason": reason or "cancelled"})
        )
        return updated

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------
    def _handle_stage_failure(
        self,
        job_id: str,
        stage: str,
        exc: Exception,
        context: dict[str, object],
    ) -> None:
        entry = self.ledger.get(job_id)
        attempts = self.ledger.record_attempt(job_id)
        if attempts <= MAX_RETRIES:
            delay = BASE_BACKOFF_SECONDS * math.pow(2, attempts - 1)
            available_at = time.time() + delay
            self.kafka.publish(
                INGEST_REQUESTS_TOPIC,
                {"job_id": job_id, "pipeline": entry.pipeline if entry else None},
                key=job_id,
                headers={"x-priority": "1"},
                available_at=available_at,
                attempts=attempts,
            )
            self.ledger.mark_processing(job_id, stage="retry")
            self.events.publish(
                JobEvent(
                    job_id=job_id,
                    type="jobs.progress",
                    payload={"stage": stage, "retry_in": delay},
                )
            )
        else:
            self.ledger.mark_failed(
                job_id,
                stage=stage,
                reason=str(exc),
                metadata={"context": context},
            )
            self.kafka.publish(
                DEAD_LETTER_TOPIC,
                {"job_id": job_id, "reason": str(exc)},
                key=job_id,
            )
            self.events.publish(
                JobEvent(job_id=job_id, type="jobs.failed", payload={"reason": str(exc)})
            )

    # ------------------------------------------------------------------
    # Pipeline handlers
    # ------------------------------------------------------------------
    def _handle_metadata(
        self, entry: JobLedgerEntry, context: dict[str, object]
    ) -> dict[str, object]:
        context = dict(context)
        context.setdefault("metadata_processed", True)
        context["pipeline"] = entry.pipeline
        return context

    def _handle_chunking(self, _: JobLedgerEntry, context: dict[str, object]) -> dict[str, object]:
        context = dict(context)
        context["chunks"] = 4
        return context

    def _handle_embedding(self, _: JobLedgerEntry, context: dict[str, object]) -> dict[str, object]:
        context = dict(context)
        context["embeddings"] = 4
        return context

    def _handle_indexing(self, _: JobLedgerEntry, context: dict[str, object]) -> dict[str, object]:
        context = dict(context)
        context["indexed"] = True
        return context

    def _handle_pdf_fetch(self, _: JobLedgerEntry, context: dict[str, object]) -> dict[str, object]:
        context = dict(context)
        context["pdf_fetched"] = True
        return context

    def _handle_mineru_parse(
        self, _: JobLedgerEntry, context: dict[str, object]
    ) -> dict[str, object]:
        context = dict(context)
        context["mineru_parsed"] = True
        return context

    def _handle_post_pdf(self, _: JobLedgerEntry, context: dict[str, object]) -> dict[str, object]:
        context = dict(context)
        context["post_pdf_completed"] = True
        return context

    def _handle_adapter_chain(
        self, entry: JobLedgerEntry, context: dict[str, object]
    ) -> dict[str, object]:
        context = dict(context)
        adapters = ["OpenAlex", "Unpaywall", "CORE", "MinerU"]
        context["adapter_chain"] = adapters
        if entry.metadata.get("item", {}).get("open_access"):
            context["pdf_fetched"] = True
        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_pipeline(self, item: dict[str, object]) -> str:
        if item.get("adapter_chain"):
            return "adapter-chain"
        if item.get("document_type") == "pdf":
            return "two-phase"
        return "auto"


__all__ = [
    "DEAD_LETTER_TOPIC",
    "INGEST_REQUESTS_TOPIC",
    "INGEST_RESULTS_TOPIC",
    "MAPPING_EVENTS_TOPIC",
    "OrchestrationError",
    "Orchestrator",
    "Pipeline",
    "PipelineStage",
]
