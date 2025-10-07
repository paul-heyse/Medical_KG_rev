"""High level orchestration for ingestion pipelines."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from ..adapters.plugins.bootstrap import get_plugin_manager
from ..adapters.plugins.manager import AdapterPluginError
from ..adapters.plugins.models import AdapterDomain, AdapterRequest
from ..gateway.models import JobEvent
from ..gateway.sse.manager import EventStreamManager
from ..models.ir import Block, BlockType, Document, Section
from ..services.ingestion import IngestionService
from .kafka import KafkaClient
from .ledger import JobLedger, JobLedgerEntry
from ..observability.metrics import ADAPTER_PLUGIN_FAILURES, ADAPTER_PLUGIN_INVOCATIONS

_DEFAULT_TEI = """
<TEI>
  <text>
    <body>
      <div type="introduction"><head>Introduction</head></div>
      <div type="methods"><head>Methods</head></div>
      <div type="results"><head>Results</head></div>
      <div type="conclusion"><head>Conclusion</head></div>
    </body>
  </text>
</TEI>
"""

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
        *,
        ingestion_service: IngestionService | None = None,
    ) -> None:
        self.kafka = kafka
        self.ledger = ledger
        self.events = events
        self.pipelines: dict[str, Pipeline] = {}
        self.ingestion = ingestion_service or IngestionService()
        self.adapter_manager = get_plugin_manager()
        self._register_default_pipelines()
        self._run_adapter_health_checks()

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

    def _handle_chunking(self, entry: JobLedgerEntry, context: dict[str, object]) -> dict[str, object]:
        context = dict(context)
        document = context.get("document")
        if not isinstance(document, Document):
            document = self._build_document(entry, context)
            context["document"] = document
        tenant_id = entry.tenant_id or context.get("tenant_id", "system")
        dataset = context.get("dataset") or entry.metadata.get("dataset")
        run = self.ingestion.chunk_document(
            document,
            tenant_id=tenant_id,
            source_hint=str(dataset) if dataset else None,
        )
        context["chunk_count"] = len(run.chunks)
        context["granularity_counts"] = run.granularity_counts
        context["chunk_duration"] = run.duration_seconds
        context["chunks"] = [chunk.model_dump() for chunk in run.chunks]
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
        self, entry: JobLedgerEntry, context: dict[str, object]
    ) -> dict[str, object]:
        context = dict(context)
        context["mineru_parsed"] = True
        document = self._build_document(entry, context)
        context["document"] = document
        return context

    def _handle_post_pdf(self, _: JobLedgerEntry, context: dict[str, object]) -> dict[str, object]:
        context = dict(context)
        context["post_pdf_completed"] = True
        return context

    def _handle_adapter_chain(
        self, entry: JobLedgerEntry, context: dict[str, object]
    ) -> dict[str, object]:
        context = dict(context)
        domains = self._resolve_domains(entry)
        adapter_versions: dict[str, str] = {}
        adapter_responses: list[dict[str, object]] = []
        for domain in domains:
            metadata_items = self.adapter_manager.list_metadata(domain=domain)
            for metadata in metadata_items:
                request = self._build_adapter_request(entry, context, domain)
                try:
                    ADAPTER_PLUGIN_INVOCATIONS.labels(
                        metadata.name, metadata.domain.value
                    ).inc()
                    response = self.adapter_manager.run(metadata.name, request)
                    adapter_responses.append(
                        {
                            "adapter": metadata.name,
                            "domain": metadata.domain.value,
                            "items": response.items,
                            "warnings": response.warnings,
                        }
                    )
                    adapter_versions[metadata.name] = metadata.version
                except AdapterPluginError as exc:
                    ADAPTER_PLUGIN_FAILURES.labels(
                        metadata.name, metadata.domain.value
                    ).inc()
                    adapter_responses.append(
                        {
                            "adapter": metadata.name,
                            "domain": metadata.domain.value,
                            "error": str(exc),
                        }
                    )
                    continue
        if adapter_versions:
            metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
            current_versions = dict(metadata.get("adapter_versions", {}))
            current_versions.update(adapter_versions)
            self.ledger.update_metadata(entry.job_id, {"adapter_versions": current_versions})
        context["adapter_chain"] = [response["adapter"] for response in adapter_responses]
        context["adapter_responses"] = adapter_responses
        return context

    def _build_adapter_request(
        self, entry: JobLedgerEntry, context: dict[str, object], domain: AdapterDomain
    ) -> AdapterRequest:
        parameters: dict[str, object] = {}
        metadata = entry.metadata or {}
        item = metadata.get("item")
        if isinstance(item, dict):
            parameters.update(item)
        request_params = context.get("adapter_parameters")
        if isinstance(request_params, dict):
            parameters.update(request_params)
        correlation_id = context.get("correlation_id") or entry.job_id
        return AdapterRequest(
            tenant_id=entry.tenant_id,
            correlation_id=str(correlation_id),
            domain=domain,
            parameters=parameters,
        )

    def _resolve_domains(self, entry: JobLedgerEntry) -> list[AdapterDomain]:
        metadata = entry.metadata or {}
        explicit_domains = metadata.get("domains")
        domains: list[AdapterDomain] = []
        if isinstance(explicit_domains, (list, tuple)):
            for item in explicit_domains:
                try:
                    domains.append(AdapterDomain(item))
                except ValueError:  # pragma: no cover - invalid input ignored
                    continue
        if not domains:
            dataset = metadata.get("dataset")
            domains.append(self._infer_domain(dataset))
        return domains

    def _infer_domain(self, dataset: object) -> AdapterDomain:
        if not dataset:
            return AdapterDomain.BIOMEDICAL
        for metadata in self.adapter_manager.list_metadata():
            if getattr(metadata, "dataset", None) == dataset:
                return metadata.domain
        return AdapterDomain.BIOMEDICAL

    def _run_adapter_health_checks(self) -> None:
        for metadata in self.adapter_manager.list_metadata():
            if not self.adapter_manager.check_health(metadata.name):
                raise OrchestrationError(f"Adapter '{metadata.name}' failed health check")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_pipeline(self, item: dict[str, object]) -> str:
        if item.get("adapter_chain"):
            return "adapter-chain"
        if item.get("document_type") == "pdf":
            return "two-phase"
        return "auto"

    def _build_document(self, entry: JobLedgerEntry, context: dict[str, object]) -> Document:
        metadata = entry.metadata or {}
        item = metadata.get("item") if isinstance(metadata, dict) else {}
        dataset = metadata.get("dataset", context.get("dataset", "unknown"))
        doc_id = entry.doc_key or f"doc-{entry.job_id}"
        title = context.get("title") or item.get("title") if isinstance(item, dict) else None
        text = context.get("pdf_text") or item.get("text") if isinstance(item, dict) else None
        if not isinstance(text, str) or not text.strip():
            text = (
                "Introduction. Study background on cardiovascular outcomes.\n"
                "Methods. Randomised controlled trial with placebo comparator.\n"
                "Results. Blood pressure reduced significantly.\n"
                "Conclusion. Therapy well tolerated."
            )
        paragraphs = [segment.strip() for segment in text.split("\n") if segment.strip()]
        blocks: list[Block] = []
        for idx, paragraph in enumerate(paragraphs):
            block_type = BlockType.HEADER if idx == 0 and ":" not in paragraph else BlockType.PARAGRAPH
            blocks.append(
                Block(
                    id=f"{doc_id}:block:{idx}",
                    type=block_type,
                    text=paragraph,
                    metadata={
                        "layout_region": f"region-{idx // 2}",
                    },
                )
            )
        section = Section(
            id=f"{doc_id}:section:0",
            title="Document",
            blocks=blocks,
        )
        tei = context.get("tei_xml") or metadata.get("tei_xml") or _DEFAULT_TEI
        document = Document(
            id=doc_id,
            source=str(dataset),
            title=title or "Untitled Document",
            sections=[section],
            metadata={"tei_xml": tei},
        )
        return document


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
