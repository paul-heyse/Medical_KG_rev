"""Built-in stage plugin implementations for Dagster orchestration."""

from __future__ import annotations

from typing import Any, Mapping

import structlog

from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.stages import (
    AdapterIngestStage,
    AdapterParseStage,
    HaystackPipelineResource,
    IRValidationStage,
    NoOpExtractStage,
    NoOpKnowledgeGraphStage,
)
from Medical_KG_rev.orchestration.haystack.components import (
    HaystackChunker,
    HaystackEmbedder,
    HaystackIndexWriter,
)
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.contracts import (
    DownloadArtifact,
    GateDecision,
    PipelineGateNotReady,
    PipelineState,
    StageContext,
)
from Medical_KG_rev.orchestration.stages.plugin_manager import (
    StagePlugin,
    StagePluginContext,
    StagePluginMetadata,
)

logger = structlog.get_logger(__name__)


class CoreStagePlugin(StagePlugin):
    """Provides the default ingest→KG stage implementations."""

    def __init__(self) -> None:
        super().__init__(
            StagePluginMetadata(
                name="core-stages",
                version="1.0.0",
                stage_types=(
                    "ingest",
                    "parse",
                    "ir-validation",
                    "chunk",
                    "embed",
                    "index",
                    "extract",
                    "knowledge-graph",
                ),
                description="Built-in stage implementations",
            )
        )
        self._adapter_manager: AdapterPluginManager | None = None
        self._pipeline_resource: HaystackPipelineResource | None = None

    def initialise(self, context: StagePluginContext) -> None:
        self._adapter_manager = context.require("adapter_manager")
        self._pipeline_resource = context.require("haystack_pipeline")

    def health_check(self, context: StagePluginContext) -> None:
        if not isinstance(self._adapter_manager, AdapterPluginManager):
            raise RuntimeError("Adapter manager not available for core stage plugin")
        if not isinstance(self._pipeline_resource, HaystackPipelineResource):
            raise RuntimeError("Haystack pipeline resource not available")

    def create_stage(self, definition: StageDefinition, context: StagePluginContext) -> object:
        assert self._adapter_manager is not None
        assert self._pipeline_resource is not None

        stage_type = definition.stage_type
        config: Mapping[str, Any] = definition.config
        if stage_type == "ingest":
            adapter = config.get("adapter")
            if not adapter:
                raise ValueError(f"Stage '{definition.name}' requires an adapter")
            strict = bool(config.get("strict", False))
            domain = config.get("domain")
            parameters = config.get("parameters", {}) if isinstance(config, Mapping) else {}
            return AdapterIngestStage(
                self._adapter_manager,
                adapter_name=str(adapter),
                strict=strict,
                default_domain=domain,
                extra_parameters=parameters if isinstance(parameters, Mapping) else {},
            )
        if stage_type == "parse":
            return AdapterParseStage()
        if stage_type == "ir-validation":
            return IRValidationStage()
        if stage_type == "chunk":
            splitter = self._pipeline_resource.splitter
            return HaystackChunker(
                splitter, chunker_name="haystack.semantic", granularity="paragraph"
            )
        if stage_type == "embed":
            embedder = self._pipeline_resource.embedder
            return HaystackEmbedder(embedder=embedder, require_gpu=False, sparse_expander=None)
        if stage_type == "index":
            return HaystackIndexWriter(
                dense_writer=self._pipeline_resource.dense_writer,
                sparse_writer=self._pipeline_resource.sparse_writer,
            )
        if stage_type == "extract":
            return NoOpExtractStage()
        if stage_type == "knowledge-graph":
            return NoOpKnowledgeGraphStage()
        raise ValueError(f"Unsupported stage type '{stage_type}' for core plugin")


class PdfTwoPhasePlugin(StagePlugin):
    """Plugin providing download and gate stages for the pdf-two-phase pipeline."""

    def __init__(self) -> None:
        super().__init__(
            StagePluginMetadata(
                name="pdf-two-phase",
                version="1.0.0",
                stage_types=("download", "gate"),
                description="PDF download + gate pipeline stages",
            )
        )
        self._ledger: JobLedger | None = None

    def initialise(self, context: StagePluginContext) -> None:
        self._ledger = context.require("job_ledger")

    def health_check(self, context: StagePluginContext) -> None:
        if not isinstance(self._ledger, JobLedger):
            raise RuntimeError("Job ledger unavailable for PDF plugin")

    def create_stage(self, definition: StageDefinition, context: StagePluginContext) -> object:
        assert self._ledger is not None
        if definition.stage_type == "download":
            return _PdfDownloadStage(self._ledger)
        if definition.stage_type == "gate":
            return _PdfGateStage(self._ledger, gate_name=definition.name)
        raise ValueError(f"Unsupported stage type '{definition.stage_type}' for PDF plugin")


class _PdfDownloadStage:
    """Download stage that records PDF acquisition in the ledger and state."""

    def __init__(self, ledger: JobLedger) -> None:
        self._ledger = ledger

    def execute(self, ctx: StageContext, state: PipelineState) -> list[DownloadArtifact]:
        payloads = list(state.require_payloads())
        if not payloads:
            raise ValueError("PDF download stage requires payloads with source metadata")

        document_id = ctx.doc_id or state.context.doc_id or f"doc-{ctx.correlation_id or 'pdf'}"
        tenant = ctx.tenant_id
        artifacts: list[DownloadArtifact] = []
        for index, payload in enumerate(payloads):
            uri = str(
                payload.get("pdf_url") or payload.get("download_url") or payload.get("uri") or ""
            )
            if not uri:
                raise ValueError("PDF payload missing 'pdf_url' or 'download_url'")
            artifacts.append(
                DownloadArtifact(
                    document_id=document_id,
                    tenant_id=tenant,
                    uri=uri,
                    metadata={"payload_index": index, "source": payload.get("source")},
                )
            )

        job_id = ctx.job_id or state.job_id
        if job_id:
            self._ledger.set_pdf_downloaded(job_id, True)
        logger.info(
            "dagster.stage.pdf_download.completed",
            job_id=job_id,
            artifacts=len(artifacts),
        )
        return artifacts


class _PdfGateStage:
    """Gate stage that validates MinerU readiness using the job ledger."""

    def __init__(self, ledger: JobLedger, *, gate_name: str) -> None:
        self._ledger = ledger
        self._gate_name = gate_name

    def execute(self, ctx: StageContext, state: PipelineState) -> GateDecision:
        job_id = ctx.job_id or state.job_id
        if not job_id:
            raise ValueError("PDF gate requires a job identifier for ledger lookup")
        entry = self._ledger.get(job_id)
        ready = bool(entry and entry.pdf_ir_ready)
        decision = GateDecision(name=self._gate_name, ready=ready)
        if not ready:
            logger.info(
                "dagster.stage.pdf_gate.blocked",
                job_id=job_id,
                gate=self._gate_name,
            )
            raise PipelineGateNotReady(
                f"MinerU IR not ready for job {job_id}", gate=self._gate_name
            )
        logger.info(
            "dagster.stage.pdf_gate.ready",
            job_id=job_id,
            gate=self._gate_name,
        )
        return decision


__all__ = [
    "CoreStagePlugin",
    "PdfTwoPhasePlugin",
]
