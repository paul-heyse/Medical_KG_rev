"""Built-in stage plugin implementations for Dagster orchestration.

This module provides the core stage plugin implementations that are available
by default in the Medical_KG_rev pipeline orchestration system. It includes
both general-purpose stages (ingest, parse, chunk, embed, index) and specialized
stages for PDF processing workflows.

The module defines two main plugin classes:
- CoreStagePlugin: Provides the standard ingest→KG pipeline stages
- PdfTwoPhasePlugin: Provides PDF-specific download and gate stages

Each plugin implements the StagePlugin interface and can be dynamically
loaded and configured through the orchestration system.

Architecture:
- Plugins are initialized with required dependencies (adapter manager, pipeline resources)
- Stage creation is delegated to plugin-specific factory methods
- Health checks ensure dependencies are properly initialized
- Stage execution follows the standard StageContext → PipelineState → Artifacts pattern

Thread Safety:
- Plugin instances are thread-safe once initialized
- Stage instances created by plugins should be stateless or thread-safe
- Shared resources (adapter manager, pipeline resources) are managed externally

Performance:
- Plugin initialization is lightweight and cached
- Stage creation involves minimal overhead
- Health checks are designed to be fast and non-blocking

Examples:
    # Core stage plugin usage
    plugin = CoreStagePlugin()
    context = StagePluginContext({"adapter_manager": manager, "haystack_pipeline": resource})
    plugin.initialise(context)

    # Create an ingest stage
    definition = StageDefinition(name="ingest", stage_type="ingest", config={"adapter": "pdf"})
    stage = plugin.create_stage(definition, context)

"""

# IMPORTS
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import structlog
from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.adapters.plugins.models import AdapterDomain
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from Medical_KG_rev.orchestration.dagster.stages import HaystackPipelineResource
from Medical_KG_rev.orchestration.haystack.components import (
    HaystackChunker,
    HaystackEmbedder,
    HaystackIndexWriter,
)

try:
    from haystack.components.preprocessors import DocumentSplitter
except ImportError:
    DocumentSplitter = Any
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.contracts import (
    DownloadArtifact,
    GateDecision,
    PipelineGateNotReady,
    PipelineState,
    StageContext,
)
from Medical_KG_rev.orchestration.stages.docling_vlm_stage import DoclingVLMProcessingStage
from Medical_KG_rev.orchestration.stages.pdf_download import StorageAwarePdfDownloadStage
from Medical_KG_rev.orchestration.stages.pdf_gate import SimplePdfGateStage
from Medical_KG_rev.orchestration.stages.plugin_manager import (
    StagePlugin,
    StagePluginContext,
    StagePluginMetadata,
)
from Medical_KG_rev.services.parsing.docling import DoclingVLMOutputParser
from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMService

# TYPE DEFINITIONS & CONSTANTS
logger = structlog.get_logger(__name__)


# PLUGIN IMPLEMENTATIONS
class CoreStagePlugin(StagePlugin):
    """Core stage plugin providing default ingest→KG pipeline implementations.

    This plugin provides the standard stage implementations for the Medical_KG_rev
    pipeline, including ingest, parse, validation, chunking, embedding, indexing,
    and knowledge graph stages. It serves as the default plugin for most
    pipeline configurations.

    Attributes:
        _adapter_manager: Plugin manager for adapter instances
        _pipeline_resource: Haystack pipeline resource for ML components

    Thread Safety:
        Thread-safe once initialized. Stage instances created by this plugin
        should be stateless or thread-safe.

    Lifecycle:
        1. Initialize with metadata
        2. Call initialise() with required dependencies
        3. Use health_check() to verify dependencies
        4. Create stages via create_stage()

    Examples:
        plugin = CoreStagePlugin()
        context = StagePluginContext({
            "adapter_manager": adapter_manager,
            "haystack_pipeline": pipeline_resource
        })
        plugin.initialise(context)
        stage = plugin.create_stage(definition, context)

    """

    def __init__(self) -> None:
        """Initialize the core stage plugin with metadata.

        Sets up the plugin metadata including supported stage types and
        description. Dependencies are initialized later via initialise().

        Raises:
            None: Initialization always succeeds.

        """
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
                    "pdf-download",
                    "pdf-gate",
                    "pdf-vlm-process",
                ),
                description="Built-in stage implementations",
            )
        )
        self._adapter_manager: AdapterPluginManager | None = None
        self._pipeline_resource: HaystackPipelineResource | None = None
        self._docling_service: DoclingVLMService | None = None
        self._docling_parser = DoclingVLMOutputParser()

    def initialise(self, context: StagePluginContext) -> None:
        """Initialize the plugin with required dependencies.

        Args:
            context: Plugin context containing required dependencies

        Raises:
            KeyError: If required dependencies are missing from context
            TypeError: If dependencies have incorrect types

        """
        self._adapter_manager = context.require("adapter_manager")
        self._pipeline_resource = context.require("haystack_pipeline")

    def health_check(self, context: StagePluginContext) -> None:
        """Verify that all required dependencies are properly initialized.

        Args:
            context: Plugin context (unused but required by interface)

        Raises:
            RuntimeError: If any required dependency is not available

        """
        if not isinstance(self._adapter_manager, AdapterPluginManager):
            raise RuntimeError("Adapter manager not available for core stage plugin")
        from Medical_KG_rev.orchestration.dagster.stages import HaystackPipelineResource

        if not isinstance(self._pipeline_resource, HaystackPipelineResource):
            raise RuntimeError("Haystack pipeline resource not available")

    def create_stage(self, definition: StageDefinition, context: StagePluginContext) -> object:
        """Create a stage instance based on the stage definition.

        Args:
            definition: Stage definition containing type and configuration
            context: Plugin context (unused but required by interface)

        Returns:
            Stage instance appropriate for the given definition

        Raises:
            ValueError: If stage type is unsupported or configuration is invalid
            RuntimeError: If required dependencies are not initialized

        """
        assert self._adapter_manager is not None
        assert self._pipeline_resource is not None

        stage_type = definition.stage_type
        config: Mapping[str, Any] = definition.config
        if stage_type == "ingest":
            from Medical_KG_rev.orchestration.dagster.stages import AdapterIngestStage

            adapter = config.get("adapter")
            if not adapter:
                raise ValueError(f"Stage '{definition.name}' requires an adapter")
            strict = bool(config.get("strict", False))
            domain_value = config.get("domain")
            domain = (
                AdapterDomain(domain_value)
                if domain_value is not None
                else AdapterDomain.BIOMEDICAL
            )
            parameters = config.get("parameters", {}) if isinstance(config, Mapping) else {}
            return AdapterIngestStage(
                self._adapter_manager,
                adapter_name=str(adapter),
                strict=strict,
                default_domain=domain,
                extra_parameters=parameters if isinstance(parameters, Mapping) else {},
            )
        if stage_type == "parse":
            from Medical_KG_rev.orchestration.dagster.stages import AdapterParseStage

            return AdapterParseStage()
        if stage_type == "ir-validation":
            from Medical_KG_rev.orchestration.dagster.stages import IRValidationStage

            return IRValidationStage()
        if stage_type == "chunk":
            splitter = self._pipeline_resource.splitter
            # SimpleDocumentSplitter implements the required run() method for DocumentSplitter interface
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
            from Medical_KG_rev.orchestration.dagster.stages import NoOpExtractStage

            return NoOpExtractStage()
        if stage_type == "knowledge-graph":
            from Medical_KG_rev.orchestration.dagster.stages import NoOpKnowledgeGraphStage

            return NoOpKnowledgeGraphStage()
        if stage_type == "pdf-download":
            return StorageAwarePdfDownloadStage()
        if stage_type == "pdf-gate":
            return SimplePdfGateStage()
        if stage_type == "pdf-vlm-process":
            if self._docling_service is None:
                self._docling_service = DoclingVLMService()
            storage_client = context.get("pdf_storage")
            return DoclingVLMProcessingStage(
                service=self._docling_service,
                parser=self._docling_parser,
                storage_client=storage_client,
            )
        raise ValueError(f"Unsupported stage type '{stage_type}' for core plugin")


class PdfTwoPhasePlugin(StagePlugin):
    """Plugin providing download and gate stages for the pdf-two-phase pipeline.

    This plugin specializes in PDF processing workflows, providing stages for
    downloading PDFs and gating pipeline execution based on backend readiness.
    It's designed for two-phase PDF processing where the first phase handles
    PDF acquisition and the second phase waits for MinerU or Docling processing
    completion depending on the active feature flag.

    Attributes:
        _ledger: Job ledger for tracking PDF processing state

    Thread Safety:
        Thread-safe once initialized. Stage instances created by this plugin
        should be stateless or thread-safe.

    Lifecycle:
        1. Initialize with metadata
        2. Call initialise() with job ledger dependency
        3. Use health_check() to verify ledger availability
        4. Create stages via create_stage()

    Examples:
        plugin = PdfTwoPhasePlugin()
        context = StagePluginContext({"job_ledger": ledger})
        plugin.initialise(context)
        stage = plugin.create_stage(definition, context)

    """

    def __init__(self) -> None:
        """Initialize the PDF two-phase plugin with metadata.

        Sets up the plugin metadata including supported stage types and
        description. Dependencies are initialized later via initialise().

        Raises:
            None: Initialization always succeeds.

        """
        super().__init__(
            StagePluginMetadata(
                name="pdf-two-phase",
                version="1.0.0",
                stage_types=("download", "gate"),
                description="PDF download + gate pipeline stages",
            )
        )
        self._ledger: JobLedger | None = None
        self._backend = get_settings().feature_flags.pdf_processing_backend

    def initialise(self, context: StagePluginContext) -> None:
        """Initialize the plugin with required dependencies.

        Args:
            context: Plugin context containing required dependencies

        Raises:
            KeyError: If required dependencies are missing from context
            TypeError: If dependencies have incorrect types

        """
        self._ledger = context.require("job_ledger")

    def health_check(self, context: StagePluginContext) -> None:
        """Verify that all required dependencies are properly initialized.

        Args:
            context: Plugin context (unused but required by interface)

        Raises:
            RuntimeError: If any required dependency is not available

        """
        if not isinstance(self._ledger, JobLedger):
            raise RuntimeError("Job ledger unavailable for PDF plugin")

    def create_stage(self, definition: StageDefinition, context: StagePluginContext) -> object:
        """Create a stage instance based on the stage definition.

        Args:
            definition: Stage definition containing type and configuration
            context: Plugin context (unused but required by interface)

        Returns:
            Stage instance appropriate for the given definition

        Raises:
            ValueError: If stage type is unsupported or configuration is invalid
            RuntimeError: If required dependencies are not initialized

        """
        assert self._ledger is not None
        if definition.stage_type == "download":
            return _PdfDownloadStage(self._ledger, backend=self._backend)
        if definition.stage_type == "gate":
            return _PdfGateStage(self._ledger, gate_name=definition.name, backend=self._backend)
        raise ValueError(f"Unsupported stage type '{definition.stage_type}' for PDF plugin")


# PRIVATE STAGE IMPLEMENTATIONS
class _PdfDownloadStage:
    """Download stage that records PDF acquisition in the ledger and state.

    This private stage implementation handles PDF download operations for the
    two-phase PDF processing pipeline. It extracts PDF URLs from payloads,
    creates download artifacts, and records the download completion in the
    job ledger.

    Attributes:
        _ledger: Job ledger for tracking download state

    Thread Safety:
        Thread-safe. The ledger is assumed to be thread-safe.

    Examples:
        stage = _PdfDownloadStage(ledger)
        artifacts = stage.execute(context, state)

    """

    def __init__(self, ledger: JobLedger, *, backend: str) -> None:
        """Initialize the PDF download stage.

        Args:
            ledger: Job ledger for tracking download state
            backend: Active PDF processing backend identifier

        Raises:
            None: Initialization always succeeds.

        """
        self._ledger = ledger
        self._backend = backend

    def execute(self, ctx: StageContext, state: PipelineState) -> list[DownloadArtifact]:
        """Execute the PDF download stage.

        Extracts PDF URLs from payloads, creates download artifacts, and records
        the download completion in the job ledger.

        Args:
            ctx: Stage execution context
            state: Current pipeline state

        Returns:
            List of download artifacts created from payloads

        Raises:
            ValueError: If no payloads are available or payloads lack required URLs

        """
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
            if self._backend == "docling_vlm":
                self._ledger.set_pdf_vlm_ready(job_id, True)
        logger.info(
            "dagster.stage.pdf_download.completed",
            job_id=job_id,
            artifacts=len(artifacts),
        )
        return artifacts


class _PdfGateStage:
    """Gate stage that validates downstream processing readiness using the ledger.

    This private stage implementation handles pipeline gating for PDF processing.
    It checks the job ledger to determine if the configured backend (MinerU or
    Docling VLM) has completed processing and either allows pipeline continuation
    or raises a gate not ready exception.

    Attributes:
        _ledger: Job ledger for checking processing state
        _gate_name: Name of this gate for logging and error reporting

    Thread Safety:
        Thread-safe. The ledger is assumed to be thread-safe.

    Examples:
        stage = _PdfGateStage(ledger, gate_name="mineru-ready")
        decision = stage.execute(context, state)

    """

    def __init__(self, ledger: JobLedger, *, gate_name: str, backend: str) -> None:
        """Initialize the PDF gate stage.

        Args:
            ledger: Job ledger for checking processing state
            gate_name: Name of this gate for logging and error reporting
            backend: Active PDF processing backend identifier

        Raises:
            None: Initialization always succeeds.

        """
        self._ledger = ledger
        self._gate_name = gate_name
        self._backend = backend

    def execute(self, ctx: StageContext, state: PipelineState) -> GateDecision:
        """Execute the PDF gate stage.

        Checks the job ledger to determine if MinerU processing is complete.
        If ready, returns a gate decision allowing continuation. If not ready,
        raises a PipelineGateNotReady exception to block pipeline execution.

        Args:
            ctx: Stage execution context
            state: Current pipeline state

        Returns:
            Gate decision indicating readiness status

        Raises:
            ValueError: If no job identifier is available for ledger lookup
            PipelineGateNotReady: If MinerU processing is not yet complete

        """
        job_id = ctx.job_id or state.job_id
        if not job_id:
            raise ValueError("PDF gate requires a job identifier for ledger lookup")
        entry = self._ledger.get(job_id)
        if self._backend == "docling_vlm":
            ready = bool(entry and entry.pdf_vlm_ready)
        else:
            ready = bool(entry and entry.pdf_ir_ready)
        decision = GateDecision(name=self._gate_name, ready=ready)
        if not ready:
            logger.info(
                "dagster.stage.pdf_gate.blocked",
                job_id=job_id,
                gate=self._gate_name,
            )
            raise PipelineGateNotReady(
                (
                    f"Docling VLM not ready for job {job_id}"
                    if self._backend == "docling_vlm"
                    else f"MinerU IR not ready for job {job_id}"
                ),
                gate=self._gate_name,
            )
        logger.info(
            "dagster.stage.pdf_gate.ready",
            job_id=job_id,
            gate=self._gate_name,
        )
        return decision


# EXPORTS
__all__ = [
    "CoreStagePlugin",
    "PdfTwoPhasePlugin",
]
