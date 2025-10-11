"""Default stage implementations and builder helpers for Dagster pipelines."""

from __future__ import annotations

# Import additional plugin framework classes from plugins.py module
# (These are in plugins.py, not plugin_manager.py)
# Import directly from the plugins.py file to avoid circular imports
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Import additional plugin framework classes from plugins.py module
# (These are in plugins.py, not plugin_manager.py)
# Define minimal versions locally to avoid circular dependencies
from typing import Any, Callable
from uuid import uuid4

from attrs import define

import structlog
from Medical_KG_rev.adapters import AdapterPluginError
from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.adapters.plugins.models import AdapterDomain
from Medical_KG_rev.models.entities import Claim, Entity
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.haystack.components import (
    HaystackChunker,
    HaystackEmbedder,
    HaystackIndexWriter,
)


@define(slots=True)
class StagePluginResources:
    """Shared resources handed to plugin builders during registration."""
    adapter_manager: Any
    pipeline_resource: Any
    job_ledger: Any | None = None
    object_store: Any | None = None
    cache_backend: Any | None = None
    pdf_storage: Any | None = None
    document_storage: Any | None = None
    object_storage_settings: Any | None = None
    redis_cache_settings: Any | None = None

@define(slots=True)
class StagePluginRegistration:
    """Registration record returned by plugin implementations."""
    metadata: Any
    builder: Callable[["StageDefinition", StagePluginResources], object]
    provider: Any | None = None

# Import plugin framework from the plugins.py module (not the plugins/ package)
# Python resolves "plugins" as the package directory, so we import the module differently
from Medical_KG_rev.orchestration.stages.contracts import (
    ChunkStage,
    DownloadStage,
    EmbedStage,
    ExtractStage,
    GateStage,
    GraphWriteReceipt,
    IndexStage,
    IngestStage,
    KGStage,
    ParseStage,
    PdfAsset,
    PipelineState,
    RawPayload,
    StageContext,
)

# Import plugin framework from the plugin_manager module (the canonical location)
from Medical_KG_rev.orchestration.stages.plugin_manager import (
    StagePlugin,
    StagePluginContext,
    StagePluginManager,
    StagePluginMetadata,
    hookimpl,
)

# StagePluginResources and StagePluginRegistration are now defined locally above

# Note: CoreStagePlugin and PdfTwoPhasePlugin are imported lazily where needed
# to avoid circular imports (plugins/builtin.py imports from dagster/configuration.py)

try:  # pragma: no cover - optional import for typing only
    from Medical_KG_rev.orchestration.ledger import JobLedger
except Exception:  # pragma: no cover - defensive guard
    JobLedger = Any  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


class AdapterIngestStage(IngestStage):
    """Fetch raw payloads from a configured adapter using the plugin manager."""

    def __init__(
        self,
        manager: AdapterPluginManager,
        *,
        adapter_name: str,
        strict: bool = False,
        default_domain: AdapterDomain = AdapterDomain.BIOMEDICAL,
        extra_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        self._manager = manager
        self._adapter = adapter_name
        self._strict = strict
        self._default_domain = default_domain
        self._extra_parameters = dict(extra_parameters or {})

    def execute(self, ctx: StageContext, state: PipelineState) -> list[RawPayload]:
        request = state.adapter_request
        merged_parameters = {**self._extra_parameters, **dict(request.parameters)}
        domain = request.domain or self._default_domain  # type: ignore[union-attr]
        invocation_request = request.model_copy(
            update={"parameters": merged_parameters, "domain": domain}
        )
        try:
            result = self._manager.invoke(self._adapter, invocation_request, strict=self._strict)
        except AdapterPluginError as exc:
            logger.error(
                "dagster.stage.ingest.error",
                adapter=self._adapter,
                tenant_id=request.tenant_id,
                error=str(exc),
            )
            raise
        payloads: list[RawPayload] = []
        if result.response is not None:
            for item in result.response.items:
                if isinstance(item, Mapping):
                    payloads.append(dict(item))
                else:
                    payloads.append({"value": item})
        if not payloads:
            payloads.append({"parameters": merged_parameters})
        logger.debug(
            "dagster.stage.ingest.completed",
            adapter=self._adapter,
            tenant_id=request.tenant_id,
            payloads=len(payloads),
        )
        return payloads


class AdapterParseStage(ParseStage):
    """Parse raw adapter payloads into the IR document format."""

    def __init__(self, *, default_source: str = "unknown") -> None:
        self._default_source = default_source

    def execute(self, ctx: StageContext, state: PipelineState) -> Document:
        payloads = list(state.require_payloads())
        doc_id = ctx.doc_id or f"doc-{ctx.correlation_id or uuid4().hex}"
        source = ctx.metadata.get("dataset") if isinstance(ctx.metadata, Mapping) else None
        source = str(source or self._default_source)
        blocks: list[Block] = []
        for index, payload in enumerate(payloads):
            text = self._extract_text(payload)
            block = Block(
                id=f"{doc_id}:block:{index}",
                type=BlockType.PARAGRAPH,
                text=text,
                metadata={"payload": payload},
            )
            blocks.append(block)
        section = Section(id=f"{doc_id}:section:0", title="Document", blocks=tuple(blocks))
        document = Document(
            id=doc_id,
            source=source,
            title=ctx.metadata.get("title") if isinstance(ctx.metadata, Mapping) else None,
            sections=(section,),
            metadata={"raw_payloads": payloads},
        )
        logger.debug(
            "dagster.stage.parse.completed",
            doc_id=doc_id,
            blocks=len(blocks),
        )
        return document

    def _extract_text(self, payload: RawPayload) -> str:
        if isinstance(payload, Mapping):
            for key in ("text", "abstract", "content", "body"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return json.dumps(payload, sort_keys=True)
        return str(payload)


class IRValidationStage(ParseStage):
    """Validate that the parsed document contains content for downstream stages."""

    def execute(self, ctx: StageContext, state: PipelineState) -> Document:
        document = state.require_document()
        if not document.sections or not any(section.blocks for section in document.sections):
            raise ValueError("Document contains no content for downstream processing")
        return document


class NoOpExtractStage(ExtractStage):
    """Stub extraction stage returning empty entity and claim collections."""

    def execute(self, ctx: StageContext, state: PipelineState) -> tuple[list[Entity], list[Claim]]:
        state.require_document()
        return ([], [])


class NoOpKnowledgeGraphStage(KGStage):
    """Return an empty write receipt for pipelines without KG integration."""

    def execute(self, ctx: StageContext, state: PipelineState) -> GraphWriteReceipt:
        correlation_id = ctx.correlation_id or uuid4().hex
        entities = list(state.entities)
        claims = list(state.claims)
        return GraphWriteReceipt(
            nodes_written=len(entities),
            edges_written=len(claims),
            correlation_id=correlation_id,
            metadata={"pipeline": ctx.pipeline_name, "version": ctx.pipeline_version},
        )


@dataclass
class SimpleSplitDocument:
    content: str
    meta: dict[str, Any]
    embedding: Sequence[float] | None = None
    sparse_embedding: Mapping[str, float] | None = None
    id: str | None = None


class SimpleDocumentSplitter:
    """Lightweight splitter used when Haystack components are unavailable."""

    def __init__(self, *, sentence_length: int = 3) -> None:
        self._sentence_length = max(1, sentence_length)

    def run(self, *, documents: Sequence[Any]) -> dict[str, list[SimpleSplitDocument]]:
        results: list[SimpleSplitDocument] = []
        for doc in documents:
            content = getattr(doc, "content", "") or ""
            meta = dict(getattr(doc, "meta", {}) or {})
            if not content.strip():
                continue
            sentences = [segment.strip() for segment in content.split(".") if segment.strip()]
            if not sentences:
                sentences = [content.strip()]
            buffer: list[str] = []
            for sentence in sentences:
                buffer.append(sentence)
                if len(buffer) >= self._sentence_length:
                    chunk = ". ".join(buffer)
                    results.append(SimpleSplitDocument(content=chunk, meta=dict(meta)))
                    buffer = []
            if buffer:
                chunk = ". ".join(buffer)
                results.append(SimpleSplitDocument(content=chunk, meta=dict(meta)))
        return {"documents": results}


class StubEmbeddingDocument(SimpleSplitDocument):
    """Small extension providing a stable embedding vector for tests."""

    def __init__(self, *, content: str, meta: Mapping[str, Any]) -> None:
        vector = [float((len(content) + index) % 7) for index in range(4)]
        super().__init__(content=content, meta=dict(meta), embedding=tuple(vector))


class SimpleEmbedder:
    """Deterministic embedder producing small dense vectors."""

    def run(self, *, documents: Sequence[Any]) -> dict[str, list[StubEmbeddingDocument]]:
        embedded = [
            StubEmbeddingDocument(
                content=getattr(doc, "content", ""), meta=getattr(doc, "meta", {})
            )
            for doc in documents
        ]
        return {"documents": embedded}


class NoOpDocumentWriter:
    """Writer stub satisfying the Haystack writer interface."""

    def __init__(self, *, name: str) -> None:
        self._name = name

    def run(self, *, documents: Sequence[Any]) -> dict[str, Any]:  # pragma: no cover - trivial
        logger.debug("dagster.index.writer.noop", writer=self._name, documents=len(documents))
        return {"documents": list(documents)}


@dataclass(slots=True)
class HaystackPipelineResource:
    splitter: SimpleDocumentSplitter
    embedder: SimpleEmbedder
    dense_writer: NoOpDocumentWriter
    sparse_writer: NoOpDocumentWriter


def create_default_pipeline_resource() -> HaystackPipelineResource:
    return HaystackPipelineResource(
        splitter=SimpleDocumentSplitter(),
        embedder=SimpleEmbedder(),
        dense_writer=NoOpDocumentWriter(name="faiss"),
        sparse_writer=NoOpDocumentWriter(name="opensearch"),
    )


class PdfDownloadStage(DownloadStage):
    """Stage responsible for retrieving PDF assets for a pipeline run."""

    def __init__(
        self,
        *,
        job_ledger: JobLedger | None = None,
        urls: Sequence[str] | None = None,
        checksum_field: str | None = None,
    ) -> None:
        self._job_ledger = job_ledger
        self._urls = tuple(urls or ())
        self._checksum_field = checksum_field

    def execute(self, ctx: StageContext, state: PipelineState) -> list[PdfAsset]:
        configured_urls: Sequence[str] = self._urls
        if not configured_urls and isinstance(state.metadata, Mapping):
            metadata_urls = state.metadata.get("pdf_urls")
            if isinstance(metadata_urls, Sequence):
                configured_urls = tuple(str(url) for url in metadata_urls)
        checksum_value: str | None = None
        if isinstance(ctx.metadata, Mapping) and self._checksum_field:
            raw_value = ctx.metadata.get(self._checksum_field)
            if raw_value is not None:
                checksum_value = str(raw_value)
        assets: list[PdfAsset] = []
        for index, url in enumerate(configured_urls or ()):
            assets.append(
                PdfAsset(
                    asset_id=f"{ctx.doc_id or ctx.job_id or 'pdf'}:{index}",
                    uri=str(url),
                    checksum=checksum_value,
                    metadata={"source": "configured"},
                )
            )
        if not assets:
            assets.append(
                PdfAsset(
                    asset_id=f"{ctx.doc_id or ctx.job_id or 'pdf'}:0",
                    uri="about:blank",
                    checksum=checksum_value,
                    metadata={"generated": True},
                )
            )
        if ctx.job_id and isinstance(self._job_ledger, JobLedger):
            try:
                self._job_ledger.set_pdf_downloaded(ctx.job_id, True)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "dagster.stage.pdf_download.ledger_error",
                    job_id=ctx.job_id,
                    error=str(exc),
                )
        return assets


class PdfGateStage(GateStage):
    """Stage that evaluates ledger state to determine PDF readiness."""

    def __init__(
        self,
        *,
        job_ledger: JobLedger | None = None,
        field: str = "pdf_ir_ready",
    ) -> None:
        self._job_ledger = job_ledger
        self._field = field

    def execute(self, ctx: StageContext, state: PipelineState) -> bool:
        if not ctx.job_id or not isinstance(self._job_ledger, JobLedger):
            return False
        try:
            entry = self._job_ledger.get(ctx.job_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "dagster.stage.pdf_gate.ledger_error",
                job_id=ctx.job_id,
                error=str(exc),
            )
            return False
        if entry is None:
            return False
        return bool(getattr(entry, self._field, False))


class CoreStagePlugin(StagePlugin):
    """Register the built-in orchestration stage implementations."""

    NAME = "core-stage"
    VERSION = "1.0.0"

    def __init__(self) -> None:
        self.metadata = StagePluginMetadata(
            name=self.NAME,
            version=self.VERSION,
            stage_types=("ingest", "extract", "chunk", "embed", "index", "query"),
            description="Core orchestration stage implementations"
        )

    def registrations(self, resources: StagePluginResources) -> Sequence[StagePluginRegistration]:
        """Return empty list - this plugin uses stage_builders instead."""
        return []

    @hookimpl
    def stage_builders(self, resources: StagePluginResources) -> Sequence[StagePluginRegistration]:
        adapter_manager = resources.adapter_manager
        pipeline_resource = resources.pipeline_resource
        job_ledger = resources.job_ledger if isinstance(resources.job_ledger, JobLedger) else None

        def build_ingest(definition: StageDefinition, _: StagePluginResources) -> IngestStage:
            config = definition.config
            adapter_name = config.get("adapter")
            if not adapter_name:
                raise ValueError(f"Stage '{definition.name}' requires an adapter name")
            strict = bool(config.get("strict", False))
            domain_value = config.get("domain")
            try:
                domain = AdapterDomain(domain_value) if domain_value else AdapterDomain.BIOMEDICAL
            except Exception as exc:  # pragma: no cover - validation guard
                raise ValueError(f"Invalid adapter domain '{domain_value}'") from exc
            extra_parameters = config.get("parameters", {}) if isinstance(config, Mapping) else {}
            return AdapterIngestStage(
                adapter_manager,
                adapter_name=adapter_name,
                strict=strict,
                default_domain=domain,
                extra_parameters=extra_parameters if isinstance(extra_parameters, Mapping) else {},
            )

        def build_parse(_: StageDefinition, __: StagePluginResources) -> ParseStage:
            return AdapterParseStage()

        def build_validation(_: StageDefinition, __: StagePluginResources) -> ParseStage:
            return IRValidationStage()

        def build_chunk(_: StageDefinition, __: StagePluginResources) -> ChunkStage:
            splitter = pipeline_resource.splitter
            return HaystackChunker(
                splitter, chunker_name="haystack.semantic", granularity="paragraph"
            )

        def build_embed(_: StageDefinition, __: StagePluginResources) -> EmbedStage:
            embedder = pipeline_resource.embedder
            return HaystackEmbedder(embedder=embedder, require_gpu=False, sparse_expander=None)

        def build_index(_: StageDefinition, __: StagePluginResources) -> IndexStage:
            dense_writer = pipeline_resource.dense_writer
            sparse_writer = pipeline_resource.sparse_writer
            return HaystackIndexWriter(dense_writer=dense_writer, sparse_writer=sparse_writer)

        def build_extract(_: StageDefinition, __: StagePluginResources) -> ExtractStage:
            return NoOpExtractStage()

        def build_kg(_: StageDefinition, __: StagePluginResources) -> KGStage:
            return NoOpKnowledgeGraphStage()

        def build_download(
            definition: StageDefinition, __: StagePluginResources
        ) -> PdfDownloadStage:
            config = definition.config
            urls = config.get("urls") if isinstance(config, Mapping) else None
            checksum_field = config.get("checksum_field") if isinstance(config, Mapping) else None
            urls_seq = tuple(str(url) for url in urls) if isinstance(urls, Sequence) else None
            return PdfDownloadStage(
                job_ledger=job_ledger,
                urls=urls_seq,
                checksum_field=str(checksum_field) if checksum_field else None,
            )

        def build_gate(definition: StageDefinition, __: StagePluginResources) -> PdfGateStage:
            config = definition.config
            field_name = (
                config.get("field", "pdf_ir_ready")
                if isinstance(config, Mapping)
                else "pdf_ir_ready"
            )
            return PdfGateStage(job_ledger=job_ledger, field=str(field_name))

        return (
            self.create_registration(
                stage_type="ingest",
                builder=build_ingest,
                capabilities=("adapter",),
            ),
            self.create_registration(
                stage_type="parse",
                builder=build_parse,
                capabilities=("document",),
            ),
            self.create_registration(
                stage_type="ir-validation",
                builder=build_validation,
                capabilities=("document",),
            ),
            self.create_registration(
                stage_type="chunk",
                builder=build_chunk,
                capabilities=("haystack",),
            ),
            self.create_registration(
                stage_type="embed",
                builder=build_embed,
                capabilities=("haystack",),
            ),
            self.create_registration(
                stage_type="index",
                builder=build_index,
                capabilities=("haystack",),
            ),
            self.create_registration(
                stage_type="extract",
                builder=build_extract,
                capabilities=("noop",),
            ),
            self.create_registration(
                stage_type="knowledge-graph",
                builder=build_kg,
                capabilities=("noop",),
            ),
            self.create_registration(
                stage_type="download",
                builder=build_download,
                capabilities=("pdf",),
            ),
            self.create_registration(
                stage_type="gate",
                builder=build_gate,
                capabilities=("ledger",),
            ),
        )
        return [
            StagePluginRegistration(
                metadata=StagePluginMetadata(
                    name=f"{self.NAME}.ingest",
                    version=self.VERSION,
                    stage_type="ingest",
                    capabilities=("adapter",),
                ),
                builder=build_ingest,
            ),
            StagePluginRegistration(
                metadata=StagePluginMetadata(
                    name=f"{self.NAME}.parse",
                    version=self.VERSION,
                    stage_type="parse",
                    capabilities=("document",),
                ),
                builder=build_parse,
            ),
            StagePluginRegistration(
                metadata=StagePluginMetadata(
                    name=f"{self.NAME}.ir-validation",
                    version=self.VERSION,
                    stage_type="ir-validation",
                    capabilities=("document",),
                ),
                builder=build_validation,
            ),
            StagePluginRegistration(
                metadata=StagePluginMetadata(
                    name=f"{self.NAME}.chunk",
                    version=self.VERSION,
                    stage_type="chunk",
                    capabilities=("haystack",),
                ),
                builder=build_chunk,
            ),
            StagePluginRegistration(
                metadata=StagePluginMetadata(
                    name=f"{self.NAME}.embed",
                    version=self.VERSION,
                    stage_type="embed",
                    capabilities=("haystack",),
                ),
                builder=build_embed,
            ),
            StagePluginRegistration(
                metadata=StagePluginMetadata(
                    name=f"{self.NAME}.index",
                    version=self.VERSION,
                    stage_type="index",
                    capabilities=("haystack",),
                ),
                builder=build_index,
            ),
            StagePluginRegistration(
                metadata=StagePluginMetadata(
                    name=f"{self.NAME}.extract",
                    version=self.VERSION,
                    stage_type="extract",
                    capabilities=("noop",),
                ),
                builder=build_extract,
            ),
            StagePluginRegistration(
                metadata=StagePluginMetadata(
                    name=f"{self.NAME}.knowledge-graph",
                    version=self.VERSION,
                    stage_type="knowledge-graph",
                    capabilities=("noop",),
                ),
                builder=build_kg,
            ),
            StagePluginRegistration(
                metadata=StagePluginMetadata(
                    name=f"{self.NAME}.download",
                    version=self.VERSION,
                    stage_type="download",
                    capabilities=("pdf",),
                ),
                builder=build_download,
            ),
            StagePluginRegistration(
                metadata=StagePluginMetadata(
                    name=f"{self.NAME}.gate",
                    version=self.VERSION,
                    stage_type="gate",
                    capabilities=("ledger",),
                ),
                builder=build_gate,
            ),
        ]


def create_stage_plugin_manager(
    adapter_manager: AdapterPluginManager,
    pipeline_resource: HaystackPipelineResource | None = None,
    *,
    job_ledger: JobLedger | None = None,
    object_store=None,
    cache_backend=None,
    pdf_storage=None,
    document_storage=None,
    object_storage_settings=None,
    redis_cache_settings=None,
    load_entrypoints: bool = False,
) -> StagePluginManager:
    resources = StagePluginResources(
        adapter_manager=adapter_manager,
        pipeline_resource=pipeline_resource or create_default_pipeline_resource(),
        job_ledger=job_ledger,
        object_store=object_store,
        cache_backend=cache_backend,
        pdf_storage=pdf_storage,
        document_storage=document_storage,
        object_storage_settings=object_storage_settings,
        redis_cache_settings=redis_cache_settings,
    )
    # Convert StagePluginResources to dict manually since it uses slots=True
    resources_dict = {
        "adapter_manager": resources.adapter_manager,
        "pipeline_resource": resources.pipeline_resource,
        "job_ledger": resources.job_ledger,
        "object_store": resources.object_store,
        "cache_backend": resources.cache_backend,
        "pdf_storage": resources.pdf_storage,
        "document_storage": resources.document_storage,
        "object_storage_settings": resources.object_storage_settings,
        "redis_cache_settings": resources.redis_cache_settings,
    }
    context = StagePluginContext(resources=resources_dict)
    manager = StagePluginManager(context=context)
    manager.register(CoreStagePlugin())
    if load_entrypoints:
        manager.load_entrypoints()
    return manager


__all__ = [
    "AdapterIngestStage",
    "AdapterParseStage",
    "CoreStagePlugin",
    "HaystackPipelineResource",
    "IRValidationStage",
    "NoOpDocumentWriter",
    "NoOpExtractStage",
    "NoOpKnowledgeGraphStage",
    "PdfDownloadStage",
    "PdfGateStage",
    "SimpleDocumentSplitter",
    "SimpleEmbedder",
    "create_default_pipeline_resource",
    "create_stage_plugin_manager",
]
