"""Default stage implementations and builder helpers for Dagster pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Mapping
from uuid import uuid4

import structlog

from Medical_KG_rev.adapters import AdapterPluginError
from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterRequest
from Medical_KG_rev.models.entities import Claim, Entity
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.stage_registry import (
    StageMetadata,
    StageRegistry,
    StageRegistryError,
)
from Medical_KG_rev.orchestration.haystack.components import (
    HaystackChunker,
    HaystackEmbedder,
    HaystackIndexWriter,
)
from Medical_KG_rev.orchestration.stages.contracts import (
    ChunkStage,
    EmbedStage,
    ExtractStage,
    GraphWriteReceipt,
    IngestStage,
    IndexStage,
    KGStage,
    ParseStage,
    StageContext,
)
from Medical_KG_rev.orchestration.stages.contracts import RawPayload

logger = structlog.get_logger(__name__)


def _sequence_length(output: Any) -> int:
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        return len(output)
    return 0


def _count_single(output: Any) -> int:
    return 1 if output is not None else 0


def _count_embed(output: Any) -> int:
    vectors = getattr(output, "vectors", None)
    if isinstance(vectors, Sequence):
        return len(vectors)
    return 0


def _count_index(output: Any) -> int:
    indexed = getattr(output, "chunks_indexed", None)
    if isinstance(indexed, int):
        return max(indexed, 0)
    return 0


def _count_extract(output: Any) -> int:
    if not isinstance(output, tuple) or len(output) != 2:
        return 0
    entities, claims = output
    entity_count = _sequence_length(entities)
    claim_count = _sequence_length(claims)
    return entity_count + claim_count


def _count_graph(output: Any) -> int:
    nodes = getattr(output, "nodes_written", None)
    if isinstance(nodes, int):
        return max(nodes, 0)
    return 0


def _handle_ingest(state: dict[str, Any], _: str, output: Any) -> None:
    state["payloads"] = output


def _handle_document(state: dict[str, Any], _: str, output: Any) -> None:
    state["document"] = output


def _handle_chunks(state: dict[str, Any], _: str, output: Any) -> None:
    state["chunks"] = output


def _handle_embedding_batch(state: dict[str, Any], _: str, output: Any) -> None:
    state["embedding_batch"] = output


def _handle_index_receipt(state: dict[str, Any], _: str, output: Any) -> None:
    state["index_receipt"] = output


def _handle_extract(state: dict[str, Any], _: str, output: Any) -> None:
    entities: Any = []
    claims: Any = []
    if isinstance(output, tuple) and len(output) == 2:
        entities, claims = output
    state["entities"] = list(entities) if isinstance(entities, Sequence) else entities
    state["claims"] = list(claims) if isinstance(claims, Sequence) else claims


def _handle_graph_receipt(state: dict[str, Any], _: str, output: Any) -> None:
    state["graph_receipt"] = output


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

    def execute(self, ctx: StageContext, request: AdapterRequest) -> list[RawPayload]:
        merged_parameters = {**self._extra_parameters, **dict(request.parameters)}
        domain = request.domain or self._default_domain  # type: ignore[union-attr]
        invocation_request = request.model_copy(update={"parameters": merged_parameters, "domain": domain})
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

    def execute(self, ctx: StageContext, payloads: list[RawPayload]) -> Document:
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

    def execute(self, ctx: StageContext, document: Document) -> Document:
        if not isinstance(document, Document):
            raise TypeError("IRValidationStage expects a Document instance")
        if not document.sections or not any(section.blocks for section in document.sections):
            raise ValueError("Document contains no content for downstream processing")
        return document


class NoOpExtractStage(ExtractStage):
    """Stub extraction stage returning empty entity and claim collections."""

    def execute(self, ctx: StageContext, document: Document) -> tuple[list[Entity], list[Claim]]:
        return ([], [])


class NoOpKnowledgeGraphStage(KGStage):
    """Return an empty write receipt for pipelines without KG integration."""

    def execute(self, ctx: StageContext, entities: list[Entity], claims: list[Claim]) -> GraphWriteReceipt:
        correlation_id = ctx.correlation_id or uuid4().hex
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
            StubEmbeddingDocument(content=getattr(doc, "content", ""), meta=getattr(doc, "meta", {}))
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


def build_default_stage_factory(
    manager: AdapterPluginManager,
    pipeline: HaystackPipelineResource | None = None,
) -> StageRegistry:
    """Build the default stage registry with built-in metadata and builders."""

    pipeline = pipeline or create_default_pipeline_resource()
    splitter = pipeline.splitter
    embedder = pipeline.embedder
    dense_writer = pipeline.dense_writer
    sparse_writer = pipeline.sparse_writer

    def _ingest_builder(definition: StageDefinition) -> IngestStage:
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
            manager,
            adapter_name=adapter_name,
            strict=strict,
            default_domain=domain,
            extra_parameters=extra_parameters if isinstance(extra_parameters, Mapping) else {},
        )

    def _parse_builder(_: StageDefinition) -> ParseStage:
        return AdapterParseStage()

    def _validation_builder(_: StageDefinition) -> ParseStage:
        return IRValidationStage()

    def _chunk_builder(_: StageDefinition) -> ChunkStage:
        return HaystackChunker(splitter, chunker_name="haystack.semantic", granularity="paragraph")

    def _embed_builder(_: StageDefinition) -> EmbedStage:
        return HaystackEmbedder(embedder=embedder, require_gpu=False, sparse_expander=None)

    def _index_builder(_: StageDefinition) -> IndexStage:
        return HaystackIndexWriter(dense_writer=dense_writer, sparse_writer=sparse_writer)

    def _extract_builder(_: StageDefinition) -> ExtractStage:
        return NoOpExtractStage()

    def _kg_builder(_: StageDefinition) -> KGStage:
        return NoOpKnowledgeGraphStage()

    registry = StageRegistry()
    registry.register_stage(
        metadata=StageMetadata(
            stage_type="ingest",
            state_key="payloads",
            output_handler=_handle_ingest,
            output_counter=_sequence_length,
            description="Fetches raw payloads from an adapter",
        ),
        builder=_ingest_builder,
    )
    registry.register_stage(
        metadata=StageMetadata(
            stage_type="parse",
            state_key="document",
            output_handler=_handle_document,
            output_counter=_count_single,
            description="Parses raw payloads into an IR document",
            dependencies=("ingest",),
        ),
        builder=_parse_builder,
    )
    registry.register_stage(
        metadata=StageMetadata(
            stage_type="ir-validation",
            state_key="document",
            output_handler=_handle_document,
            output_counter=_count_single,
            description="Validates parsed documents before downstream stages",
            dependencies=("parse",),
        ),
        builder=_validation_builder,
    )
    registry.register_stage(
        metadata=StageMetadata(
            stage_type="chunk",
            state_key="chunks",
            output_handler=_handle_chunks,
            output_counter=_sequence_length,
            description="Splits documents into retrieval-ready chunks",
            dependencies=("parse", "ir-validation"),
        ),
        builder=_chunk_builder,
    )
    registry.register_stage(
        metadata=StageMetadata(
            stage_type="embed",
            state_key="embedding_batch",
            output_handler=_handle_embedding_batch,
            output_counter=_count_embed,
            description="Generates embeddings for document chunks",
            dependencies=("chunk",),
        ),
        builder=_embed_builder,
    )
    registry.register_stage(
        metadata=StageMetadata(
            stage_type="index",
            state_key="index_receipt",
            output_handler=_handle_index_receipt,
            output_counter=_count_index,
            description="Writes embeddings to downstream indexes",
            dependencies=("embed",),
        ),
        builder=_index_builder,
    )
    registry.register_stage(
        metadata=StageMetadata(
            stage_type="extract",
            state_key=("entities", "claims"),
            output_handler=_handle_extract,
            output_counter=_count_extract,
            description="Extracts biomedical entities and claims",
            dependencies=("parse",),
        ),
        builder=_extract_builder,
    )
    registry.register_stage(
        metadata=StageMetadata(
            stage_type="knowledge-graph",
            state_key="graph_receipt",
            output_handler=_handle_graph_receipt,
            output_counter=_count_graph,
            description="Persists extracted facts into the knowledge graph",
            dependencies=("extract",),
        ),
        builder=_kg_builder,
    )
    try:
        from Medical_KG_rev.orchestration import stage_plugins

        for plugin_factory in (stage_plugins.register_download_stage, stage_plugins.register_gate_stage):
            registration = None
            try:
                registration = plugin_factory()
                registry.register(registration)
            except StageRegistryError as exc:
                stage_type = registration.metadata.stage_type if registration else getattr(plugin_factory, "__name__", "unknown")
                logger.debug(
                    "dagster.stage.registry.plugin_skipped",
                    stage_type=stage_type,
                    error=str(exc),
                )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("dagster.stage.registry.plugin_init_failed", error=str(exc))
    registry.load_plugins()
    return registry
