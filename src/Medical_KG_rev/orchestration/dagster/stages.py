"""Default stage implementations and builder helpers for Dagster pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import structlog

from Medical_KG_rev.adapters import AdapterPluginError
from Medical_KG_rev.adapters.plugins.models import AdapterDomain, AdapterRequest
from Medical_KG_rev.models.entities import Claim, Entity
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
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
    PipelineState,
    StageContext,
)
from Medical_KG_rev.orchestration.stages.contracts import RawPayload

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

