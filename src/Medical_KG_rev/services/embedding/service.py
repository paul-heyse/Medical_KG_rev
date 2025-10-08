"""Stage-based embedding worker backed by Dagster components."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import structlog

from Medical_KG_rev.adapters import get_plugin_manager
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.runtime import (
    StageFactory,
    StageResolutionError,
)
from Medical_KG_rev.orchestration.dagster.stages import build_default_stage_factory
from Medical_KG_rev.orchestration.stages.contracts import (
    EmbedStage,
    EmbeddingBatch,
    StageContext,
)

from .registry import EmbeddingModelRegistry

logger = structlog.get_logger(__name__)


def _default_stage_factory() -> StageFactory:
    """Instantiate the default stage factory using registered adapters."""

    registry = build_default_stage_factory(get_plugin_manager())
    return StageFactory(registry)


@dataclass(slots=True)
class EmbeddingRequest:
    """Payload accepted by :class:`EmbeddingWorker`."""

    tenant_id: str
    texts: Sequence[str]
    chunk_ids: Sequence[str] | None = None
    normalize: bool = False
    model: str | None = None
    metadata: Sequence[Mapping[str, Any]] | None = None
    correlation_id: str | None = None
    pipeline_name: str | None = None
    pipeline_version: str | None = None


@dataclass(slots=True)
class EmbeddingVector:
    """Embedding vector returned to gateway and gRPC callers."""

    id: str
    model: str
    kind: str = "dense"
    values: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return len(self.values)


@dataclass(slots=True)
class EmbeddingResponse:
    """Container produced by :class:`EmbeddingWorker`."""

    vectors: list[EmbeddingVector] = field(default_factory=list)


class EmbeddingWorker:
    """Thin wrapper that executes the embed stage via Dagster components."""

    def __init__(
        self,
        registry: EmbeddingModelRegistry | None = None,  # noqa: ARG002 - retained for compatibility
        *,
        stage_factory: StageFactory | None = None,
        embed_stage: EmbedStage | None = None,
        stage_definition: StageDefinition | None = None,
        pipeline_name: str = "gateway-direct",
        pipeline_version: str = "v1",
    ) -> None:
        self._stage_factory = stage_factory or _default_stage_factory()
        self._embed_stage = embed_stage
        self._stage_definition = stage_definition or StageDefinition(name="embed", type="embed")
        self._pipeline_name = pipeline_name
        self._pipeline_version = pipeline_version

    # ------------------------------------------------------------------
    def run(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Execute the embed stage for the provided request."""

        cleaned_texts = [
            text.strip()
            for text in request.texts
            if isinstance(text, str) and text.strip()
        ]
        if not cleaned_texts:
            logger.warning(
                "embedding.worker.empty_payload",
                tenant_id=request.tenant_id,
            )
            return EmbeddingResponse()

        chunks = self._build_chunks(
            tenant_id=request.tenant_id,
            texts=cleaned_texts,
            chunk_ids=request.chunk_ids,
            metadata=request.metadata,
        )
        context = StageContext(
            tenant_id=request.tenant_id,
            doc_id=f"embed:{uuid.uuid4().hex[:12]}",
            correlation_id=request.correlation_id or uuid.uuid4().hex,
            metadata={
                "model": request.model,
                "normalize": request.normalize,
            },
            pipeline_name=request.pipeline_name or self._pipeline_name,
            pipeline_version=request.pipeline_version or self._pipeline_version,
        )

        stage = self._resolve_stage()
        try:
            batch = stage.execute(context, chunks)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception(
                "embedding.worker.stage_error",
                tenant_id=request.tenant_id,
                error=str(exc),
            )
            raise
        return self._response_from_batch(batch, normalize=request.normalize)

    # ------------------------------------------------------------------
    def encode_queries(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Alias for :meth:`run` retained for compatibility with rerankers."""

        return self.run(request)

    # ------------------------------------------------------------------
    def _resolve_stage(self) -> EmbedStage:
        if self._embed_stage is not None:
            return self._embed_stage
        try:
            stage = self._stage_factory.resolve(
                self._pipeline_name,
                self._stage_definition,
            )
        except StageResolutionError:
            # Retry with a generic definition that omits name, matching configuration defaults.
            stage = self._stage_factory.resolve(
                self._pipeline_name,
                StageDefinition(name="embed", type="embed"),
            )
        if not isinstance(stage, EmbedStage):  # pragma: no cover - defensive
            raise TypeError("Resolved stage does not implement EmbedStage")
        self._embed_stage = stage
        return stage

    def _build_chunks(
        self,
        *,
        tenant_id: str,
        texts: Sequence[str],
        chunk_ids: Sequence[str] | None,
        metadata: Sequence[Mapping[str, Any]] | None,
    ) -> list[Chunk]:
        ids = list(chunk_ids or [])
        if len(ids) < len(texts):
            ids.extend(f"{tenant_id}:{index}" for index in range(len(ids), len(texts)))
        else:
            ids = ids[: len(texts)]

        meta_sequence = [dict(item) for item in (metadata or [])][: len(texts)]
        while len(meta_sequence) < len(texts):
            meta_sequence.append({})

        chunks: list[Chunk] = []
        for index, (text, chunk_id, meta) in enumerate(zip(texts, ids, meta_sequence, strict=True)):
            doc_id = meta.get("doc_id") or f"{chunk_id}:doc"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=str(doc_id),
                    tenant_id=tenant_id,
                    body=text,
                    title_path=tuple(meta.get("title_path", ())),
                    section=meta.get("section"),
                    start_char=int(meta.get("start", 0)),
                    end_char=int(meta.get("end", len(text))),
                    granularity=str(meta.get("granularity") or "document"),
                    chunker=str(meta.get("chunker") or "embedding.worker"),
                    chunker_version=str(meta.get("chunker_version") or "1.0.0"),
                    meta={"input_index": index, **meta},
                )
            )
        return chunks

    def _response_from_batch(
        self,
        batch: EmbeddingBatch,
        *,
        normalize: bool,
    ) -> EmbeddingResponse:
        vectors: list[EmbeddingVector] = []
        for vector in batch.vectors:
            values = list(vector.values)
            if normalize and values:
                magnitude = math.sqrt(sum(value * value for value in values))
                if magnitude > 0:
                    values = [value / magnitude for value in values]
            metadata = dict(vector.metadata)
            kind = str(metadata.pop("vector_kind", metadata.pop("kind", "dense")))
            vectors.append(
                EmbeddingVector(
                    id=vector.id,
                    model=batch.model,
                    kind=kind,
                    values=tuple(values),
                    metadata=metadata,
                )
            )
        return EmbeddingResponse(vectors=vectors)


class EmbeddingGrpcService:
    """Async gRPC servicer bridging requests into the embedding worker."""

    def __init__(self, worker: EmbeddingWorker) -> None:
        self.worker = worker

    async def EmbedChunks(self, request, context):  # type: ignore[override]
        embed_request = EmbeddingRequest(
            tenant_id=request.tenant_id,
            chunk_ids=list(request.chunk_ids) or None,
            texts=list(request.texts),
            normalize=request.normalize,
            model=request.models[0] if request.models else None,
            correlation_id=getattr(request, "correlation_id", None),
        )
        response = self.worker.run(embed_request)

        from Medical_KG_rev.proto.gen import embedding_pb2  # type: ignore import-error

        reply = embedding_pb2.EmbedChunksResponse()
        for vector in response.vectors:
            message = reply.vectors.add()
            message.chunk_id = vector.id
            message.model = vector.model
            message.kind = vector.kind
            message.dimension = vector.dimension
            message.values.extend(vector.values)
        return reply


__all__ = [
    "EmbeddingGrpcService",
    "EmbeddingModelRegistry",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingVector",
    "EmbeddingWorker",
]

