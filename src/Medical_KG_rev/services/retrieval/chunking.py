from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4

import structlog
from Medical_KG_rev.adapters import get_plugin_manager
from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.chunking.exceptions import (
    ChunkerConfigurationError,
    ChunkingUnavailableError,
    InvalidDocumentError,
)
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.runtime import StageFactory
from Medical_KG_rev.orchestration.dagster.stages import build_default_stage_factory
from Medical_KG_rev.orchestration.stages.contracts import ChunkStage, StageContext

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ChunkingOptions:
    """Options accepted by the gateway chunking endpoint."""

    strategy: str = "semantic"
    max_tokens: int | None = None
    overlap: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkingService:
    """Thin wrapper around the Dagster Haystack chunk stage."""

    _DEFAULT_PIPELINE = "gateway-direct"
    _DEFAULT_VERSION = "v1"
    _SUPPORTED_STRATEGIES = ("semantic", "paragraph", "section")

    def __init__(
        self,
        *,
        stage_factory: StageFactory | None = None,
        adapter_manager: AdapterPluginManager | None = None,
        chunk_stage: ChunkStage | None = None,
        stage_definition: StageDefinition | None = None,
    ) -> None:
        self._chunk_stage = chunk_stage
        self._stage_definition = stage_definition or StageDefinition(name="chunk", type="chunk")
        if stage_factory is None:
            manager = adapter_manager or get_plugin_manager()
            registry = build_default_stage_factory(manager)
            stage_factory = StageFactory(registry)
        self._stage_factory = stage_factory

    def chunk(self, *args: Any, **kwargs: Any) -> list[Chunk]:
        """Chunk raw text into retrieval-ready spans."""

        tenant_id = kwargs.pop("tenant_id", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if len(args) < 3:
            raise TypeError("chunk() missing required positional arguments")
        if tenant_id is None:
            tenant_id = args[0]
            document_id = args[1]
            text = args[2]
            options = args[3] if len(args) > 3 else None
        else:
            document_id = args[0]
            text = args[1]
            options = args[2] if len(args) > 2 else None
        if not isinstance(text, str) or not text.strip():
            raise InvalidDocumentError("Text payload must be a non-empty string")
        chunk_options = options if isinstance(options, ChunkingOptions) else ChunkingOptions()
        stage = self._resolve_stage()
        context = StageContext(
            tenant_id=tenant_id,
            doc_id=document_id,
            correlation_id=uuid4().hex,
            metadata=self._context_metadata(chunk_options),
            pipeline_name=self._DEFAULT_PIPELINE,
            pipeline_version=self._DEFAULT_VERSION,
        )
        document = self._build_document(document_id, text, chunk_options.metadata)
        logger.debug(
            "gateway.chunking.execute",
            tenant_id=tenant_id,
            doc_id=document_id,
            strategy=chunk_options.strategy,
        )
        try:
            chunks = stage.execute(context, document)
        except ChunkerConfigurationError:
            raise
        except InvalidDocumentError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("gateway.chunking.stage_error", doc_id=document_id, error=str(exc))
            raise ChunkingUnavailableError(retry_after=30.0) from exc
        return list(chunks)

    def available_strategies(self) -> list[str]:
        return list(self._SUPPORTED_STRATEGIES)

    def _resolve_stage(self) -> ChunkStage:
        if self._chunk_stage is not None:
            return self._chunk_stage
        stage = self._stage_factory.resolve(self._DEFAULT_PIPELINE, self._stage_definition)
        if not isinstance(stage, ChunkStage):  # pragma: no cover - defensive guard
            raise TypeError("Resolved chunk stage does not implement ChunkStage")
        return stage

    def _build_document(self, document_id: str, text: str, metadata: Mapping[str, Any]) -> Document:
        paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]
        blocks: list[Block] = []
        for index, paragraph in enumerate(paragraphs):
            blocks.append(
                Block(
                    id=f"{document_id}:block:{index}",
                    type=BlockType.PARAGRAPH,
                    text=paragraph,
                    metadata={"paragraph_index": index},
                )
            )
        section_title = metadata.get("section_title") if isinstance(metadata, Mapping) else None
        section = Section(
            id=f"{document_id}:section:0",
            title=str(section_title) if section_title else None,
            blocks=blocks,
        )
        source = metadata.get("source") if isinstance(metadata, Mapping) else None
        document = Document(
            id=document_id,
            source=str(source or "gateway"),
            title=metadata.get("title") if isinstance(metadata, Mapping) else None,
            sections=[section],
            metadata=dict(metadata or {}),
        )
        return document

    def _context_metadata(self, options: ChunkingOptions) -> dict[str, Any]:
        metadata = {
            "strategy": options.strategy,
            "max_tokens": options.max_tokens,
            "overlap": options.overlap,
        }
        metadata.update(dict(options.metadata))
        return metadata


__all__ = ["Chunk", "ChunkingOptions", "ChunkingService"]
