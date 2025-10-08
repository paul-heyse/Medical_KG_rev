from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import structlog
from pydantic import BaseModel, Field, field_validator

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
from Medical_KG_rev.orchestration.dagster.runtime import StageFactory, build_stage_factory
from Medical_KG_rev.orchestration.dagster.stages import create_default_pipeline_resource
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.dagster.runtime import StageFactory
from Medical_KG_rev.orchestration.dagster.stages import create_stage_plugin_manager
from Medical_KG_rev.orchestration.stages.contracts import ChunkStage, StageContext
from Medical_KG_rev.services.retrieval.chunking_command import ChunkCommand

logger = structlog.get_logger(__name__)


class ChunkCommand(BaseModel):
    """Command object carrying all information required for chunking."""

    tenant_id: str
    document_id: str
    text: str
    strategy: str = Field(default="section")
    chunk_size: int | None = Field(default=None, ge=64, le=4096)
    overlap: float | None = Field(default=None, ge=0.0, lt=1.0)
    options: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = Field(default_factory=lambda: uuid4().hex)
    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidDocumentError("Chunking requests require non-empty text")
        return value

    @field_validator("strategy")
    @classmethod
    def _normalise_strategy(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Strategy must be a string")
        return value.strip().lower()

    def context_metadata(self) -> dict[str, Any]:
        metadata = {
            "strategy": self.strategy,
            "max_tokens": self.chunk_size,
            "overlap": self.overlap,
            "requested_at": self.requested_at.isoformat(),
        }
        metadata.update(self.options_without_text())
        return metadata

    def options_without_text(self) -> dict[str, Any]:
        return {k: v for k, v in self.options.items() if k != "text"}

    def build_document(self) -> Document:
        text = self.text.strip()
        paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
        if not paragraphs:
            paragraphs = [text]
        blocks: list[Block] = []
        for index, paragraph in enumerate(paragraphs):
            blocks.append(
                Block(
                    id=f"{self.document_id}:block:{index}",
                    type=BlockType.PARAGRAPH,
                    text=paragraph,
                    metadata={"paragraph_index": index},
                )
            )
        metadata = self.options_without_text()
        section_title = metadata.get("section_title") if isinstance(metadata, Mapping) else None
        section = Section(
            id=f"{self.document_id}:section:0",
            title=str(section_title) if section_title else None,
            blocks=blocks,
        )
        source = metadata.get("source") if isinstance(metadata, Mapping) else None
        document = Document(
            id=self.document_id,
            source=str(source or "gateway"),
            title=metadata.get("title") if isinstance(metadata, Mapping) else None,
            sections=[section],
            metadata=dict(metadata or {}),
        )
        return document


@dataclass(slots=True)
class ChunkingOptions:
    """Internal options passed to the chunking stage."""

    strategy: str
    max_tokens: int | None
    overlap: float | None
    metadata: Mapping[str, Any]


class ChunkingService:
    """Thin wrapper around the Dagster Haystack chunk stage."""

    _DEFAULT_PIPELINE = "gateway-direct"
    _DEFAULT_VERSION = "v1"
    _SUPPORTED_STRATEGIES = ("semantic", "paragraph", "section", "table", "sliding-window")

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
            pipeline_resource = create_default_pipeline_resource()
            job_ledger = JobLedger()
            stage_factory = build_stage_factory(manager, pipeline_resource, job_ledger)
            plugin_manager = create_stage_plugin_manager(manager)
            stage_factory = StageFactory(plugin_manager)
        self._stage_factory = stage_factory

    def chunk(self, command: ChunkCommand) -> list[Chunk]:
        """Chunk raw text into retrieval-ready spans."""

        if command.strategy not in self._SUPPORTED_STRATEGIES:
            raise ChunkerConfigurationError(
                f"Unsupported chunking strategy '{command.strategy}'"
            )

        chunk_options = ChunkingOptions(
            strategy=command.strategy,
            max_tokens=command.chunk_size,
            overlap=command.overlap,
            metadata=dict(command.metadata),
        )
        stage = self._resolve_stage()
        options = ChunkingOptions(
            strategy=command.strategy,
            max_tokens=command.chunk_size,
            overlap=command.overlap,
            metadata=command.options_without_text(),
        )
        context = StageContext(
            tenant_id=command.tenant_id,
            doc_id=command.document_id,
            correlation_id=command.correlation_id,
            metadata=self._context_metadata(options),
            pipeline_name=self._DEFAULT_PIPELINE,
            pipeline_version=self._DEFAULT_VERSION,
        )
        document = command.build_document()
        logger.debug(
            "gateway.chunking.execute",
            tenant_id=command.tenant_id,
            doc_id=command.document_id,
            strategy=command.strategy,
        )
            metadata=self._context_metadata(chunk_options, command),
            pipeline_name=self._DEFAULT_PIPELINE,
            pipeline_version=self._DEFAULT_VERSION,
        )
        document = self._build_document(command.document_id, command.text, command.metadata)
        logger.bind(**command.log_context()).debug("gateway.chunking.execute")
        try:
            chunks = stage.execute(context, document)
        except ChunkerConfigurationError:
            raise
        except InvalidDocumentError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception(
                "gateway.chunking.stage_error",
                doc_id=command.document_id,
                error=str(exc),
            )
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

    def _context_metadata(self, options: ChunkingOptions) -> dict[str, Any]:
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

    def _context_metadata(self, options: ChunkingOptions, command: ChunkCommand) -> dict[str, Any]:
        metadata = {
            "strategy": options.strategy,
            "max_tokens": options.max_tokens,
            "overlap": options.overlap,
            "profile": command.profile,
            "issued_at": command.issued_at_iso,
        }
        metadata.update(dict(options.metadata))
        metadata.update(dict(command.context))
        return metadata


__all__ = ["Chunk", "ChunkCommand", "ChunkingOptions", "ChunkingService"]
