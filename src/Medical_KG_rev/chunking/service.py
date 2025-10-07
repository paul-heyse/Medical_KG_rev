"""High level chunking service bridging configuration and chunkers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section

from .configuration import ChunkerSettings, ChunkingConfig, DEFAULT_CONFIG_PATH
from .factory import ChunkerFactory
from .models import Chunk, Granularity
from .pipeline import MultiGranularityPipeline


@dataclass(slots=True)
class ChunkingOptions:
    strategy: str | None = None
    granularity: Granularity | None = None
    params: dict[str, object] | None = None
    enable_multi_granularity: bool | None = None
    auxiliaries: Sequence[ChunkerSettings] | None = None


class ChunkingService:
    """Entry point consumed by other services."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        registry_factory: ChunkerFactory | None = None,
    ) -> None:
        path = config_path or DEFAULT_CONFIG_PATH
        self.config = ChunkingConfig.load(path)
        self.factory = registry_factory or ChunkerFactory()

    def chunk_document(
        self,
        document: Document,
        *,
        tenant_id: str,
        source: str | None = None,
        options: ChunkingOptions | None = None,
    ) -> list[Chunk]:
        profile = self.config.profile_for_source(source)
        allow_multi = (
            profile.enable_multi_granularity
            if options is None or options.enable_multi_granularity is None
            else options.enable_multi_granularity
        )
        chunker_settings: list[ChunkerSettings]
        if options and options.strategy:
            primary = ChunkerSettings(
                strategy=options.strategy,
                granularity=options.granularity,
                params=dict(options.params or {}),
            )
            auxiliaries = list(options.auxiliaries or [])
            chunker_settings = [primary, *auxiliaries]
        else:
            chunker_settings = [profile.primary, *profile.auxiliaries]
        registered = self.factory.create_many(chunker_settings, allow_experimental=True)
        pipeline = MultiGranularityPipeline(
            chunkers=[(entry.instance, entry.granularity) for entry in registered],
            enable_multi_granularity=allow_multi,
        )
        return pipeline.chunk(document, tenant_id=tenant_id)

    def chunk_text(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
        *,
        options: ChunkingOptions | None = None,
    ) -> list[Chunk]:
        document = self._document_from_text(document_id, text)
        return self.chunk_document(document, tenant_id=tenant_id, source=None, options=options)

    def _document_from_text(self, document_id: str, text: str) -> Document:
        block = Block(
            id=f"{document_id}:block:0",
            type=BlockType.PARAGRAPH,
            text=text,
            spans=[],
            metadata={},
        )
        section = Section(id=f"{document_id}:section:0", title="Document", blocks=[block])
        return Document(id=document_id, source="ad-hoc", title="Document", sections=[section])
