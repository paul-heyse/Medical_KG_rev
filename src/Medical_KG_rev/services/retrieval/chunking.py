"""Compatibility wrapper around the modular chunking service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from Medical_KG_rev.chunking import (
    Chunk,
    ChunkingOptions as ModularOptions,
    ChunkingService as ModularChunkingService,
    Granularity,
)

STRATEGY_ALIASES: dict[str, tuple[str, Granularity]] = {
    "section": ("section_aware", "section"),
    "section_aware": ("section_aware", "section"),
    "semantic": ("semantic_splitter", "paragraph"),
    "paragraph": ("semantic_splitter", "paragraph"),
    "semantic_splitter": ("semantic_splitter", "paragraph"),
    "table": ("table", "table"),
    "sliding-window": ("sliding_window", "window"),
    "sliding_window": ("sliding_window", "window"),
    "clinical_role": ("clinical_role", "paragraph"),
}


@dataclass(slots=True)
class ChunkingOptions:
    """Thin wrapper mapping legacy options to the modular chunker."""

    strategy: str | None = None
    granularity: Granularity | None = None
    max_tokens: int | None = None
    overlap: float | None = None
    params: dict[str, object] = field(default_factory=dict)
    enable_multi_granularity: bool | None = None


class ChunkingService:
    """Service used by retrieval components for chunking text."""

    def __init__(self, *, config_path: Path | None = None) -> None:
        self._service = ModularChunkingService(config_path=config_path)

    def chunk(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
        options: ChunkingOptions | None = None,
    ) -> list[Chunk]:
        modular_options = self._translate_options(options)
        return self._service.chunk_text(
            tenant_id=tenant_id,
            document_id=document_id,
            text=text,
            options=modular_options,
        )

    def chunk_sections(self, tenant_id: str, document_id: str, text: str) -> list[Chunk]:
        return self.chunk(
            tenant_id,
            document_id,
            text,
            ChunkingOptions(strategy="section_aware", granularity="section"),
        )

    def chunk_paragraphs(self, tenant_id: str, document_id: str, text: str) -> list[Chunk]:
        return self.chunk(
            tenant_id,
            document_id,
            text,
            ChunkingOptions(strategy="semantic_splitter", granularity="paragraph"),
        )

    def chunk_tables(self, tenant_id: str, document_id: str, text: str) -> list[Chunk]:
        return self.chunk(
            tenant_id,
            document_id,
            text,
            ChunkingOptions(strategy="table", granularity="table"),
        )

    def sliding_window(
        self, tenant_id: str, document_id: str, text: str, max_tokens: int, overlap: float
    ) -> list[Chunk]:
        return self.chunk(
            tenant_id,
            document_id,
            text,
            ChunkingOptions(
                strategy="sliding_window",
                granularity="window",
                max_tokens=max_tokens,
                overlap=overlap,
            ),
        )

    def available_strategies(self) -> list[str]:
        return self._service.list_strategies()

    def _translate_options(self, options: ChunkingOptions | None) -> ModularOptions | None:
        if options is None:
            return None
        params = dict(options.params)
        strategy = options.strategy
        granularity = options.granularity
        if strategy:
            alias = STRATEGY_ALIASES.get(strategy)
            if alias:
                strategy, default_granularity = alias
                if granularity is None:
                    granularity = default_granularity
        if options.max_tokens is not None:
            params.setdefault("target_tokens", options.max_tokens)
        if options.overlap is not None:
            params.setdefault("overlap_ratio", options.overlap)
        return ModularOptions(
            strategy=strategy,
            granularity=granularity,
            params=params or None,
            enable_multi_granularity=options.enable_multi_granularity,
        )
