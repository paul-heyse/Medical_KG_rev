"""Compatibility wrapper around the modular chunking service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

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
        # Backwards compatibility: allow calls without tenant identifier where the first
        # argument is the document_id and the second is the text payload.
        if isinstance(text, ChunkingOptions) and options is None:
            options = text
            text = document_id
            document_id = tenant_id
            tenant_id = "system"
        modular_options = self._translate_options(options)
        try:
            chunks = self._service.chunk_text(
                tenant_id=tenant_id,
                document_id=document_id,
                text=text,
                options=modular_options,
            )
        except TypeError:
            chunks = self._fallback_chunk(tenant_id, document_id, text, options)
        return [self._normalize(chunk) for chunk in chunks]

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
        self,
        tenant_id: str,
        document_id: str,
        text: str | None = None,
        *,
        max_tokens: int,
        overlap: float,
    ) -> list[Chunk]:
        if text is None:
            text = document_id
            document_id = tenant_id
            tenant_id = "system"
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

    def _normalize(self, chunk: Chunk) -> Chunk | SimpleNamespace:
        metadata = dict(getattr(chunk, "meta", getattr(chunk, "metadata", {})))
        metadata.setdefault("segment_type", chunk.granularity)
        token_count = metadata.get("token_count")
        if token_count is None:
            token_count = len(chunk.body.split())
            metadata["token_count"] = token_count
        return SimpleNamespace(
            id=chunk.chunk_id,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            tenant_id=chunk.tenant_id,
            text=chunk.body,
            body=chunk.body,
            granularity=chunk.granularity,
            chunker=chunk.chunker,
            metadata=metadata,
            meta=metadata,
            token_count=token_count,
        )

    def _fallback_chunk(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
        options: ChunkingOptions | None,
    ) -> list[SimpleNamespace]:
        strategy = options.strategy if options else None
        segments: list[str]
        granularity: Granularity
        if strategy in {"section", "section_aware"}:
            segments = text.split("\n\n")
            granularity = "section"
        elif strategy == "paragraph":
            segments = [segment for segment in text.split("\n\n") if segment.strip()]
            granularity = "paragraph"
        elif strategy == "table":
            tables = [segment for segment in text.split("\n\n") if "|" in segment]
            segments = tables or [text]
            granularity = "table"
        elif strategy == "sliding_window":
            tokens = text.split()
            size = (options.max_tokens if options and options.max_tokens else 50)
            overlap = options.overlap if options and options.overlap is not None else 0.0
            stride = max(1, int(size * (1 - overlap)))
            if stride > size:
                stride = size
            segments = [
                " ".join(tokens[i : i + size])
                for i in range(0, len(tokens) or 1, stride)
            ]
            granularity = "window"
        else:
            segments = [text]
            granularity = (
                options.granularity
                if options and options.granularity
                else "paragraph"
            )
        result: list[SimpleNamespace] = []
        for index, segment in enumerate(segment for segment in segments if segment.strip()):
            metadata = {"segment_type": granularity, "token_count": len(segment.split())}
            result.append(
                SimpleNamespace(
                    id=f"{document_id}:{granularity}:{index}",
                    chunk_id=f"{document_id}:{granularity}:{index}",
                    doc_id=document_id,
                    tenant_id=tenant_id,
                    text=segment,
                    body=segment,
                    granularity=granularity,
                    chunker="fallback",
                    metadata=metadata,
                    meta=metadata,
                    token_count=metadata["token_count"],
                )
            )
        return result

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
            params.setdefault("max_tokens", options.max_tokens)
            params.setdefault("target_tokens", options.max_tokens)
        if options.overlap is not None:
            params.setdefault("overlap_ratio", options.overlap)
        return ModularOptions(
            strategy=strategy,
            granularity=granularity,
            params=params or None,
            enable_multi_granularity=options.enable_multi_granularity,
        )
