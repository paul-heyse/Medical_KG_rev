"""Compatibility wrapper around the modular chunking service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from Medical_KG_rev.chunking import (
    Chunk,
    ChunkingOptions as ModularOptions,
    ChunkingService as ModularChunkingService,
    Granularity,
)
from Medical_KG_rev.chunking.exceptions import ChunkerConfigurationError

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

    def chunk(self, *args, **kwargs) -> list[Chunk]:
        """Chunk text, supporting both legacy and modern call signatures."""

        tenant_id = kwargs.pop("tenant_id", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        document_id: str
        text: str
        options: ChunkingOptions | None

        if len(args) >= 3 and tenant_id is None and all(
            isinstance(value, str) for value in args[:3]
        ):
            tenant_id = args[0]
            document_id = args[1]
            text = args[2]
            options = args[3] if len(args) > 3 else None
        elif len(args) >= 2:
            document_id = args[0]
            text = args[1]
            options = args[2] if len(args) > 2 else None
        else:  # pragma: no cover - defensive guard
            raise TypeError("chunk() missing required positional arguments")

        if tenant_id is None:
            tenant_id = "default"

        modular_options = self._translate_options(options)
        try:
            return self._service.chunk_text(
                tenant_id=tenant_id,
                document_id=document_id,
                text=text,
                options=modular_options,
            )
        except ChunkerConfigurationError:
            return self._fallback_chunks(tenant_id, document_id, text)

    def chunk_sections(self, tenant_id: str, document_id: str, text: str) -> list[Chunk]:
        return self.chunk(
            document_id,
            text,
            ChunkingOptions(strategy="section_aware", granularity="section"),
            tenant_id=tenant_id,
        )

    def chunk_paragraphs(self, tenant_id: str, document_id: str, text: str) -> list[Chunk]:
        return self.chunk(
            document_id,
            text,
            ChunkingOptions(strategy="semantic_splitter", granularity="paragraph"),
            tenant_id=tenant_id,
        )

    def chunk_tables(self, tenant_id: str, document_id: str, text: str) -> list[Chunk]:
        return self.chunk(
            document_id,
            text,
            ChunkingOptions(strategy="table", granularity="table"),
            tenant_id=tenant_id,
        )

    def sliding_window(
        self,
        document_id: str,
        text: str,
        max_tokens: int,
        overlap: float,
        *,
        tenant_id: str = "default",
    ) -> list[Chunk]:
        modular_chunks = self.chunk(
            document_id,
            text,
            ChunkingOptions(
                strategy="sliding_window",
                granularity="window",
                max_tokens=max_tokens,
                overlap=overlap,
            ),
            tenant_id=tenant_id,
        )
        if len(modular_chunks) <= 1 and len(text.split()) > max_tokens:
            return self._sliding_window_legacy(
                tenant_id=tenant_id,
                document_id=document_id,
                text=text,
                max_tokens=max_tokens,
                overlap=overlap,
            )
        return modular_chunks

    def _sliding_window_legacy(
        self,
        *,
        tenant_id: str,
        document_id: str,
        text: str,
        max_tokens: int,
        overlap: float,
    ) -> list[LegacyChunk]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        tokens = text.split()
        if not tokens:
            return []
        stride = max(1, int(round(max_tokens * (1 - overlap))))
        legacy_chunks: list[LegacyChunk] = []
        index = 0
        for start in range(0, len(tokens), stride):
            window_tokens = tokens[start : start + max_tokens]
            if not window_tokens:
                break
            body = " ".join(window_tokens)
            chunk_id = f"{document_id}:sliding_window:window:{index}"
            metadata = {
                "segment_type": "window",
                "token_count": len(window_tokens),
                "chunker": "sliding_window",
                "chunker_version": "legacy",
            }
            legacy_chunks.append(
                LegacyChunk(
                    id=chunk_id,
                    text=body,
                    metadata=metadata,
                    chunker="sliding_window",
                    chunker_version="legacy",
                    granularity="window",
                    tenant_id=tenant_id,
                    document_id=document_id,
                    token_count=len(window_tokens),
                )
            )
            index += 1
            if len(window_tokens) < max_tokens:
                break
        return legacy_chunks

    def _table_chunks_legacy(
        self,
        *,
        document: Document,
        tenant_id: str,
    ) -> list[LegacyChunk]:
        legacy_chunks: list[LegacyChunk] = []
        index = 0
        for section in document.sections:
            title = (section.title or "").strip()
            for block in section.blocks:
                text = (block.text or "").strip()
                if not text:
                    continue
                if block.type != BlockType.TABLE and "|" not in text:
                    continue
                body = f"{title}\n{text}".strip() if title else text
                token_count = len(body.split())
                metadata = {
                    "segment_type": "table",
                    "chunker": "table",
                    "chunker_version": "legacy",
                    "section_id": section.id,
                    "block_ids": [block.id],
                    "token_count": token_count,
                }
                legacy_chunks.append(
                    LegacyChunk(
                        id=f"{document.id}:table:table:{index}",
                        text=body,
                        metadata=metadata,
                        chunker="table",
                        chunker_version="legacy",
                        granularity="table",
                        tenant_id=tenant_id,
                        document_id=document.id,
                        token_count=token_count,
                    )
                )
                index += 1
        return legacy_chunks

    def _build_document(self, document_id: str, text: str) -> Document:
        paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
        sections: list[Section] = []
        current_blocks: list[Block] = []
        current_title = "Document"
        section_index = 0
        block_index = 0

        def flush_section() -> None:
            nonlocal current_blocks, section_index
            if not current_blocks:
                return
            sections.append(
                Section(
                    id=f"{document_id}:section:{section_index}",
                    title=current_title,
                    blocks=list(current_blocks),
                )
            )
            section_index += 1
            current_blocks = []

        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines:
                continue
            heading: str | None = None
            body_lines: Iterable[str] = lines
            if len(lines) > 1 and self._looks_like_heading(lines[0]):
                heading = lines[0]
                body_lines = lines[1:]
            if heading and current_blocks:
                if heading != current_title:
                    flush_section()
                current_title = heading or current_title
            elif heading:
                current_title = heading
            content_body = "\n".join(body_lines).strip()
            if heading and content_body:
                content = f"{heading}\n{content_body}".strip()
            elif heading:
                content = heading
            else:
                content = content_body
            if not content:
                continue
            block_type = BlockType.TABLE if any("|" in line for line in content.splitlines()) else BlockType.PARAGRAPH
            metadata: dict[str, object] = {}
            if block_type == BlockType.TABLE:
                metadata["is_table"] = True
            block = Block(
                id=f"{document_id}:block:{block_index}",
                type=block_type,
                text=content,
                spans=(),
                metadata=metadata,
            )
            block_index += 1
            current_blocks.append(block)

        flush_section()
        if not sections:
            sections.append(
                Section(
                    id=f"{document_id}:section:{section_index}",
                    title=current_title,
                    blocks=list(current_blocks),
                )
            )
        return Document(
            id=document_id,
            source="ad-hoc",
            title="Document",
            sections=sections,
        )

    def _fallback_chunks(self, tenant_id: str, document_id: str, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        cursor = 0
        for idx, segment in enumerate(text.splitlines(keepends=True)):
            body = segment.strip()
            length = len(segment)
            if not body:
                cursor += length
                continue
            start_offset = segment.index(body)
            start = cursor + start_offset
            end = start + len(body)
            chunk_id = f"{document_id}-fallback-{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document_id,
                    tenant_id=tenant_id,
                    body=body,
                    title_path=(),
                    section=None,
                    start_char=start,
                    end_char=end,
                    granularity="paragraph",
                    chunker="fallback",
                    chunker_version="v1",
                    meta={},
                )
            )
            cursor += length
        if not chunks and text.strip():
            stripped = text.strip()
            start = text.index(stripped)
            end = start + len(stripped)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-fallback-0",
                    doc_id=document_id,
                    tenant_id=tenant_id,
                    body=stripped,
                    title_path=(),
                    section=None,
                    start_char=start,
                    end_char=end,
                    granularity="document",
                    chunker="fallback",
                    chunker_version="v1",
                    meta={},
                )
            )
        return chunks

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
        canonical = strategy or "section_aware"
        if options.max_tokens is not None:
            translated = {
                "section_aware": "target_tokens",
                "sliding_window": "target_tokens",
                "semantic_splitter": "min_tokens",
                "clinical_role": "min_tokens",
            }.get(canonical)
            if translated:
                params.setdefault(translated, options.max_tokens)
        if options.overlap is not None and canonical == "sliding_window":
            params.setdefault("overlap_ratio", options.overlap)
        return ModularOptions(
            strategy=strategy,
            granularity=granularity,
            params=params or None,
            enable_multi_granularity=options.enable_multi_granularity,
        )
@dataclass(slots=True)
class LegacyChunk:
    id: str
    text: str
    metadata: dict[str, object]
    chunker: str
    chunker_version: str
    granularity: Granularity
    tenant_id: str
    document_id: str
    token_count: int

    @property
    def chunk_id(self) -> str:
        return self.id

    @property
    def body(self) -> str:
        return self.text

    @property
    def doc_id(self) -> str:
        return self.document_id

    @property
    def meta(self) -> dict[str, object]:
        return self.metadata

