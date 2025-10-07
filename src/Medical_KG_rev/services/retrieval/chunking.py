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
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section

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
        canonical = strategy or "section_aware"
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

