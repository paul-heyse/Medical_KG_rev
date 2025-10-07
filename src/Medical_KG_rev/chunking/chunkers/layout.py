"""Chunker that relies on layout heuristics such as headings and font deltas."""

from __future__ import annotations

from typing import Iterable

from Medical_KG_rev.models.ir import BlockType, Document

from ..assembly import ChunkAssembler
from ..models import Chunk, Granularity
from ..provenance import BlockContext, ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from ..ports import BaseChunker


class LayoutHeuristicChunker(BaseChunker):
    name = "layout_heuristic"
    version = "v1"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        max_tokens: int = 600,
        heading_level_key: str = "heading_level",
        font_size_key: str = "font_size",
        whitespace_threshold: float = 0.25,
        font_delta_threshold: float = 2.0,
    ) -> None:
        self.counter = token_counter or default_token_counter()
        self.max_tokens = max_tokens
        self.heading_level_key = heading_level_key
        self.font_size_key = font_size_key
        self.whitespace_threshold = whitespace_threshold
        self.font_delta_threshold = font_delta_threshold
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        contexts = [
            ctx
            for ctx in self.normalizer.iter_block_contexts(document)
            if ctx.text and not ctx.is_table
        ]
        if not contexts:
            return []
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        buffer: list[BlockContext] = []
        token_total = 0
        last_heading_level = None
        last_font_size = None
        for ctx in contexts:
            meta = ctx.block.metadata or {}
            heading_level = int(meta.get(self.heading_level_key, 9))
            font_size = float(meta.get(self.font_size_key, 0.0))
            whitespace_ratio = float(meta.get("whitespace_ratio", 0.0))
            is_header_block = ctx.block.type == BlockType.HEADER
            should_flush = False
            if buffer:
                if heading_level <= (last_heading_level or heading_level):
                    should_flush = True
                if last_font_size is not None and abs(font_size - last_font_size) >= self.font_delta_threshold:
                    should_flush = True
                if whitespace_ratio >= self.whitespace_threshold:
                    should_flush = True
            if should_flush and buffer:
                chunks.append(
                    assembler.build(
                        buffer,
                        metadata={
                            "segment_type": "layout",
                            "heading_level": last_heading_level,
                        },
                    )
                )
                buffer = []
                token_total = 0
            buffer.append(ctx)
            token_total += ctx.token_count
            last_heading_level = heading_level if not is_header_block else 0
            last_font_size = font_size or last_font_size
            if token_total >= self.max_tokens:
                chunks.append(
                    assembler.build(
                        buffer,
                        metadata={
                            "segment_type": "layout",
                            "heading_level": heading_level,
                            "token_budget_exhausted": True,
                        },
                    )
                )
                buffer = []
                token_total = 0
        if buffer:
            chunks.append(
                assembler.build(
                    buffer,
                    metadata={
                        "segment_type": "layout",
                        "heading_level": last_heading_level,
                    },
                )
            )
        return chunks

    def explain(self) -> dict[str, object]:
        return {
            "max_tokens": self.max_tokens,
            "heading_level_key": self.heading_level_key,
            "font_size_key": self.font_size_key,
            "whitespace_threshold": self.whitespace_threshold,
            "font_delta_threshold": self.font_delta_threshold,
        }

