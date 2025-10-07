"""Sliding window chunker with overlap support."""

from __future__ import annotations

from typing import Iterable

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..models import Chunk, Granularity
from ..provenance import BlockContext, ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from ..ports import BaseChunker


class SlidingWindowChunker(BaseChunker):
    name = "sliding_window"
    version = "v1"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        target_tokens: int = 512,
        overlap_ratio: float = 0.25,
        min_tokens: int = 128,
    ) -> None:
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be between 0 and 1")
        self.counter = token_counter or default_token_counter()
        self.target_tokens = target_tokens
        self.overlap_ratio = overlap_ratio
        self.min_tokens = min_tokens
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
            granularity=granularity or "window",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        index = 0
        while index < len(contexts):
            window: list[BlockContext] = []
            token_total = 0
            j = index
            while j < len(contexts) and token_total < self.target_tokens:
                window.append(contexts[j])
                token_total += contexts[j].token_count
                j += 1
            if not window:
                break
            chunks.append(
                assembler.build(window, metadata={"segment_type": "window"})
            )
            step = max(1, int(len(window) * (1 - self.overlap_ratio)))
            index += step
        return chunks

    def explain(self) -> dict[str, object]:
        return {
            "target_tokens": self.target_tokens,
            "overlap_ratio": self.overlap_ratio,
            "min_tokens": self.min_tokens,
        }
