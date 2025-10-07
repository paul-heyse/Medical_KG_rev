"""Section aware chunker implementation."""

from __future__ import annotations

from itertools import groupby
from typing import Iterable

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..models import Chunk, Granularity
from ..provenance import BlockContext, ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from ..ports import BaseChunker


class SectionAwareChunker(BaseChunker):
    """Chunker that respects document sections with domain aware defaults."""

    name = "section_aware"
    version = "v1"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        target_tokens: int = 450,
        min_tokens: int = 180,
        max_tokens: int = 900,
        preserve_tables: bool = True,
    ) -> None:
        self.counter = token_counter or default_token_counter()
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.preserve_tables = preserve_tables
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
            context
            for context in self.normalizer.iter_block_contexts(document)
            if context.text
        ]
        if not contexts:
            return []
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "section",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        for _, section_blocks in groupby(contexts, key=lambda ctx: ctx.section.id):
            section_list = list(section_blocks)
            if self.preserve_tables:
                table_chunks = [ctx for ctx in section_list if ctx.is_table]
                section_list = [ctx for ctx in section_list if not ctx.is_table]
                for table_ctx in table_chunks:
                    chunks.append(
                        assembler.build([table_ctx], metadata={"is_table": True})
                    )
            if not section_list:
                continue
            buffer: list[BlockContext] = []
            token_budget = 0
            for ctx in section_list:
                buffer.append(ctx)
                token_budget += ctx.token_count
                if token_budget >= self.target_tokens:
                    chunks.append(
                        assembler.build(buffer, metadata={"segment_type": "section"})
                    )
                    buffer = []
                    token_budget = 0
            if buffer:
                if chunks and buffer and token_budget < self.min_tokens:
                    chunks.pop()
                    merged_contexts = list(section_list[-len(buffer) :])
                    chunks.append(
                        assembler.build(
                            merged_contexts,
                            metadata={"segment_type": "section", "merged": True},
                        )
                    )
                else:
                    chunks.append(
                        assembler.build(buffer, metadata={"segment_type": "section"})
                    )
        return chunks

    def explain(self) -> dict[str, object]:
        return {
            "target_tokens": self.target_tokens,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "preserve_tables": self.preserve_tables,
        }
