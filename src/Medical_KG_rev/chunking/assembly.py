"""Utilities for constructing chunk models from block contexts."""

from __future__ import annotations

from collections.abc import Iterable

from Medical_KG_rev.models.ir import Document

from .models import Chunk, Granularity
from .provenance import BlockContext, make_chunk_id
from .tokenization import TokenCounter, default_token_counter



class ChunkAssembler:
    """Helper that turns block contexts into Chunk objects."""

    def __init__(
        self,
        document: Document,
        *,
        tenant_id: str,
        chunker_name: str,
        chunker_version: str,
        granularity: Granularity,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.document = document
        self.tenant_id = tenant_id
        self.chunker_name = chunker_name
        self.chunker_version = chunker_version
        self.granularity = granularity
        self._index = 0
        self.counter = token_counter or default_token_counter()

    def build(self, contexts: Iterable[BlockContext], *, metadata: dict | None = None) -> Chunk:
        blocks = list(contexts)
        if not blocks:
            raise ValueError("Cannot build chunk from empty contexts")
        body_parts = [context.text for context in blocks if context.text]
        body = "\n\n".join(part for part in body_parts if part)
        if not body:
            raise ValueError("Chunk body is empty after normalization")
        start = min(context.start_char for context in blocks)
        end = max(context.end_char for context in blocks)
        title_path = blocks[0].title_path
        section_title = blocks[0].section_title
        page_numbers = [context.page_no for context in blocks if context.page_no is not None]
        page_no = page_numbers[0] if page_numbers else None
        chunk_id = make_chunk_id(self.document.id, self.chunker_name, self.granularity, self._index)
        self._index += 1
        token_count = self.counter.count(body)
        meta = {
            "block_ids": [context.block.id for context in blocks],
            "section_id": blocks[0].section.id,
            "token_count": token_count,
        }
        if page_numbers:
            meta.setdefault("page_numbers", page_numbers)
        if metadata:
            meta.update(metadata)
        return Chunk(
            chunk_id=chunk_id,
            doc_id=self.document.id,
            tenant_id=self.tenant_id,
            body=body,
            title_path=title_path,
            section=section_title,
            start_char=start,
            end_char=end,
            granularity=self.granularity,
            chunker=self.chunker_name,
            chunker_version=self.chunker_version,
            page_no=page_no,
            meta=meta,
        )
