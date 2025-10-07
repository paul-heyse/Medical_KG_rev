"""Chunker that keeps MinerU tables and context together."""

from __future__ import annotations

from collections.abc import Sequence

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import BlockContext, ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter


class TableAwareChunker(BaseChunker):
    """Chunker that preserves table boundaries and captions."""

    name = "mineru.table-aware"
    version = "v1"

    def __init__(
        self,
        *,
        max_paragraphs: int = 4,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.max_paragraphs = max(1, max_paragraphs)
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Sequence | None = None,
    ) -> list[Chunk]:
        contexts = list(self.normalizer.iter_block_contexts(document))
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

        def flush_buffer() -> None:
            if not buffer:
                return
            metadata = {"segment_type": "text", "strategy": self.name}
            chunks.append(assembler.build(list(buffer), metadata=metadata))
            buffer.clear()

        for context in contexts:
            if context.is_table:
                flush_buffer()
                chunks.append(assembler.build([context], metadata=self._table_metadata(context)))
                continue
            buffer.append(context)
            if len(buffer) >= self.max_paragraphs or context.block.metadata.get("table_context_break"):
                flush_buffer()

        flush_buffer()
        return chunks

    def explain(self) -> dict[str, object]:
        return {"strategy": self.name, "max_paragraphs": self.max_paragraphs}

    def _table_metadata(self, context: BlockContext) -> dict[str, object]:
        metadata: dict[str, object] = {
            "segment_type": "table",
            "strategy": self.name,
            "table_block_id": context.block.id,
        }
        table = context.block.table
        if table is not None:
            metadata.update(
                {
                    "table_id": table.id,
                    "table_caption": table.caption or "",
                    "table_headers": list(table.headers),
                }
            )
            exports = getattr(table, "metadata", {}).get("exports") if hasattr(table, "metadata") else None
            if isinstance(exports, dict):
                for key in ("json", "csv", "markdown"):
                    value = exports.get(key)
                    if value:
                        metadata[f"table_{key}"] = value
        return metadata


__all__ = ["TableAwareChunker"]
