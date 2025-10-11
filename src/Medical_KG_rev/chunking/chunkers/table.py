"""Table specific chunker preserving atomic structure."""

from __future__ import annotations

from collections.abc import Iterable

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import ProvenanceNormalizer
from ..tables import TableHandler
from ..tokenization import TokenCounter, default_token_counter



class TableChunker(BaseChunker):
    name = "table"
    version = "v1"

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        mode: str = "row",
    ) -> None:
        self.counter = token_counter or default_token_counter()
        if mode not in {"row", "rowgroup", "summary"}:
            raise ValueError("Unsupported table chunking mode")
        self.mode = mode
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)
        self.table_handler = TableHandler(token_counter=self.counter, mode=mode)

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
            if ctx.text and ctx.is_table
        ]
        if not contexts:
            return []
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=f"table_{self.mode}",
            chunker_version=self.version,
            granularity=granularity or "table",
            token_counter=self.counter,
        )
        chunks = []
        for ctx in contexts:
            for table_slice in self.table_handler.iter_slices(ctx):
                metadata = dict(table_slice.metadata)
                metadata.setdefault("mode", self.mode)
                metadata.setdefault("segment_type", "table")
                chunks.append(assembler.build(table_slice.contexts, metadata=metadata))
        return chunks

    def explain(self) -> dict[str, object]:
        return {"mode": self.mode}
