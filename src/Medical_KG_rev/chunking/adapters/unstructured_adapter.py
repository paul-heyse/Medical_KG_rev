"""Wrapper around Unstructured chunking strategies."""

from __future__ import annotations

from collections.abc import Iterable

from unstructured.chunking.basic import chunk_elements  # type: ignore
from unstructured.chunking.page import chunk_by_page  # type: ignore
from unstructured.chunking.title import chunk_by_title  # type: ignore
from unstructured.partition.auto import partition  # type: ignore

from Medical_KG_rev.models.ir import Document
from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from .mapping import OffsetMapper


def _chunk_with_unstructured(text: str, strategy: str) -> list[str]:
    """Chunk text using Unstructured library."""
    try:
        elements = partition(text=text)
        if strategy == "title":
            chunks = chunk_by_title(elements)
            return [getattr(chunk, "text", str(chunk)) for chunk in chunks if str(chunk).strip()]
        elif strategy == "element":
            chunks = chunk_elements(elements)
            return [getattr(chunk, "text", str(chunk)) for chunk in chunks if str(chunk).strip()]
        elif strategy == "page":
            chunks = chunk_by_page(elements)
            return [getattr(chunk, "text", str(chunk)) for chunk in chunks if str(chunk).strip()]
        raise ChunkerConfigurationError(f"Unknown unstructured strategy '{strategy}'")
    except Exception as exc:
        raise ChunkerConfigurationError(
            f"Unstructured chunking failed for strategy '{strategy}': {exc}"
        ) from exc


class UnstructuredChunker(BaseChunker):
    """Chunker that uses Unstructured library for document segmentation."""

    def __init__(
        self,
        *,
        strategy: str = "element",
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize the Unstructured chunker."""
        super().__init__()
        self.strategy = strategy
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        """Chunk a document using Unstructured library."""
        try:
            from unstructured.chunking.basic import chunk_elements
            from unstructured.chunking.page import chunk_by_page
            from unstructured.chunking.title import chunk_by_title
            from unstructured.partition.auto import partition
        except ImportError as exc:
            raise ChunkerConfigurationError(
                "unstructured must be installed to use UnstructuredChunker"
            ) from exc

        contexts = [ctx for ctx in self.normalizer.iter_block_contexts(document) if ctx.text]
        if not contexts:
            return []

        mapper = OffsetMapper(contexts, token_counter=self.counter)
        aggregated_text = mapper.aggregated_text
        segments = _chunk_with_unstructured(aggregated_text, self.strategy)

        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name="unstructured",
            chunker_version="v1",
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )

        chunks: list[Chunk] = []
        cursor = 0
        for text_segment in segments:
            projection = mapper.project(text_segment, start_hint=cursor)
            cursor = projection.end_offset
            if not projection.contexts:
                continue

            chunk_meta = {
                "segment_type": "framework",
                "framework": "unstructured",
                "strategy": self.strategy,
            }

            chunks.append(assembler.build(projection.contexts, metadata=chunk_meta))

        return chunks

    def explain(self) -> dict[str, object]:
        """Explain the chunking strategy."""
        return {
            "framework": "unstructured",
            "strategy": self.strategy,
        }
