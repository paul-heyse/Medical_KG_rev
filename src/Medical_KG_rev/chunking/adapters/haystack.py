"""Haystack preprocessor chunker wrapper."""

from __future__ import annotations

from collections.abc import Iterable

from haystack import Document as HaystackDocument  # type: ignore
from haystack.components.preprocessors import DocumentSplitter  # type: ignore

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from .mapping import OffsetMapper


class HaystackPreprocessorChunker(BaseChunker):
    """Chunker that uses Haystack's DocumentSplitter component."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize the Haystack preprocessor chunker."""
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_counter = token_counter or default_token_counter()

    def chunk(self, document: Document) -> Iterable[Chunk]:
        """Chunk a document using Haystack's DocumentSplitter."""
        try:
            from haystack.components.preprocessors import DocumentSplitter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ChunkerConfigurationError(
                "haystack-ai must be installed to use HaystackPreprocessorChunker"
            ) from exc

        # Convert to Haystack document format
        mapper = OffsetMapper(document)
        haystack_doc = HaystackDocument(content=mapper.aggregated_text)
        result = self.preprocessor.run(documents=[haystack_doc])
        docs = result["documents"]

        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        cursor = 0
        for doc in docs:
            content = getattr(doc, "content", None)
            if not content:
                continue
            metadata = getattr(doc, "meta", {}) or {}
            projection = mapper.project(str(content), start_hint=cursor)
            cursor = projection.end_offset
            if not projection.contexts:
                continue
            chunk_meta = {
                "segment_type": "framework",
                "framework": "haystack",
                "split_by": self.split_by,
            }
            if metadata:
                chunk_meta.update(metadata)
            chunks.append(assembler.build(projection.contexts, metadata=chunk_meta))
        return chunks

    def explain(self) -> dict[str, object]:
        return {"framework": "haystack", "split_by": self.split_by}
