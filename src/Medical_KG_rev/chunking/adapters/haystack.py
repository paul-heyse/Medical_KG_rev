"""Haystack preprocessor chunker wrapper."""

from __future__ import annotations

from typing import Iterable

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from .mapping import OffsetMapper


def _create_preprocessor(**kwargs):
    try:  # pragma: no cover - optional dependency
        from haystack.components.preprocessors import DocumentSplitter  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ChunkerConfigurationError(
            "haystack-ai must be installed to use HaystackPreprocessorChunker"
        ) from exc
    return DocumentSplitter(**kwargs)


class HaystackPreprocessorChunker(BaseChunker):
    name = "haystack.preprocessor"
    version = "v1"

    def __init__(
        self,
        *,
        split_length: int = 200,
        split_overlap: int = 20,
        split_by: str = "word",
        token_counter: TokenCounter | None = None,
        preprocessor: object | None = None,
    ) -> None:
        self.counter = token_counter or default_token_counter()
        self.preprocessor = preprocessor or _create_preprocessor(
            split_length=split_length,
            split_overlap=split_overlap,
            split_by=split_by,
            respect_sentence_boundary=True,
        )
        self.split_by = split_by
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        contexts = [ctx for ctx in self.normalizer.iter_block_contexts(document) if ctx.text]
        if not contexts:
            return []
        mapper = OffsetMapper(contexts, token_counter=self.counter)

        # Convert our Document to Haystack Document format
        try:  # pragma: no cover - optional dependency
            from haystack import Document as HaystackDocument  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ChunkerConfigurationError(
                "haystack-ai must be installed to use HaystackPreprocessorChunker"
            ) from exc

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
