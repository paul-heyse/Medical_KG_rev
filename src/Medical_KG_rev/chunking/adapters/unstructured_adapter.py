"""Wrapper around Unstructured chunking strategies."""

from __future__ import annotations

from typing import Iterable

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..provenance import ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from ..ports import BaseChunker
from .mapping import OffsetMapper


def _chunk_with_unstructured(text: str, strategy: str) -> list[str]:
    try:  # pragma: no cover - optional dependency
        from unstructured.partition.auto import partition  # type: ignore
        from unstructured.chunking.basic import chunk_elements  # type: ignore
        from unstructured.chunking.title import chunk_by_title  # type: ignore
        from unstructured.chunking.page import chunk_by_page  # type: ignore
    except Exception:
        return _fallback_chunk(text, strategy)

    elements = partition(text=text)
    if strategy == "title":
        chunks = chunk_by_title(elements)
        return [getattr(chunk, "text", str(chunk)) for chunk in chunks if str(chunk).strip()]
    if strategy == "element":
        chunks = chunk_elements(elements)
        return [getattr(chunk, "text", str(chunk)) for chunk in chunks if str(chunk).strip()]
    if strategy == "page":
        chunks = chunk_by_page(elements)
        return [getattr(chunk, "text", str(chunk)) for chunk in chunks if str(chunk).strip()]
    raise ChunkerConfigurationError(f"Unsupported unstructured strategy '{strategy}'")


def _fallback_chunk(text: str, strategy: str) -> list[str]:
    lines = text.splitlines()
    if strategy == "title":
        chunks: list[str] = []
        buffer: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped.isupper():
                if buffer:
                    chunks.append("\n".join(buffer).strip())
                    buffer = []
            buffer.append(line)
        if buffer:
            chunks.append("\n".join(buffer).strip())
        return [chunk for chunk in chunks if chunk]
    if strategy == "page":
        return [segment.strip() for segment in text.split("\f") if segment.strip()]
    # default element strategy
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if not line.strip():
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
        else:
            buffer.append(line.strip())
    if buffer:
        paragraphs.append(" ".join(buffer).strip())
    return [para for para in paragraphs if para]


class UnstructuredChunker(BaseChunker):
    name = "unstructured.adapter"
    version = "v1"

    def __init__(
        self,
        *,
        strategy: str = "title",
        token_counter: TokenCounter | None = None,
    ) -> None:
        if strategy not in {"title", "element", "page"}:
            raise ChunkerConfigurationError(
                "strategy must be one of 'title', 'element', or 'page'"
            )
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
        contexts = [
            ctx
            for ctx in self.normalizer.iter_block_contexts(document)
            if ctx.text
        ]
        if not contexts:
            return []
        mapper = OffsetMapper(contexts, token_counter=self.counter)
        segments = _chunk_with_unstructured(mapper.aggregated_text, self.strategy)
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=f"{self.name}.{self.strategy}",
            chunker_version=self.version,
            granularity=granularity or "paragraph",
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        cursor = 0
        for segment in segments:
            projection = mapper.project(segment, start_hint=cursor)
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
        return {"framework": "unstructured", "strategy": self.strategy}

