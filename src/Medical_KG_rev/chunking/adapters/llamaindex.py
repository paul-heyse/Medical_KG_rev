"""Adapters that wrap LlamaIndex node parsers for the chunking system."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Type

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import ProvenanceNormalizer
from ..tokenization import TokenCounter, default_token_counter
from .mapping import OffsetMapper

try:  # pragma: no cover - optional dependency
    from llama_index.core.node_parser import (  # type: ignore
        HierarchicalNodeParser,
        SemanticSplitterNodeParser,
        SentenceSplitterNodeParser,
    )
    from llama_index.core.schema import Document as LlamaDocument  # type: ignore
except Exception as exc:  # pragma: no cover - graceful fallback when llama-index missing
    raise ImportError("llama-index is required for LlamaIndex chunkers") from exc


def _load_node_parser(class_name: str, **kwargs: Any) -> tuple[Any, Type[Any]]:
    """Instantiate a LlamaIndex node parser and return it with the document class."""
    mapping: dict[str, Type[Any]] = {
        "SemanticSplitterNodeParser": SemanticSplitterNodeParser,
        "HierarchicalNodeParser": HierarchicalNodeParser,
        "SentenceSplitterNodeParser": SentenceSplitterNodeParser,
    }
    parser_cls = mapping.get(class_name)
    if parser_cls is None:
        raise ChunkerConfigurationError(f"Unsupported LlamaIndex parser '{class_name}'")
    parser = parser_cls(**kwargs)
    return parser, LlamaDocument


class _BaseLlamaIndexChunker(BaseChunker):
    """Shared functionality for LlamaIndex-based chunkers."""

    framework = "llama_index"
    version = "v1"

    def __init__(
        self,
        *,
        parser: Any,
        document_cls: Type[Any],
        name: str,
        parser_name: str,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.name = name
        self.parser = parser
        self.document_cls = document_cls
        self.parser_name = parser_name
        self.counter = token_counter or default_token_counter()
        self.normalizer = ProvenanceNormalizer(token_counter=self.counter)

    def _parse(self, text: str) -> list[tuple[str, dict[str, object]]]:
        document = self.document_cls(text=text)
        nodes = self.parser.get_nodes_from_documents([document])
        segments: list[tuple[str, dict[str, object]]] = []
        for node in nodes:
            maybe_content = getattr(node, "text", None) or getattr(node, "get_content", None)
            content = maybe_content() if callable(maybe_content) else maybe_content
            if not content:
                continue
            metadata = getattr(node, "metadata", {}) or {}
            segments.append((str(content), dict(metadata)))
        return segments

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
        segments = self._parse(mapper.aggregated_text)
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name=self.name,
            chunker_version=self.version,
            granularity=granularity or self.default_granularity,
            token_counter=self.counter,
        )
        chunks: list[Chunk] = []
        cursor = 0
        for text_segment, metadata in segments:
            projection = mapper.project(text_segment, start_hint=cursor)
            cursor = projection.end_offset
            if not projection.contexts:
                continue
            chunk_meta = {
                "segment_type": "framework",
                "framework": self.framework,
                "parser": self.parser_name,
            }
            if metadata:
                chunk_meta.update(metadata)
            chunks.append(assembler.build(projection.contexts, metadata=chunk_meta))
        return chunks

    def explain(self) -> dict[str, object]:
        return {"framework": self.framework, "parser": self.parser_name}


class LlamaIndexNodeParserChunker(_BaseLlamaIndexChunker):
    """Chunker backed by the LlamaIndex semantic splitter parser."""

    def __init__(
        self,
        *,
        embed_model: str | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        parser, document_cls = _load_node_parser(
            "SemanticSplitterNodeParser",
            embed_model=embed_model,
        )
        super().__init__(
            parser=parser,
            document_cls=document_cls,
            name="llama_index.semantic_splitter",
            parser_name="SemanticSplitterNodeParser",
            token_counter=token_counter,
        )


class LlamaIndexHierarchicalChunker(_BaseLlamaIndexChunker):
    """Chunker backed by the LlamaIndex hierarchical node parser."""

    def __init__(
        self,
        *,
        chunk_sizes: tuple[int, ...] = (2048, 512),
        token_counter: TokenCounter | None = None,
    ) -> None:
        parser, document_cls = _load_node_parser(
            "HierarchicalNodeParser",
            chunk_sizes=chunk_sizes,
        )
        super().__init__(
            parser=parser,
            document_cls=document_cls,
            name="llama_index.hierarchical",
            parser_name="HierarchicalNodeParser",
            token_counter=token_counter,
        )


class LlamaIndexSentenceChunker(_BaseLlamaIndexChunker):
    """Chunker backed by the LlamaIndex sentence splitter parser."""

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        token_counter: TokenCounter | None = None,
    ) -> None:
        parser, document_cls = _load_node_parser(
            "SentenceSplitterNodeParser",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        super().__init__(
            parser=parser,
            document_cls=document_cls,
            name="llama_index.sentence",
            parser_name="SentenceSplitterNodeParser",
            token_counter=token_counter,
        )
