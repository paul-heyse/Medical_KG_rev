"""Chunker implementation that mimics LlamaIndex sentence window parsing."""

from __future__ import annotations

from typing import Any

from Medical_KG_rev.models.ir import Document

from ..port import Chunk
from .base import BaseProfileChunker


def _try_import_llamaindex():  # pragma: no cover - optional dependency
    try:
        from llama_index.core import Document as LlamaDocument  # type: ignore
        from llama_index.core.node_parser import SentenceWindowNodeParser  # type: ignore
    except ImportError:
        return None, None
    return SentenceWindowNodeParser, LlamaDocument


class LlamaIndexChunker(BaseProfileChunker):
    """Profile-aware chunker leveraging sentence window semantics."""

    name = "llamaindex_sentence_window"

    def __init__(self, *, profile: dict[str, Any]) -> None:
        super().__init__(profile=profile)
        parser_cls, document_cls = _try_import_llamaindex()
        metadata = profile.get("metadata", {})
        self._window_size = metadata.get("window_size", 3)
        self._parser = None
        self._document_cls = document_cls
        if parser_cls is not None and document_cls is not None:
            self._parser = parser_cls(
                window_size=self._window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_sentence",
            )
        self._sentence_split = self._sentence_separator()

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:
        if self._parser is None or self._document_cls is None:
            return self._fallback_chunk(document)
        return self._llamaindex_chunk(document)

    def _llamaindex_chunk(self, document: Document) -> list[Chunk]:
        filtered_document, groups = self._prepare_groups(document)
        documents: list[tuple[int, Any]] = []
        for index, group in enumerate(groups):
            combined = "\n\n".join(ctx.text for ctx in group if ctx.text)
            if not combined:
                continue
            metadata = {"group_index": index}
            documents.append((index, self._document_cls(text=combined, metadata=metadata)))  # type: ignore[call-arg]
        if not documents:
            return []
        nodes = self._parser.get_nodes_from_documents([doc for _, doc in documents])  # type: ignore[operator]
        chunk_texts: list[str] = []
        chunk_to_group: list[int] = []
        for node in nodes:
            text = getattr(node, "text", None)
            if not text:
                continue
            origin = getattr(node, "metadata", {}).get("group_index")
            if origin is None:
                origin = 0
            chunk_texts.append(text)
            chunk_to_group.append(int(origin))
        return self._assemble(
            document=filtered_document,
            groups=groups,
            chunk_texts=chunk_texts,
            chunk_to_group_index=chunk_to_group,
        )

    def _fallback_chunk(self, document: Document) -> list[Chunk]:
        filtered_document, groups = self._prepare_groups(document)
        chunk_texts: list[str] = []
        chunk_to_group: list[int] = []
        for index, group in enumerate(groups):
            sentences: list[str] = []
            for ctx in group:
                if ctx.text:
                    for _, _, sentence in self._sentence_split(ctx.text):
                        sentences.append(sentence)
            if not sentences:
                continue
            window = self._window_size
            if window <= 0:
                window = 1
            for start in range(0, len(sentences)):
                window_sentences = sentences[start : start + window]
                if not window_sentences:
                    break
                chunk_texts.append(" ".join(window_sentences))
                chunk_to_group.append(index)
        return self._assemble(
            document=filtered_document,
            groups=groups,
            chunk_texts=chunk_texts,
            chunk_to_group_index=chunk_to_group,
        )


def register() -> None:
    from ..port import register_chunker

    register_chunker(LlamaIndexChunker.name, lambda *, profile: LlamaIndexChunker(profile=profile))


__all__ = ["LlamaIndexChunker", "register"]
