"""Lightweight fallback chunker used in tests."""

from __future__ import annotations

from typing import Any, List

from Medical_KG_rev.models.ir import Document

from ..port import Chunk
from .base import BaseProfileChunker


class SimpleChunker(BaseProfileChunker):
    """Chunker that splits on sentences using the configured splitter."""

    name = "simple"

    def __init__(self, *, profile: dict[str, Any]) -> None:
        super().__init__(profile=profile)
        self._sentence_split = self._sentence_separator()
        self._target_tokens = profile.get("target_tokens", 256)
        self._overlap_tokens = profile.get("overlap_tokens", 0)

    def chunk(self, document: Document, *, profile: str) -> List[Chunk]:
        filtered_document, groups = self._prepare_groups(document)
        chunk_texts: List[str] = []
        chunk_to_group: List[int] = []
        for index, group in enumerate(groups):
            sentences: list[str] = []
            for ctx in group:
                if ctx.text:
                    for _, _, sentence in self._sentence_split(ctx.text):
                        sentences.append(sentence)
            if not sentences:
                continue
            current: list[str] = []
            for sentence in sentences:
                candidate = " ".join((*current, sentence)).strip()
                if len(candidate.split()) > self._target_tokens and current:
                    chunk_texts.append(" ".join(current))
                    chunk_to_group.append(index)
                    if self._overlap_tokens and current:
                        overlap = " ".join(current[-1:])
                        current = [overlap, sentence]
                    else:
                        current = [sentence]
                else:
                    current.append(sentence)
            if current:
                chunk_texts.append(" ".join(current))
                chunk_to_group.append(index)
        return self._assemble(
            document=filtered_document,
            groups=groups,
            chunk_texts=chunk_texts,
            chunk_to_group_index=chunk_to_group,
        )


def register() -> None:
    from ..port import register_chunker

    register_chunker(SimpleChunker.name, lambda *, profile: SimpleChunker(profile=profile))
