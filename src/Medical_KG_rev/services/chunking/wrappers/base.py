"""Base building blocks for chunker wrappers."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from Medical_KG_rev.models.ir import Document

from ..port import Chunk, ChunkerPort
from ..runtime import assemble_chunks, group_contexts, iter_block_contexts, default_intent_provider


class BaseProfileChunker(ChunkerPort):
    """Common utilities shared by chunker implementations."""

    def __init__(self, *, profile: dict[str, Any]) -> None:
        self.profile = profile
        self.profile_name = profile["name"]
        self.respect_boundaries: Sequence[str] = profile.get("respect_boundaries", [])

    def _prepare_groups(self, document: Document):
        contexts = list(iter_block_contexts(document))
        return group_contexts(contexts, respect_boundaries=self.respect_boundaries)

    def _assemble(
        self,
        *,
        document: Document,
        groups,
        chunk_texts,
        chunk_to_group_index,
    ) -> list[Chunk]:
        return assemble_chunks(
            document=document,
            profile_name=self.profile_name,
            groups=groups,
            chunk_texts=chunk_texts,
            chunk_to_group_index=chunk_to_group_index,
            intent_hint_provider=self._intent_hint_for_section,
        )

    def _intent_hint_for_section(self, section) -> str | None:
        metadata = self.profile.get("metadata", {})
        intent_map: dict[str, str] = metadata.get("intent_hints", {})
        if section is None:
            return None
        if section.title and section.title in intent_map:
            return intent_map[section.title]
        return default_intent_provider(section)

    def _sentence_separator(self) -> Callable[[str], list[str]]:
        from ..sentence_splitters import get_sentence_splitter

        splitter_name = self.profile.get("sentence_splitter", "syntok")
        return get_sentence_splitter(splitter_name)

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:  # pragma: no cover - defined in subclasses
        raise NotImplementedError
