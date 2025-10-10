"""Base building blocks for chunker wrappers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from Medical_KG_rev.models.ir import Document

from ..port import Chunk, ChunkerPort
from ..runtime import assemble_chunks, default_intent_provider, group_contexts, iter_block_contexts


class BaseProfileChunker(ChunkerPort):
    """Common utilities shared by chunker implementations."""

    def __init__(self, *, profile: dict[str, Any]) -> None:
        self.profile = profile
        self.profile_name = profile["name"]
        self.respect_boundaries: Sequence[str] = profile.get("respect_boundaries", [])
        self._filters: Sequence[str] = profile.get("filters", [])
        metadata = profile.get("metadata", {})
        self._chunker_version: str = metadata.get("chunker_version", self.__class__.__name__)

    def _prepare_groups(self, document: Document):
        filtered_document = self._apply_filters(document)
        contexts = list(iter_block_contexts(filtered_document))
        groups = group_contexts(contexts, respect_boundaries=self.respect_boundaries)
        return filtered_document, groups

    def _apply_filters(self, document: Document) -> Document:
        if not self._filters:
            return document
        from ..filters import apply_filter_chain

        return apply_filter_chain(document, self._filters)

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
            metadata_provider=self._metadata_provider(document),
        )

    def _intent_hint_for_section(self, section) -> str | None:
        metadata = self.profile.get("metadata", {})
        intent_map: dict[str, str] = metadata.get("intent_hints", {})
        if section is None:
            return None
        if section.title and section.title in intent_map:
            return intent_map[section.title]
        return default_intent_provider(section)

    def _sentence_separator(self) -> Callable[[str], list[tuple[int, int, str]]]:
        from ..sentence_splitters import get_sentence_splitter

        splitter_name = self.profile.get("sentence_splitter", "syntok")
        return get_sentence_splitter(splitter_name)

    def chunk(
        self, document: Document, *, profile: str
    ) -> list[Chunk]:  # pragma: no cover - defined in subclasses
        raise NotImplementedError

    def _metadata_provider(self, document: Document):
        source_system = document.source

        def provider(contexts):
            metadata: dict[str, str] = {
                "source_system": source_system,
                "chunker_version": self._chunker_version,
            }
            if contexts:
                section = contexts[0].section
                if section and section.title:
                    metadata.setdefault("section_title", section.title)
            return metadata

        return provider
