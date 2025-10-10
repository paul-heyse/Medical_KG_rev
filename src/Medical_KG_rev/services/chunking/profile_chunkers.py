"""Domain-specific chunker implementations backed by profile metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from Medical_KG_rev.models.ir import BlockType, Document, Section

from .port import Chunk
from .runtime import build_chunk, iter_block_contexts
from .wrappers.base import BaseProfileChunker


class _ContextCache:
    """Helper structure to reuse block contexts efficiently."""

    def __init__(self, document: Document) -> None:
        self._contexts = list(iter_block_contexts(document))
        self._by_section: dict[str, list] = defaultdict(list)
        self._by_block: dict[str, list] = {}
        for ctx in self._contexts:
            self._by_section[ctx.section.id].append(ctx)
            self._by_block.setdefault(ctx.block.id, []).append(ctx)

    def section(self, section: Section) -> list:
        return list(self._by_section.get(section.id, []))

    def block(self, block_id: str) -> list:
        return list(self._by_block.get(block_id, []))


class CTGovRegistryChunker(BaseProfileChunker):
    """Chunker tailored for CT.gov registry records."""

    name = "ctgov_registry"

    _SECTION_UNITS = {
        "Eligibility Criteria": "eligibility",
        "Outcome Measures": "outcome",
        "Adverse Events": "ae",
        "Results": "results",
    }

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:
        filtered = self._apply_filters(document)
        cache = _ContextCache(filtered)
        groups: list[list] = []
        texts: list[str] = []
        metadata: list[dict[str, Any]] = []

        for section in filtered.sections:
            title = (section.title or "").strip()
            if not title:
                continue
            if title == "Eligibility Criteria":
                contexts = cache.section(section)
                text = self._render_group_text(contexts)
                if text:
                    groups.append(contexts)
                    texts.append(text)
                    metadata.append(self._build_metadata(section, contexts))
                continue
            if title == "Outcome Measures":
                for block in section.blocks:
                    contexts = cache.block(block.id)
                    text = self._render_group_text(contexts)
                    if not text:
                        continue
                    groups.append(contexts)
                    metadata.append(self._build_metadata(section, contexts, block_metadata=True))
                    texts.append(text)
                continue
            if title == "Adverse Events":
                for block in section.blocks:
                    if block.type != BlockType.TABLE:
                        continue
                    contexts = cache.block(block.id)
                    text = self._render_group_text(contexts)
                    if not text:
                        continue
                    groups.append(contexts)
                    metadata.append(self._build_metadata(section, contexts, table=True))
                    texts.append(text)
                continue
            if title == "Results":
                for block in section.blocks:
                    contexts = cache.block(block.id)
                    text = self._render_group_text(contexts)
                    if not text:
                        continue
                    groups.append(contexts)
                    metadata.append(self._build_metadata(section, contexts))
                    texts.append(text)

        return self._finalize_chunks(filtered, profile, groups, texts, metadata)

    def _finalize_chunks(
        self,
        document: Document,
        profile: str,
        groups: Sequence[Sequence],
        texts: Sequence[str],
        metadata: Sequence[dict[str, Any]],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for contexts, text, extra in zip(groups, texts, metadata, strict=False):
            if not text:
                continue
            chunk = _build_chunk_from_contexts(
                document=document,
                profile_name=profile,
                contexts=contexts,
                text=text,
                intent_hint=self._intent_hint_for_section(
                    contexts[0].section if contexts else None
                ),
                extra_metadata=extra,
            )
            chunks.append(chunk)
        return chunks

    def _build_metadata(
        self,
        section: Section,
        contexts: Sequence,
        block_metadata: bool = False,
        table: bool = False,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "chunker_version": self.__class__.__name__,
            "registry_unit": self._SECTION_UNITS.get((section.title or "").strip(), "other"),
        }
        if contexts:
            block = contexts[0].block
            block_meta = getattr(block, "metadata", {})
            if isinstance(block_meta, dict):
                if block_metadata:
                    for key in ("title", "time_frame", "measure_type"):
                        if block_meta.get(key):
                            metadata[f"ctgov_{key}"] = block_meta[key]
                if table and "html" in block_meta and block_meta["html"]:
                    metadata.setdefault("table_html", block_meta["html"])
        return metadata

    def _render_group_text(self, contexts: Iterable) -> str:
        parts: list[str] = []
        for ctx in contexts:
            block = ctx.block
            if block.type == BlockType.TABLE and isinstance(block.metadata, dict):
                html = block.metadata.get("html")
                if html:
                    parts.append(html)
                    continue
            if ctx.text:
                parts.append(ctx.text.strip())
        return "\n\n".join(part for part in parts if part).strip()


class SPLLabelChunker(BaseProfileChunker):
    """Chunker for Structured Product Label (SPL) documents."""

    name = "spl_label"

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:
        filtered = self._apply_filters(document)
        cache = _ContextCache(filtered)
        groups: list[list] = []
        texts: list[str] = []
        section_info: list[tuple[Section, dict[str, Any]]] = []

        for section in filtered.sections:
            contexts = cache.section(section)
            text = self._render_group_text(contexts)
            if not text:
                continue
            loinc_code = self._resolve_loinc(section, contexts)
            groups.append(contexts)
            texts.append(text)
            section_info.append(
                (
                    section,
                    {
                        "loinc_code": loinc_code,
                        "chunker_version": self.__class__.__name__,
                    },
                )
            )

        chunks = []
        for contexts, text, (section, extras) in zip(groups, texts, section_info, strict=False):
            chunk = _build_chunk_from_contexts(
                document=filtered,
                profile_name=profile,
                contexts=contexts,
                text=text,
                intent_hint=self._intent_hint_for_section(section),
                extra_metadata=extras,
            )
            label = self._format_section_label(section, extras.get("loinc_code"))
            chunk.section_label = label
            chunk.metadata.setdefault("loinc_code", extras.get("loinc_code"))
            chunks.append(chunk)
        return chunks

    def _render_group_text(self, contexts: Iterable) -> str:
        parts: list[str] = []
        for ctx in contexts:
            if ctx.text:
                parts.append(ctx.text.strip())
        return "\n\n".join(part for part in parts if part).strip()

    def _resolve_loinc(self, section: Section, contexts: Sequence) -> str | None:
        metadata = self.profile.get("metadata", {})
        loinc_map = metadata.get("loinc_map", {})
        title = (section.title or "").strip()
        if title in loinc_map:
            return loinc_map[title]
        for ctx in contexts:
            block_meta = getattr(ctx.block, "metadata", {})
            if isinstance(block_meta, dict):
                code = block_meta.get("loinc_code")
                if code:
                    return code
        if "LOINC:" in title:
            return title.split("LOINC:", 1)[1].split()[0].strip()
        return None

    def _format_section_label(self, section: Section, loinc_code: str | None) -> str:
        base_title = (section.title or "").strip()
        if loinc_code and not base_title.startswith("LOINC:"):
            return f"LOINC:{loinc_code} {base_title}".strip()
        return base_title


class GuidelineChunker(BaseProfileChunker):
    """Chunker tuned for clinical guideline recommendations."""

    name = "guideline_recommendation"

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:
        filtered = self._apply_filters(document)
        cache = _ContextCache(filtered)
        groups: list[list] = []
        texts: list[str] = []
        metadata: list[dict[str, Any]] = []

        for section in filtered.sections:
            title = (section.title or "").strip()
            if title.lower() == "recommendations":
                for block in section.blocks:
                    contexts = cache.block(block.id)
                    text = self._render_group_text(contexts)
                    if not text:
                        continue
                    groups.append(contexts)
                    texts.append(text)
                    metadata.append(self._recommendation_metadata(contexts))
                continue
            if title.lower() == "evidence summary":
                for block in section.blocks:
                    contexts = cache.block(block.id)
                    text = self._render_group_text(contexts)
                    if not text:
                        continue
                    groups.append(contexts)
                    texts.append(text)
                    metadata.append(self._evidence_metadata(contexts))
                continue

        chunks: list[Chunk] = []
        for contexts, text, extra in zip(groups, texts, metadata, strict=False):
            chunk = _build_chunk_from_contexts(
                document=filtered,
                profile_name=profile,
                contexts=contexts,
                text=text,
                intent_hint=self._intent_hint_for_section(
                    contexts[0].section if contexts else None
                ),
                extra_metadata=extra,
            )
            chunks.append(chunk)
        return chunks

    def _render_group_text(self, contexts: Iterable) -> str:
        parts: list[str] = []
        for ctx in contexts:
            block = ctx.block
            if block.type == BlockType.TABLE and isinstance(block.metadata, dict):
                html = block.metadata.get("html")
                if html:
                    parts.append(html)
                    continue
            if ctx.text:
                parts.append(ctx.text.strip())
        return "\n\n".join(part for part in parts if part).strip()

    def _recommendation_metadata(self, contexts: Sequence) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "guideline_unit": "recommendation",
            "chunker_version": self.__class__.__name__,
        }
        if contexts:
            block_meta = getattr(contexts[0].block, "metadata", {})
            if isinstance(block_meta, dict):
                for key in ("recommendation_id", "strength", "certainty"):
                    if block_meta.get(key):
                        metadata[key] = block_meta[key]
        return metadata

    def _evidence_metadata(self, contexts: Sequence) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "guideline_unit": "evidence",
            "chunker_version": self.__class__.__name__,
        }
        if contexts and contexts[0].block.type == BlockType.TABLE:
            block_meta = getattr(contexts[0].block, "metadata", {})
            if isinstance(block_meta, dict) and block_meta.get("html"):
                metadata["table_html"] = block_meta["html"]
        return metadata


def _build_chunk_from_contexts(
    *,
    document: Document,
    profile_name: str,
    contexts: Sequence,
    text: str,
    intent_hint: str | None,
    extra_metadata: dict[str, Any] | None = None,
) -> Chunk:
    mapping: list[int | None] = []
    assembled: list[str] = []
    for ctx in contexts:
        if ctx.text:
            assembled.append(ctx.text)
            mapping.extend(range(ctx.start, ctx.end))
            assembled.append("\n\n")
            mapping.append(None)
    assembled_text = "".join(assembled)
    start_index = assembled_text.find(text)
    if start_index == -1:
        start_index = 0
    end_index = start_index + len(text)
    mapping_slice = [offset for offset in mapping[start_index:end_index] if offset is not None]
    metadata: dict[str, Any] = {}
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata.setdefault("chunker_version", "profile_chunker")
    return build_chunk(
        document=document,
        profile_name=profile_name,
        text=text,
        mapping=mapping_slice,
        section=contexts[0].section if contexts else None,
        intent_hint=intent_hint,
        metadata=metadata,
    )


def register() -> None:
    from .port import register_chunker

    register_chunker(
        CTGovRegistryChunker.name, lambda *, profile: CTGovRegistryChunker(profile=profile)
    )
    register_chunker(SPLLabelChunker.name, lambda *, profile: SPLLabelChunker(profile=profile))
    register_chunker(GuidelineChunker.name, lambda *, profile: GuidelineChunker(profile=profile))
