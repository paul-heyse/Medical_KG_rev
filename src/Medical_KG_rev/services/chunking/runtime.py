"""Runtime helpers for profile-based chunking."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section

from .port import Chunk


@dataclass
class _BlockContext:
    block: Block
    section: Section
    text: str
    start: int
    end: int


def iter_block_contexts(document: Document) -> Iterable[_BlockContext]:
    """Yield block contexts with absolute offsets within the document."""

    cursor = 0
    for section in document.sections:
        for block in section.blocks:
            text = block.text or ""
            start = cursor
            end = start + len(text)
            cursor = end
            yield _BlockContext(
                block=block,
                section=section,
                text=text,
                start=start,
                end=end,
            )


def group_contexts(
    contexts: Iterable[_BlockContext],
    *,
    respect_boundaries: Sequence[str],
) -> list[list[_BlockContext]]:
    """Group contexts based on the requested boundary hints."""

    groups: list[list[_BlockContext]] = []
    current: list[_BlockContext] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    boundaries = set(respect_boundaries)
    for ctx in contexts:
        if "section" in boundaries and current:
            if ctx.section is not current[-1].section:
                flush()
        if "table" in boundaries and ctx.block.type == BlockType.TABLE:
            flush()
            groups.append([ctx])
            continue
        current.append(ctx)
    flush()
    return groups


def build_chunk(
    *,
    document: Document,
    profile_name: str,
    text: str,
    mapping: list[int | None],
    section: Section | None,
    intent_hint: str | None,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Create a :class:`Chunk` from assembled text and mapping."""

    doc_offsets = [offset for offset in mapping if offset is not None]
    if not doc_offsets:
        start = end = 0
    else:
        start = doc_offsets[0]
        end = doc_offsets[-1] + 1
    return Chunk(
        chunk_id=f"{document.id}:{uuid.uuid4().hex}",
        doc_id=document.id,
        text=text,
        char_offsets=(start, end),
        section_label=section.title if section else None,
        intent_hint=intent_hint,
        metadata=metadata or {"profile": profile_name},
    )


def assemble_chunks(
    *,
    document: Document,
    profile_name: str,
    groups: Sequence[list[_BlockContext]],
    chunk_texts: Sequence[str],
    chunk_to_group_index: Sequence[int],
    intent_hint_provider: Callable[[Section | None], str | None],
) -> list[Chunk]:
    """Materialize chunks based on generated text pieces."""

    chunks: list[Chunk] = []
    for idx, text in enumerate(chunk_texts):
        group_idx = chunk_to_group_index[idx]
        contexts = groups[group_idx]
        mapping: list[int | None] = []
        assembled_chars: list[str] = []
        for ctx in contexts:
            if not ctx.text:
                continue
            assembled_chars.append(ctx.text)
            mapping.extend(range(ctx.start, ctx.end))
            assembled_chars.append("\n\n")
            mapping.append(None)
        assembled = "".join(assembled_chars)
        # align chunk text inside assembled text
        start_index = assembled.find(text)
        if start_index == -1:
            start_index = 0
        end_index = start_index + len(text)
        mapping_slice = mapping[start_index:end_index]
        section = contexts[0].section if contexts else None
        chunks.append(
            build_chunk(
                document=document,
                profile_name=profile_name,
                text=text,
                mapping=mapping_slice,
                section=section,
                intent_hint=intent_hint_provider(section),
            )
        )
    return chunks


def identity_intent_provider(section: Section | None) -> str | None:
    return None if section is None else section.metadata.get("intent")  # type: ignore[return-value]


def default_intent_provider(section: Section | None) -> str | None:
    if section is None:
        return None
    return section.metadata.get("intent_hint") if isinstance(section.metadata, dict) else None
