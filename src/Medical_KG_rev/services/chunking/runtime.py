"""Runtime helpers for profile-based chunking.

This module provides runtime utilities for processing documents through
profile-based chunking systems. It handles document parsing, block
context management, chunk assembly, and metadata extraction.

Key Components:
    - _BlockContext: Internal data structure for block processing
    - iter_block_contexts: Document parser with absolute offsets
    - group_contexts: Boundary-aware context grouping
    - build_chunk: Chunk construction from assembled text
    - assemble_chunks: Batch chunk materialization
    - Intent providers: Metadata extraction utilities

Responsibilities:
    - Parse documents into block contexts with absolute offsets
    - Group contexts respecting section and table boundaries
    - Assemble chunks from grouped contexts with proper metadata
    - Extract intent hints and metadata from document sections
    - Generate unique chunk IDs and timestamps

Collaborators:
    - Document models (Block, Section, Document)
    - Chunk port interface
    - Profile-based chunking systems

Side Effects:
    - Generates UUIDs for chunk identification
    - Creates timestamps for chunk metadata
    - Modifies document text during assembly

Thread Safety:
    - Thread-safe: All operations are stateless
    - No shared mutable state between operations

Performance Characteristics:
    - Document parsing is O(n) where n is document length
    - Context grouping respects boundaries efficiently
    - Chunk assembly involves string operations and mapping

Example:
    >>> document = Document(id="doc1", sections=[...])
    >>> contexts = list(iter_block_contexts(document))
    >>> groups = group_contexts(contexts, respect_boundaries=["section"])
    >>> chunks = assemble_chunks(
    ...     document=document,
    ...     profile_name="medical",
    ...     groups=groups,
    ...     chunk_texts=["text1", "text2"],
    ...     chunk_to_group_index=[0, 1],
    ...     intent_hint_provider=default_intent_provider
    ... )
    >>> print(f"Generated {len(chunks)} chunks")
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section

from .port import Chunk

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class _BlockContext:
    """Internal data structure for block processing with absolute offsets.

    Represents a block within its document context, including absolute
    character offsets for proper chunk assembly and mapping.

    Attributes:
        block: The block being processed
        section: The section containing the block
        text: The text content of the block
        start: Absolute start offset within the document
        end: Absolute end offset within the document

    Invariants:
        - start <= end
        - text length equals (end - start)
        - block and section are never None

    Example:
        >>> context = _BlockContext(
        ...     block=block, section=section, text="Hello world",
        ...     start=100, end=111
        ... )
        >>> print(f"Text: {context.text}, Range: {context.start}-{context.end}")
    """
    block: Block
    section: Section
    text: str
    start: int
    end: int


# ==============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ==============================================================================

def iter_block_contexts(document: Document) -> Iterable[_BlockContext]:
    """Yield block contexts with absolute offsets within the document.

    Args:
        document: Document to process into block contexts

    Yields:
        _BlockContext instances with absolute character offsets

    Note:
        This function maintains a running cursor to calculate absolute
        offsets across all sections and blocks in the document.

    Example:
        >>> document = Document(id="doc1", sections=[...])
        >>> for context in iter_block_contexts(document):
        ...     print(f"Block: {context.text[:50]}... at {context.start}-{context.end}")
    """
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
    """Group contexts based on the requested boundary hints.

    Args:
        contexts: Iterable of block contexts to group
        respect_boundaries: Sequence of boundary types to respect
            (e.g., "section", "table")

    Returns:
        List of context groups, where each group respects the specified
        boundaries

    Note:
        This function groups contexts while respecting section and table
        boundaries. Tables are always isolated into their own groups.

    Example:
        >>> contexts = list(iter_block_contexts(document))
        >>> groups = group_contexts(
        ...     contexts, respect_boundaries=["section", "table"]
        ... )
        >>> print(f"Created {len(groups)} groups")
    """
    groups: list[list[_BlockContext]] = []
    current: list[_BlockContext] = []

    def flush() -> None:
        """Flush current group to groups list and reset current."""
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


# ==============================================================================
# CHUNK CONSTRUCTION FUNCTIONS
# ==============================================================================

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
    """Create a :class:`Chunk` from assembled text and mapping.

    Args:
        document: Source document for the chunk
        profile_name: Name of the chunking profile used
        text: The text content of the chunk
        mapping: List of character offsets mapping to document positions
        section: Optional section containing the chunk
        intent_hint: Optional intent hint for the chunk
        metadata: Optional additional metadata

    Returns:
        Constructed Chunk instance with proper metadata and offsets

    Note:
        This function calculates character offsets from the mapping,
        generates a unique chunk ID, and sets up metadata including
        timestamps and source information.

    Example:
        >>> chunk = build_chunk(
        ...     document=doc,
        ...     profile_name="medical",
        ...     text="Sample text",
        ...     mapping=[100, 101, 102, 103],
        ...     section=section,
        ...     intent_hint="diagnosis"
        ... )
        >>> print(f"Chunk ID: {chunk.chunk_id}")
    """
    doc_offsets = [offset for offset in mapping if offset is not None]
    if not doc_offsets:
        start = end = 0
    else:
        start = doc_offsets[0]
        end = doc_offsets[-1] + 1
    section_label = ""
    if section and section.title:
        section_label = section.title.strip()
    resolved_metadata: dict[str, Any] = {"chunking_profile": profile_name}
    if metadata:
        resolved_metadata.update(metadata)
    resolved_metadata.setdefault("source_system", document.source)
    resolved_metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    resolved_metadata.setdefault("chunker_version", "unknown")
    intent_value = intent_hint or ""
    section_metadata = getattr(section, "metadata", None)
    if isinstance(section_metadata, dict):
        resolved_metadata.setdefault("section_metadata", section_metadata)
    return Chunk(
        chunk_id=f"{document.id}:{uuid.uuid4().hex}",
        doc_id=document.id,
        text=text,
        char_offsets=(start, end),
        section_label=section_label,
        intent_hint=intent_value,
        metadata=resolved_metadata,
    )


def assemble_chunks(
    *,
    document: Document,
    profile_name: str,
    groups: Sequence[list[_BlockContext]],
    chunk_texts: Sequence[str],
    chunk_to_group_index: Sequence[int],
    intent_hint_provider: Callable[[Section | None], str | None],
    metadata_provider: Callable[[Sequence[_BlockContext]], dict[str, Any]] | None = None,
) -> list[Chunk]:
    """Materialize chunks based on generated text pieces.

    Args:
        document: Source document for chunks
        profile_name: Name of the chunking profile used
        groups: Sequence of context groups
        chunk_texts: Sequence of chunk text content
        chunk_to_group_index: Mapping from chunk index to group index
        intent_hint_provider: Function to extract intent hints from sections
        metadata_provider: Optional function to extract metadata from contexts

    Returns:
        List of constructed Chunk instances

    Note:
        This function assembles chunks by aligning chunk texts with
        their corresponding context groups and building proper character
        offset mappings.

    Example:
        >>> chunks = assemble_chunks(
        ...     document=doc,
        ...     profile_name="medical",
        ...     groups=groups,
        ...     chunk_texts=["text1", "text2"],
        ...     chunk_to_group_index=[0, 1],
        ...     intent_hint_provider=default_intent_provider
        ... )
        >>> print(f"Assembled {len(chunks)} chunks")
    """
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
        metadata = metadata_provider(contexts) if metadata_provider else None
        chunks.append(
            build_chunk(
                document=document,
                profile_name=profile_name,
                text=text,
                mapping=mapping_slice,
                section=section,
                intent_hint=intent_hint_provider(section),
                metadata=metadata,
            )
        )
    return chunks


# ==============================================================================
# INTENT PROVIDER FUNCTIONS
# ==============================================================================

def identity_intent_provider(section: Section | None) -> str | None:
    """Extract intent from section metadata using 'intent' key.

    Args:
        section: Section to extract intent from

    Returns:
        Intent string if found, None otherwise

    Note:
        This provider looks for the 'intent' key in section metadata.

    Example:
        >>> intent = identity_intent_provider(section)
        >>> print(f"Intent: {intent}")
    """
    if section is None:
        return None
    metadata = getattr(section, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get("intent")
    return None


def default_intent_provider(section: Section | None) -> str | None:
    """Extract intent from section metadata using 'intent_hint' key.

    Args:
        section: Section to extract intent from

    Returns:
        Intent hint string if found, None otherwise

    Note:
        This provider looks for the 'intent_hint' key in section metadata.
        This is the default provider used in most chunking operations.

    Example:
        >>> intent_hint = default_intent_provider(section)
        >>> print(f"Intent hint: {intent_hint}")
    """
    if section is None:
        return None
    metadata = getattr(section, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get("intent_hint")
    return None
