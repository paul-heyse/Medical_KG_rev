"""Domain-specific chunker implementations backed by profile metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from Medical_KG_rev.models.ir import BlockType, Document, Section

from .port import Chunk
from .port import register_chunker
from .runtime import build_chunk, iter_block_contexts
from .wrappers.base import BaseProfileChunker


register_chunker(
CTGovRegistryChunker.name, lambda *, profile: CTGovRegistryChunker(profile=profile)
)
register_chunker(SPLLabelChunker.name, lambda *, profile: SPLLabelChunker(profile=profile))
register_chunker(GuidelineChunker.name, lambda *, profile: GuidelineChunker(profile=profile))
