"""Protocol definitions for chunkers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from Medical_KG_rev.models.ir import Block, Document

from .models import Chunk, Granularity


class BaseChunker(Protocol):
    """Protocol all chunkers must follow."""

    name: str
    version: str

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Sequence[Block] | None = None,
    ) -> list[Chunk]:
        """Split the document into coherent chunks."""

    def explain(self) -> dict[str, Any]:
        """Return configuration useful for debugging or evaluation."""


class SupportsSentenceSplitting(Protocol):
    """Protocol for sentence splitting adapters."""

    def split(self, text: str) -> Iterable[str]:
        ...
