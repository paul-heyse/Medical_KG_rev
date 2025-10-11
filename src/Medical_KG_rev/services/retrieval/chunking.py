"""Simplified chunking service used by the torch-free gateway."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from Medical_KG_rev.chunking.models import Chunk


@dataclass(slots=True)
class ChunkCommand:
    """Minimal command object for the legacy chunking service."""

    tenant_id: str
    doc_id: str
    text: str


class ChunkingService:
    """Simple paragraph splitter used as a stop-gap implementation."""

    def chunk(self, command: ChunkCommand) -> List[Chunk]:
        raise RuntimeError(
            "Fallback chunking service is disabled. Use the standard chunking pipeline."
        )
