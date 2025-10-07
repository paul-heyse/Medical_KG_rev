"""Chunking aware ingestion helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import structlog

from Medical_KG_rev.chunking import Chunk, ChunkingOptions, ChunkingService
from Medical_KG_rev.models.ir import Document
from Medical_KG_rev.observability.metrics import (
    observe_chunking_latency,
    record_chunk_size,
)

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ChunkingRun:
    """Summary returned for each ingestion chunking execution."""

    document_id: str
    profile: str
    duration_seconds: float
    chunks: Sequence[Chunk]
    granularity_counts: Mapping[str, int]


class ChunkStorage:
    """Protocol-like base class for chunk storage backends."""

    def store(self, tenant_id: str, document_id: str, chunks: Sequence[Chunk]) -> None:
        raise NotImplementedError

    def list(self, tenant_id: str, document_id: str) -> list[Chunk]:
        raise NotImplementedError


class InMemoryChunkStorage(ChunkStorage):
    """Lightweight in-memory storage used for tests and local execution."""

    def __init__(self) -> None:
        self._storage: dict[tuple[str, str], list[Chunk]] = {}

    def store(self, tenant_id: str, document_id: str, chunks: Sequence[Chunk]) -> None:
        key = (tenant_id, document_id)
        existing = self._storage.setdefault(key, [])
        existing.extend(chunks)

    def list(self, tenant_id: str, document_id: str) -> list[Chunk]:
        return list(self._storage.get((tenant_id, document_id), []))


class IngestionService:
    """Coordinates chunking during ingestion pipelines."""

    def __init__(
        self,
        *,
        chunking_service: ChunkingService | None = None,
        storage: ChunkStorage | None = None,
    ) -> None:
        self.chunking = chunking_service or ChunkingService()
        self.storage = storage or InMemoryChunkStorage()

    def detect_profile(self, document: Document, source_hint: str | None) -> str:
        return source_hint or document.source or self.chunking.config.default_profile

    def chunk_document(
        self,
        document: Document,
        *,
        tenant_id: str,
        source_hint: str | None = None,
        options: ChunkingOptions | None = None,
    ) -> ChunkingRun:
        profile = self.detect_profile(document, source_hint)
        started = perf_counter()
        chunks = list(
            self.chunking.chunk_document(
                document,
                tenant_id=tenant_id,
                source=profile,
                options=options,
            )
        )
        duration = perf_counter() - started
        self._ensure_chunk_ids(document.id, chunks)
        self.storage.store(tenant_id, document.id, chunks)
        counts = defaultdict(int)
        for chunk in chunks:
            counts[chunk.granularity] += 1
            record_chunk_size(profile, chunk.granularity, len(chunk.body))
        observe_chunking_latency(profile, duration)
        logger.info(
            "ingestion.chunked",
            document_id=document.id,
            tenant_id=tenant_id,
            profile=profile,
            chunks=len(chunks),
            duration=round(duration, 4),
        )
        return ChunkingRun(
            document_id=document.id,
            profile=profile,
            duration_seconds=duration,
            chunks=chunks,
            granularity_counts=dict(counts),
        )

    def list_chunks(self, tenant_id: str, document_id: str) -> list[Chunk]:
        return self.storage.list(tenant_id, document_id)

    def _ensure_chunk_ids(self, document_id: str, chunks: list[Chunk]) -> None:
        for index, chunk in enumerate(chunks):
            if chunk.chunk_id.startswith(f"{document_id}:"):
                continue
            updated = chunk.model_copy(update={"chunk_id": f"{document_id}:{chunk.chunker}:{chunk.granularity}:{index}"})
            chunks[index] = updated  # type: ignore[index]

