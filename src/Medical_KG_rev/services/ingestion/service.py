"""Chunking aware ingestion helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import structlog

from Medical_KG_rev.chunking import Chunk, ChunkingOptions, ChunkingService
from Medical_KG_rev.chunking.exceptions import (
    ChunkerConfigurationError,
    ChunkingFailedError,
    ChunkingUnavailableError,
    InvalidDocumentError,
    ProfileNotFoundError,
    TokenizerMismatchError,
)
from Medical_KG_rev.models.ir import Document
from Medical_KG_rev.observability.metrics import (
    observe_chunking_latency,
    record_chunk_size,
    record_chunking_document,
    record_chunking_failure,
)
from Medical_KG_rev.services.chunking.events import ChunkingEventEmitter

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ChunkingRun:
    """Summary returned for each ingestion chunking execution."""

    document_id: str
    profile: str
    duration_seconds: float
    chunks: Sequence[Chunk]
    granularity_counts: Mapping[str, int]
    average_chars: float
    average_tokens: float
    section_distribution: Mapping[str, int]
    intent_distribution: Mapping[str, int]


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
        events: ChunkingEventEmitter | None = None,
    ) -> None:
        self.chunking = chunking_service or ChunkingService()
        self.storage = storage or InMemoryChunkStorage()
        self.events = events or ChunkingEventEmitter()

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
        self.events.emit_started(
            tenant_id=tenant_id,
            document_id=document.id,
            profile=profile,
            correlation_id=None,
            source=source_hint,
        )
        started = perf_counter()
        try:
            chunks = list(
                self.chunking.chunk_document(
                    document,
                    tenant_id=tenant_id,
                    source=profile,
                    options=options,
                )
            )
        except (
            ProfileNotFoundError,
            TokenizerMismatchError,
            ChunkingFailedError,
            ChunkerConfigurationError,
            InvalidDocumentError,
            ChunkingUnavailableError,
        ) as exc:
            record_chunking_failure(profile, exc.__class__.__name__)
            self.events.emit_failed(
                tenant_id=tenant_id,
                document_id=document.id,
                profile=profile,
                correlation_id=None,
                error_type=exc.__class__.__name__,
                message=str(exc),
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            record_chunking_failure(profile, exc.__class__.__name__)
            self.events.emit_failed(
                tenant_id=tenant_id,
                document_id=document.id,
                profile=profile,
                correlation_id=None,
                error_type=exc.__class__.__name__,
                message=str(exc),
            )
            raise
        duration = perf_counter() - started
        self._ensure_chunk_ids(document.id, chunks)
        self.storage.store(tenant_id, document.id, chunks)
        counts = defaultdict(int)
        section_counts: Counter[str] = Counter()
        intent_counts: Counter[str] = Counter()
        token_totals = 0
        length_totals = 0
        for chunk in chunks:
            counts[chunk.granularity] += 1
            record_chunk_size(profile, chunk.granularity, len(chunk.body))
            section_label = chunk.section or chunk.meta.get("section", "unknown")
            section_counts[section_label or "unknown"] += 1
            intent_value = (
                chunk.meta.get("intent_hint") or chunk.meta.get("intent") or "unspecified"
            )
            intent_counts[str(intent_value)] += 1
            token_totals += int(chunk.meta.get("token_count", 0) or 0)
            length_totals += len(chunk.body)
        chunk_count = len(chunks)
        average_chars = (length_totals / chunk_count) if chunk_count else 0.0
        average_tokens = (token_totals / chunk_count) if chunk_count else 0.0
        observe_chunking_latency(profile, duration)
        record_chunking_document(profile, duration, chunk_count)
        logger.info(
            "ingestion.chunked",
            document_id=document.id,
            tenant_id=tenant_id,
            profile=profile,
            chunks=len(chunks),
            duration=round(duration, 4),
        )
        logger.info(
            "ingestion.chunk_quality",
            document_id=document.id,
            tenant_id=tenant_id,
            profile=profile,
            average_chars=round(average_chars, 2),
            average_tokens=round(average_tokens, 2),
            section_distribution=dict(section_counts),
            intent_distribution=dict(intent_counts),
        )
        self.events.emit_completed(
            tenant_id=tenant_id,
            document_id=document.id,
            profile=profile,
            correlation_id=None,
            duration_ms=duration * 1000,
            chunks=chunk_count,
            average_tokens=average_tokens,
            average_chars=average_chars,
        )
        return ChunkingRun(
            document_id=document.id,
            profile=profile,
            duration_seconds=duration,
            chunks=chunks,
            granularity_counts=dict(counts),
            average_chars=average_chars,
            average_tokens=average_tokens,
            section_distribution=dict(section_counts),
            intent_distribution=dict(intent_counts),
        )

    def list_chunks(self, tenant_id: str, document_id: str) -> list[Chunk]:
        return self.storage.list(tenant_id, document_id)

    def _ensure_chunk_ids(self, document_id: str, chunks: list[Chunk]) -> None:
        for index, chunk in enumerate(chunks):
            if chunk.chunk_id.startswith(f"{document_id}:"):
                continue
            updated = chunk.model_copy(
                update={"chunk_id": f"{document_id}:{chunk.chunker}:{chunk.granularity}:{index}"}
            )
            chunks[index] = updated  # type: ignore[index]
