"""Runtime helpers to orchestrate reusable chunking sessions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from Medical_KG_rev.models.ir import Document

from .configuration import ChunkerSettings
from .factory import ChunkerFactory, RegisteredChunker
from .models import Chunk
from .pipeline import MultiGranularityPipeline



@dataclass(slots=True, frozen=True)
class ChunkerPlan:
    """Definition of a compiled chunking plan."""

    entries: tuple[RegisteredChunker, ...]
    enable_multi_granularity: bool


class ChunkerSession:
    """Reusable view over a compiled chunking plan."""

    def __init__(
        self,
        plan: ChunkerPlan,
        *,
        pipeline: MultiGranularityPipeline,
    ) -> None:
        self._plan = plan
        self._pipeline = pipeline

    @property
    def plan(self) -> ChunkerPlan:
        return self._plan

    def chunk(self, document: Document, *, tenant_id: str) -> list[Chunk]:
        return self._pipeline.chunk(document, tenant_id=tenant_id)

    def explain(self) -> dict[str, object]:
        return {
            "enable_multi_granularity": self._plan.enable_multi_granularity,
            "chunkers": [
                {
                    "name": entry.instance.name,
                    "version": getattr(entry.instance, "version", "unknown"),
                    "granularity": entry.granularity,
                    "config": entry.instance.explain(),
                }
                for entry in self._plan.entries
            ],
        }


class ChunkingRuntime:
    """Orchestrates chunker instantiation and session creation."""

    def __init__(self, factory: ChunkerFactory | None = None) -> None:
        self.factory = factory or ChunkerFactory()

    def create_session(
        self,
        settings: Sequence[ChunkerSettings],
        *,
        allow_experimental: bool,
        enable_multi_granularity: bool,
    ) -> ChunkerSession:
        registered = self.factory.create_many(settings, allow_experimental=allow_experimental)
        plan = ChunkerPlan(
            entries=tuple(registered),
            enable_multi_granularity=enable_multi_granularity,
        )
        pipeline = MultiGranularityPipeline(
            chunkers=[(entry.instance, entry.granularity) for entry in plan.entries],
            enable_multi_granularity=plan.enable_multi_granularity,
        )
        return ChunkerSession(plan, pipeline=pipeline)

    def describe(self, session: ChunkerSession) -> dict[str, object]:
        return session.explain()

    def instantiate_many(
        self,
        settings: Iterable[ChunkerSettings],
        *,
        allow_experimental: bool,
    ) -> list[RegisteredChunker]:
        return self.factory.create_many(settings, allow_experimental=allow_experimental)
