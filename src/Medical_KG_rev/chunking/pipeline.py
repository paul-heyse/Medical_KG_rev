"""Multi granularity pipeline orchestrating multiple chunkers."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

from Medical_KG_rev.models.ir import Document

from .base import ContextualChunker
from .models import Chunk, Granularity
from .ports import BaseChunker
from .provenance import BlockContext


class MultiGranularityPipeline:
    """Execute multiple chunkers in parallel and merge results."""

    def __init__(
        self,
        chunkers: Sequence[tuple[BaseChunker, Granularity | None]],
        *,
        enable_multi_granularity: bool = True,
    ) -> None:
        if not chunkers:
            raise ValueError("MultiGranularityPipeline requires at least one chunker")
        self.chunkers = list(chunkers)
        self.enable_multi = enable_multi_granularity

    async def achunk(
        self,
        document: Document,
        *,
        tenant_id: str,
    ) -> list[Chunk]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._execute, document, tenant_id)

    def chunk(self, document: Document, *, tenant_id: str) -> list[Chunk]:
        return self._execute(document, tenant_id)

    def _iter_chunkers(self) -> Iterable[tuple[BaseChunker, Granularity | None]]:
        if self.enable_multi:
            yield from self.chunkers
        else:
            chunker, granularity = self.chunkers[0]
            yield chunker, granularity

    def _execute(self, document: Document, tenant_id: str) -> list[Chunk]:
        chunker_iter = list(self._iter_chunkers())
        if len(chunker_iter) == 1:
            chunker, granularity = chunker_iter[0]
            return self._run_chunker(
                chunker, document, tenant_id=tenant_id, granularity=granularity
            )

        chunks: list[Chunk] = []
        context_cache: dict[int, list[BlockContext]] = {}
        with ThreadPoolExecutor(max_workers=len(chunker_iter)) as executor:
            futures = [
                executor.submit(
                    self._run_chunker,
                    chunker,
                    document,
                    tenant_id,
                    granularity,
                    context_cache,
                )
                for chunker, granularity in chunker_iter
            ]
            for future in futures:
                chunks.extend(future.result())
        return chunks

    def _run_chunker(
        self,
        chunker: BaseChunker,
        document: Document,
        tenant_id: str,
        granularity: Granularity | None,
        context_cache: dict[int, list[BlockContext]] | None = None,
    ) -> list[Chunk]:
        if isinstance(chunker, ContextualChunker):
            contexts = self._prepare_contexts(chunker, document, context_cache)
            return chunker.chunk_with_contexts(
                document,
                contexts,
                tenant_id=tenant_id,
                granularity=granularity,
            )
        return chunker.chunk(
            document,
            tenant_id=tenant_id,
            granularity=granularity,
        )

    def _prepare_contexts(
        self,
        chunker: ContextualChunker,
        document: Document,
        context_cache: dict[int, list[BlockContext]] | None,
    ) -> list[BlockContext]:
        if context_cache is None:
            return chunker.prepare_contexts(document)
        key = id(chunker)
        if key not in context_cache:
            context_cache[key] = chunker.prepare_contexts(document)
        return context_cache[key]
