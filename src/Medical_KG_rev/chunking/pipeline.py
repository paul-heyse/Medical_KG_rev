"""Multi granularity pipeline orchestrating multiple chunkers."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

from Medical_KG_rev.models.ir import Document

from .models import Chunk, Granularity
from .ports import BaseChunker


class MultiGranularityPipeline:
    """Execute multiple chunkers in parallel and merge results."""

    def __init__(
        self,
        *,
        chunkers: Sequence[tuple[BaseChunker, Granularity | None]],
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
        return await loop.run_in_executor(
            None, self._execute, document, tenant_id
        )

    def chunk(self, document: Document, *, tenant_id: str) -> list[Chunk]:
        return self._execute(document, tenant_id)

    def _iter_chunkers(self) -> Iterable[tuple[BaseChunker, Granularity | None]]:
        if self.enable_multi:
            yield from self.chunkers
        else:
            chunker, granularity = self.chunkers[0]
            yield chunker, granularity

    def _execute(self, document: Document, tenant_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        with ThreadPoolExecutor(max_workers=len(self.chunkers)) as executor:
            futures = [
                executor.submit(
                    chunker.chunk,
                    document,
                    tenant_id=tenant_id,
                    granularity=granularity,
                )
                for chunker, granularity in self._iter_chunkers()
            ]
            for future in futures:
                chunks.extend(future.result())
        return chunks
