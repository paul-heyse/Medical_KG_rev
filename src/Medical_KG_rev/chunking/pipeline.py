"""Multi granularity pipeline orchestrating multiple chunkers."""

from __future__ import annotations

import asyncio
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
        tasks = []
        for chunker, granularity in self._iter_chunkers():
            tasks.append(
                asyncio.to_thread(
                    chunker.chunk,
                    document,
                    tenant_id=tenant_id,
                    granularity=granularity,
                )
            )
        results = await asyncio.gather(*tasks)
        chunks: list[Chunk] = []
        for result in results:
            chunks.extend(result)
        return chunks

    def chunk(self, document: Document, *, tenant_id: str) -> list[Chunk]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.achunk(document, tenant_id=tenant_id))
        else:
            return loop.run_until_complete(self.achunk(document, tenant_id=tenant_id))

    def _iter_chunkers(self) -> Iterable[tuple[BaseChunker, Granularity | None]]:
        if self.enable_multi:
            yield from self.chunkers
        else:
            chunker, granularity = self.chunkers[0]
            yield chunker, granularity
