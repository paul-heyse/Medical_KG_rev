"""SSE event manager with in-memory pub/sub suitable for tests."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List

from ..models import JobEvent


class EventStreamManager:
    """Manages per-job event queues for SSE streaming."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue[JobEvent]]] = defaultdict(list)
        self._pending: Dict[str, List[JobEvent]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> AsyncIterator[JobEvent]:
        queue: asyncio.Queue[JobEvent] = asyncio.Queue()
        async with self._lock:
            self._subscribers[job_id].append(queue)
            pending = self._pending.pop(job_id, [])
        for event in pending:
            queue.put_nowait(event)

        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            async with self._lock:
                self._subscribers[job_id].remove(queue)
                if not self._subscribers[job_id]:
                    self._subscribers.pop(job_id, None)

    def publish(self, event: JobEvent) -> None:
        queues = list(self._subscribers.get(event.job_id, []))
        if not queues:
            self._pending[event.job_id].append(event)
            return
        for queue in queues:
            queue.put_nowait(event)

    @asynccontextmanager
    async def open_stream(self, job_id: str) -> AsyncIterator[AsyncIterator[JobEvent]]:
        yield self.subscribe(job_id)
