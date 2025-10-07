"""SSE event manager with in-memory pub/sub suitable for tests."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from ..models import JobEvent


class EventStreamManager:
    """Manages per-job event queues for SSE streaming."""

    _HISTORY_LIMIT = 200

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[JobEvent]]] = defaultdict(list)
        self._pending: dict[str, list[JobEvent]] = defaultdict(list)
        self._history: dict[str, deque[JobEvent]] = defaultdict(
            lambda: deque(maxlen=self._HISTORY_LIMIT)
        )
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
        self._history[event.job_id].append(event)
        queues = list(self._subscribers.get(event.job_id, []))
        if not queues:
            self._pending[event.job_id].append(event)
            return
        for queue in queues:
            queue.put_nowait(event)

    def history(self, job_id: str, *, since: datetime | None = None) -> list[JobEvent]:
        events = list(self._history.get(job_id, []))
        if since is not None:
            events = [event for event in events if event.emitted_at >= since]
        # Include pending events that haven't been streamed yet.
        pending = self._pending.get(job_id, [])
        if pending:
            if since is None:
                events.extend(pending)
            else:
                events.extend(event for event in pending if event.emitted_at >= since)
        return sorted(events, key=lambda event: event.emitted_at)

    @asynccontextmanager
    async def open_stream(self, job_id: str) -> AsyncIterator[AsyncIterator[JobEvent]]:
        yield self.subscribe(job_id)
