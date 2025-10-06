"""Ledger/state tracking implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .base import LedgerStore


@dataclass
class LedgerRecord:
    job_id: str
    state: dict[str, Any]
    updated_at: datetime


class InMemoryLedger(LedgerStore):
    """In-memory implementation for tests."""

    def __init__(self) -> None:
        self._records: dict[str, LedgerRecord] = {}
        self._lock = asyncio.Lock()

    async def record_state(self, job_id: str, state: dict[str, Any]) -> None:
        async with self._lock:
            self._records[job_id] = LedgerRecord(
                job_id=job_id, state=state, updated_at=datetime.now(UTC)
            )

    async def get_state(self, job_id: str) -> dict[str, Any] | None:
        async with self._lock:
            record = self._records.get(job_id)
            return record.state if record else None

    async def list_jobs(self) -> dict[str, LedgerRecord]:
        async with self._lock:
            return dict(self._records)
