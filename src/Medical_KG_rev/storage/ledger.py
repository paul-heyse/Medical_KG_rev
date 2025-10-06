"""Ledger/state tracking implementation."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .base import LedgerStore


@dataclass
class LedgerRecord:
    job_id: str
    state: Dict[str, Any]
    updated_at: datetime


class InMemoryLedger(LedgerStore):
    """In-memory implementation for tests."""

    def __init__(self) -> None:
        self._records: Dict[str, LedgerRecord] = {}
        self._lock = asyncio.Lock()

    async def record_state(self, job_id: str, state: Dict[str, Any]) -> None:
        async with self._lock:
            self._records[job_id] = LedgerRecord(job_id=job_id, state=state, updated_at=datetime.now(timezone.utc))

    async def get_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            record = self._records.get(job_id)
            return record.state if record else None

    async def list_jobs(self) -> Dict[str, LedgerRecord]:
        async with self._lock:
            return dict(self._records)
