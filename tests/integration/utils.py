"""Utilities shared across MinerU integration tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


def run_async(awaitable: Awaitable[T]) -> T:
    """Execute an awaitable in a freshly created event loop and return the result."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(awaitable)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


__all__ = ["run_async"]
