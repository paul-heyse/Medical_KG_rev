"""Cache backend implementations."""

from __future__ import annotations

import asyncio

from redis.asyncio import Redis

from .base import CacheBackend



class InMemoryCache(CacheBackend):
    """Simple in-memory cache with TTL support."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, float | None]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> bytes | None:
        async with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expires_at = item
            if expires_at is not None and expires_at < asyncio.get_event_loop().time():
                self._data.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        async with self._lock:
            expires_at = None
            if ttl:
                expires_at = asyncio.get_event_loop().time() + ttl
            self._data[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)


class RedisCache(CacheBackend):
    """Redis backed cache implementation."""

    def __init__(self, client: Redis | None = None) -> None:
        if Redis is None:
            raise RuntimeError("redis dependency is required for RedisCache")
        self._client = client or Redis()

    async def get(self, key: str) -> bytes | None:
        value = await self._client.get(key)
        return value

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        await self._client.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)
