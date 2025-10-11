"""Cache manager for distributed caching."""

from __future__ import annotations

import gzip
import json
import logging
import pickle
from typing import Any

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


class CacheManager:
    """Distributed cache manager using Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        """Initialize the cache manager."""
        self.redis_url = redis_url
        self.logger = logger
        self._redis_client: redis.Redis | None = None
        self._compression_enabled = True
        self._serialization_method = "json"

    async def _get_redis(self) -> redis.Redis:
        """Get Redis client."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url)
        return self._redis_client

    def _serialize(self, value: Any) -> bytes:
        """Serialize a value."""
        if self._serialization_method == "json":
            data = json.dumps(value).encode()
        else:  # pickle
            data = pickle.dumps(value)

        if self._compression_enabled:
            data = gzip.compress(data)

        return data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data."""
        if self._compression_enabled:
            data = gzip.decompress(data)

        if self._serialization_method == "json":
            return json.loads(data.decode())
        else:  # pickle
            return pickle.loads(data)

    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        try:
            redis_client = await self._get_redis()
            data = await redis_client.get(key)

            if data is None:
                return None

            # Deserialize and decompress value
            return self._deserialize(data)

        except Exception as exc:
            self.logger.error(f"Failed to get cache key {key}: {exc}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in cache."""
        try:
            redis_client = await self._get_redis()

            # Serialize and compress value
            data = self._serialize(value)

            # Set with optional TTL
            if ttl:
                await redis_client.setex(key, ttl, data)
            else:
                await redis_client.set(key, data)

            return True

        except Exception as exc:
            self.logger.error(f"Failed to set cache key {key}: {exc}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.delete(key)
            return result > 0

        except Exception as exc:
            self.logger.error(f"Failed to delete cache key {key}: {exc}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.exists(key)
            return result > 0

        except Exception as exc:
            self.logger.error(f"Failed to check cache key {key}: {exc}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for a key."""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.expire(key, ttl)
            return result

        except Exception as exc:
            self.logger.error(f"Failed to set expiration for cache key {key}: {exc}")
            return False

    async def ttl(self, key: str) -> int:
        """Get TTL for a key."""
        try:
            redis_client = await self._get_redis()
            return await redis_client.ttl(key)

        except Exception as exc:
            self.logger.error(f"Failed to get TTL for cache key {key}: {exc}")
            return -1

    async def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching pattern."""
        try:
            redis_client = await self._get_redis()
            keys = await redis_client.keys(pattern)
            return [key.decode() for key in keys]

        except Exception as exc:
            self.logger.error(f"Failed to get keys with pattern {pattern}: {exc}")
            return []

    async def flush(self) -> bool:
        """Flush all keys from cache."""
        try:
            redis_client = await self._get_redis()
            await redis_client.flushdb()
            return True

        except Exception as exc:
            self.logger.error(f"Failed to flush cache: {exc}")
            return False

    async def info(self) -> dict[str, Any]:
        """Get cache information."""
        try:
            redis_client = await self._get_redis()
            info = await redis_client.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }

        except Exception as exc:
            self.logger.error(f"Failed to get cache info: {exc}")
            return {}

    async def health_check(self) -> dict[str, Any]:
        """Check cache health."""
        try:
            redis_client = await self._get_redis()
            await redis_client.ping()

            return {
                "status": "healthy",
                "redis_url": self.redis_url,
                "compression_enabled": self._compression_enabled,
                "serialization_method": self._serialization_method,
            }

        except Exception as exc:
            return {
                "status": "unhealthy",
                "error": str(exc),
                "redis_url": self.redis_url,
            }

    def set_compression(self, enabled: bool) -> None:
        """Enable or disable compression."""
        self._compression_enabled = enabled

    def set_serialization(self, method: str) -> None:
        """Set serialization method."""
        if method not in ["json", "pickle"]:
            raise ValueError("Serialization method must be 'json' or 'pickle'")
        self._serialization_method = method

    async def close(self) -> None:
        """Close the cache manager."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None


class CacheManagerFactory:
    """Factory for creating cache managers."""

    @staticmethod
    def create(redis_url: str = "redis://localhost:6379") -> CacheManager:
        """Create a cache manager instance."""
        return CacheManager(redis_url)

    @staticmethod
    def create_with_config(config: dict[str, Any]) -> CacheManager:
        """Create a cache manager with configuration."""
        redis_url = config.get("redis_url", "redis://localhost:6379")
        manager = CacheManager(redis_url)

        if "compression_enabled" in config:
            manager.set_compression(config["compression_enabled"])

        if "serialization_method" in config:
            manager.set_serialization(config["serialization_method"])

        return manager


# Global cache manager instance
_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManagerFactory.create()

    return _cache_manager


def create_cache_manager(redis_url: str = "redis://localhost:6379") -> CacheManager:
    """Create a new cache manager instance."""
    return CacheManagerFactory.create(redis_url)
