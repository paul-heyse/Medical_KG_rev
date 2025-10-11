"""Service caching manager for GPU services.

This module provides caching strategies for gRPC service calls to improve
performance and reduce latency for repeated operations.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import redis.asyncio as redis
from pydantic import BaseModel, Field

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
cache_hits_total = Counter("cache_hits_total", "Total cache hits", ["service", "operation"])
cache_misses_total = Counter("cache_misses_total", "Total cache misses", ["service", "operation"])
cache_operations_total = Counter(
    "cache_operations_total", "Total cache operations", ["service", "operation", "result"]
)
cache_size_bytes = Gauge("cache_size_bytes", "Cache size in bytes", ["service"])
cache_ttl_seconds = Histogram("cache_ttl_seconds", "Cache TTL distribution", ["service"])

T = TypeVar("T")


class CacheConfig(BaseModel):
    """Configuration for cache operations."""

    ttl_seconds: int = Field(default=3600, description="Time to live in seconds")
    max_size_bytes: int = Field(
        default=1024 * 1024 * 1024, description="Maximum cache size in bytes"
    )
    compression_enabled: bool = Field(
        default=True, description="Enable compression for cached data"
    )
    serialization_format: str = Field(default="json", description="Serialization format")


class CacheEntry(BaseModel):
    """A cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    ttl_seconds: int
    size_bytes: int
    hit_count: int = 0
    last_accessed: float = Field(default_factory=time.time)


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Set a value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        pass


class InMemoryCacheStrategy(CacheStrategy):
    """In-memory cache strategy using LRU eviction."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self._total_size_bytes = 0
        self._lock = asyncio.Lock()

    def _generate_key(self, service: str, operation: str, params: dict[str, Any]) -> str:
        """Generate a cache key from service, operation, and parameters."""
        key_data = {"service": service, "operation": operation, "params": params}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize a value for storage."""
        if self.config.serialization_format == "json":
            return json.dumps(value, default=str).encode("utf-8")
        elif self.config.serialization_format == "pickle":
            import pickle

            return pickle.dumps(value)
        else:
            raise ValueError(
                f"Unsupported serialization format: {self.config.serialization_format}"
            )

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize a value from storage."""
        if self.config.serialization_format == "json":
            return json.loads(data.decode("utf-8"))
        elif self.config.serialization_format == "pickle":
            import pickle

            return pickle.loads(data)
        else:
            raise ValueError(
                f"Unsupported serialization format: {self.config.serialization_format}"
            )

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if compression is enabled."""
        if not self.config.compression_enabled:
            return data

        import gzip

        return gzip.compress(data)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if compression is enabled."""
        if not self.config.compression_enabled:
            return data

        import gzip

        return gzip.decompress(data)

    def _evict_lru(self) -> None:
        """Evict least recently used entries to make space."""
        while self._total_size_bytes > self.config.max_size_bytes and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                entry = self._cache[oldest_key]
                self._total_size_bytes -= entry.size_bytes
                del self._cache[oldest_key]
                logger.debug(f"Evicted LRU cache entry: {oldest_key}")

    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check if entry has expired
            if time.time() - entry.created_at > entry.ttl_seconds:
                self._total_size_bytes -= entry.size_bytes
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None

            # Update access information
            entry.hit_count += 1
            entry.last_accessed = time.time()

            # Move to end of access order (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            # Deserialize and decompress value
            try:
                compressed_data = self._deserialize_value(entry.value)
                decompressed_data = self._decompress_data(compressed_data)
                return self._deserialize_value(decompressed_data)
            except Exception as e:
                logger.error(f"Failed to deserialize cache entry {key}: {e}")
                # Remove corrupted entry
                self._total_size_bytes -= entry.size_bytes
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Set a value in cache."""
        async with self._lock:
            ttl = ttl_seconds or self.config.ttl_seconds

            # Serialize and compress value
            try:
                serialized_data = self._serialize_value(value)
                compressed_data = self._compress_data(serialized_data)
                final_data = self._serialize_value(compressed_data)
            except Exception as e:
                logger.error(f"Failed to serialize cache value for key {key}: {e}")
                return

            size_bytes = len(final_data)

            # Check if we need to evict entries
            while (
                self._total_size_bytes + size_bytes > self.config.max_size_bytes
                and self._access_order
            ):
                self._evict_lru()

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=final_data,
                created_at=time.time(),
                ttl_seconds=ttl,
                size_bytes=size_bytes,
            )

            # Remove existing entry if it exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_size_bytes -= old_entry.size_bytes
                if key in self._access_order:
                    self._access_order.remove(key)

            # Add new entry
            self._cache[key] = entry
            self._total_size_bytes += size_bytes
            self._access_order.append(key)

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        async with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            self._total_size_bytes -= entry.size_bytes
            del self._cache[key]

            if key in self._access_order:
                self._access_order.remove(key)

            return True

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._total_size_bytes = 0

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_hits = sum(entry.hit_count for entry in self._cache.values())
            total_entries = len(self._cache)

            return {
                "total_entries": total_entries,
                "total_size_bytes": self._total_size_bytes,
                "total_hits": total_hits,
                "hit_rate": total_hits / max(total_entries, 1),
                "max_size_bytes": self.config.max_size_bytes,
                "utilization_percent": (self._total_size_bytes / self.config.max_size_bytes) * 100,
            }


class RedisCacheStrategy(CacheStrategy):
    """Redis-based cache strategy."""

    def __init__(self, config: CacheConfig, redis_url: str = "redis://localhost:6379"):
        self.config = config
        self.redis_url = redis_url
        self._redis: redis.Redis | None = None
        self._key_prefix = "medical_kg_cache:"

    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=False)
        return self._redis

    def _generate_key(self, service: str, operation: str, params: dict[str, Any]) -> str:
        """Generate a cache key from service, operation, and parameters."""
        key_data = {"service": service, "operation": operation, "params": params}
        key_str = json.dumps(key_data, sort_keys=True)
        hash_key = hashlib.sha256(key_str.encode()).hexdigest()
        return f"{self._key_prefix}{hash_key}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize a value for storage."""
        if self.config.serialization_format == "json":
            return json.dumps(value, default=str).encode("utf-8")
        elif self.config.serialization_format == "pickle":
            import pickle

            return pickle.dumps(value)
        else:
            raise ValueError(
                f"Unsupported serialization format: {self.config.serialization_format}"
            )

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize a value from storage."""
        if self.config.serialization_format == "json":
            return json.loads(data.decode("utf-8"))
        elif self.config.serialization_format == "pickle":
            import pickle

            return pickle.loads(data)
        else:
            raise ValueError(
                f"Unsupported serialization format: {self.config.serialization_format}"
            )

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if compression is enabled."""
        if not self.config.compression_enabled:
            return data

        import gzip

        return gzip.compress(data)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if compression is enabled."""
        if not self.config.compression_enabled:
            return data

        import gzip

        return gzip.decompress(data)

    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        try:
            redis_client = await self._get_redis()
            data = await redis_client.get(key)

            if data is None:
                return None

            # Deserialize and decompress value
            compressed_data = self._deserialize_value(data)
            decompressed_data = self._decompress_data(compressed_data)
            return self._deserialize_value(decompressed_data)

        except Exception as e:
            logger.error(f"Failed to get cache entry {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Set a value in cache."""
        try:
            redis_client = await self._get_redis()
            ttl = ttl_seconds or self.config.ttl_seconds

            # Serialize and compress value
            serialized_data = self._serialize_value(value)
            compressed_data = self._compress_data(serialized_data)
            final_data = self._serialize_value(compressed_data)

            await redis_client.setex(key, ttl, final_data)

        except Exception as e:
            logger.error(f"Failed to set cache entry {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.delete(key)
            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete cache entry {key}: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            redis_client = await self._get_redis()
            keys = await redis_client.keys(f"{self._key_prefix}*")
            if keys:
                await redis_client.delete(*keys)

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            redis_client = await self._get_redis()
            info = await redis_client.info("memory")
            keys = await redis_client.keys(f"{self._key_prefix}*")

            return {
                "total_entries": len(keys),
                "total_size_bytes": info.get("used_memory", 0),
                "hit_rate": 0.0,  # Redis doesn't provide hit rate in basic info
                "max_size_bytes": info.get("maxmemory", 0),
                "utilization_percent": 0.0,  # Would need more complex calculation
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


class ServiceCacheManager:
    """Manager for service caching operations."""

    def __init__(self, strategy: CacheStrategy, config: CacheConfig):
        self.strategy = strategy
        self.config = config
        self._service_configs: dict[str, CacheConfig] = {}

    def configure_service(self, service_name: str, config: CacheConfig) -> None:
        """Configure caching for a specific service."""
        self._service_configs[service_name] = config

    def _generate_key(self, service: str, operation: str, params: dict[str, Any]) -> str:
        """Generate a cache key from service, operation, and parameters."""
        key_data = {"service": service, "operation": operation, "params": params}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_service_config(self, service: str) -> CacheConfig:
        """Get cache configuration for a service."""
        return self._service_configs.get(service, self.config)

    async def get(self, service: str, operation: str, params: dict[str, Any]) -> Any | None:
        """Get a cached result for a service operation."""
        key = self._generate_key(service, operation, params)
        result = await self.strategy.get(key)

        if result is not None:
            cache_hits_total.labels(service=service, operation=operation).inc()
            logger.debug(f"Cache hit for {service}.{operation}")
        else:
            cache_misses_total.labels(service=service, operation=operation).inc()
            logger.debug(f"Cache miss for {service}.{operation}")

        return result

    async def set(
        self,
        service: str,
        operation: str,
        params: dict[str, Any],
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """Set a cached result for a service operation."""
        key = self._generate_key(service, operation, params)
        service_config = self._get_service_config(service)
        ttl = ttl_seconds or service_config.ttl_seconds

        await self.strategy.set(key, value, ttl)
        cache_operations_total.labels(service=service, operation=operation, result="set").inc()

        logger.debug(f"Cached result for {service}.{operation}")

    async def delete(self, service: str, operation: str, params: dict[str, Any]) -> bool:
        """Delete a cached result for a service operation."""
        key = self._generate_key(service, operation, params)
        result = await self.strategy.delete(key)

        if result:
            cache_operations_total.labels(
                service=service, operation=operation, result="delete"
            ).inc()
            logger.debug(f"Deleted cache entry for {service}.{operation}")

        return result

    async def clear_service(self, service: str) -> None:
        """Clear all cache entries for a service."""
        # This is a simplified implementation - in practice, you'd need to track keys per service
        logger.warning(f"Clear service cache not fully implemented for {service}")

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return await self.strategy.get_stats()

    async def close(self) -> None:
        """Close the cache manager and cleanup resources."""
        if hasattr(self.strategy, "close"):
            await self.strategy.close()


def create_cache_manager(
    cache_type: str = "memory", config: CacheConfig | None = None
) -> ServiceCacheManager:
    """Create a cache manager with the specified strategy."""
    if config is None:
        config = CacheConfig()

    if cache_type == "memory":
        strategy: CacheStrategy = InMemoryCacheStrategy(config)
    elif cache_type == "redis":
        strategy = RedisCacheStrategy(config)
    else:
        raise ValueError(f"Unsupported cache type: {cache_type}")

    return ServiceCacheManager(strategy, config)
