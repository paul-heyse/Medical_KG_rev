"""Caching system for VLM operations.

This module provides intelligent caching for Docling VLM operations,
including PDF processing results, model outputs, and intermediate computations.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog

try:
    import redis
    from redis.asyncio import Redis
except ImportError:
    redis = None
    Redis = None

logger = structlog.get_logger(__name__)


class CacheStrategy(Enum):
    """Cache strategy for VLM operations."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


class CacheLevel(Enum):
    """Cache level for different types of operations."""
    PDF_CONTENT = "pdf_content"  # Raw PDF content hash
    VLM_OUTPUT = "vlm_output"  # VLM model output
    DOCTAGS_RESULT = "doctags_result"  # Processed Doctags result
    INTERMEDIATE = "intermediate"  # Intermediate processing steps


@dataclass
class CacheConfig:
    """Configuration for VLM caching."""
    enabled: bool = True
    strategy: CacheStrategy = CacheStrategy.LRU
    max_size: int = 1000  # Maximum number of cached items
    ttl_seconds: int = 3600  # Default TTL in seconds
    memory_limit_mb: int = 1024  # Memory limit in MB
    redis_url: Optional[str] = None
    cache_directory: str = "/tmp/vlm_cache"
    compression_enabled: bool = True
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None


@dataclass
class CacheEntry:
    """Cache entry for VLM operations."""
    key: str
    value: Any
    level: CacheLevel
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    total_entries: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0


class VLMCache:
    """Intelligent cache for VLM operations."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self.redis_client: Optional[Redis] = None
        self.cache_dir = Path(config.cache_directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Redis if configured
        if config.redis_url and redis:
            try:
                self.redis_client = Redis.from_url(config.redis_url)
                logger.info("Redis cache initialized", url=config.redis_url)
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self.redis_client = None

        logger.info(
            "VLM cache initialized",
            enabled=config.enabled,
            strategy=config.strategy.value,
            max_size=config.max_size,
            ttl_seconds=config.ttl_seconds,
            redis_available=self.redis_client is not None
        )

    def _generate_cache_key(
        self,
        pdf_content: bytes,
        config: Dict[str, Any],
        options: Dict[str, Any],
        level: CacheLevel
    ) -> str:
        """Generate cache key for VLM operation."""
        # Create hash of PDF content
        pdf_hash = hashlib.sha256(pdf_content).hexdigest()[:16]

        # Create hash of configuration
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Create hash of options
        options_str = json.dumps(options, sort_keys=True)
        options_hash = hashlib.md5(options_str.encode()).hexdigest()[:8]

        return f"vlm:{level.value}:{pdf_hash}:{config_hash}:{options_hash}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for caching."""
        if self.config.compression_enabled:
            import gzip
            data = json.dumps(value, default=str).encode()
            return gzip.compress(data)
        else:
            return json.dumps(value, default=str).encode()

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from cache."""
        if self.config.compression_enabled:
            import gzip
            data = gzip.decompress(data)
            return json.loads(data.decode())
        else:
            return json.loads(data.decode())

    def _encrypt_value(self, data: bytes) -> bytes:
        """Encrypt cached value."""
        if not self.config.encryption_enabled or not self.config.encryption_key:
            return data

        try:
            from cryptography.fernet import Fernet
            key = self.config.encryption_key.encode()
            if len(key) != 32:
                key = key.ljust(32, b'0')[:32]
            fernet = Fernet(key)
            return fernet.encrypt(data)
        except ImportError:
            logger.warning("cryptography not available, encryption disabled")
            return data
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data

    def _decrypt_value(self, data: bytes) -> bytes:
        """Decrypt cached value."""
        if not self.config.encryption_enabled or not self.config.encryption_key:
            return data

        try:
            from cryptography.fernet import Fernet
            key = self.config.encryption_key.encode()
            if len(key) != 32:
                key = key.ljust(32, b'0')[:32]
            fernet = Fernet(key)
            return fernet.decrypt(data)
        except ImportError:
            logger.warning("cryptography not available, decryption disabled")
            return data
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return data

    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            return None

        try:
            data = await self.redis_client.get(key)
            if data:
                data = self._decrypt_value(data)
                return self._deserialize_value(data)
        except Exception as e:
            logger.error(f"Redis get failed: {e}")

        return None

    async def _set_to_redis(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value to Redis cache."""
        if not self.redis_client:
            return False

        try:
            data = self._serialize_value(value)
            data = self._encrypt_value(data)

            if ttl_seconds:
                await self.redis_client.setex(key, ttl_seconds, data)
            else:
                await self.redis_client.set(key, data)

            return True
        except Exception as e:
            logger.error(f"Redis set failed: {e}")
            return False

    async def _get_from_file(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        cache_file = self.cache_dir / f"{key}.cache"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                data = f.read()

            data = self._decrypt_value(data)
            return self._deserialize_value(data)
        except Exception as e:
            logger.error(f"File cache get failed: {e}")
            return None

    async def _set_to_file(self, key: str, value: Any) -> bool:
        """Set value to file cache."""
        cache_file = self.cache_dir / f"{key}.cache"

        try:
            data = self._serialize_value(value)
            data = self._encrypt_value(data)

            with open(cache_file, "wb") as f:
                f.write(data)

            return True
        except Exception as e:
            logger.error(f"File cache set failed: {e}")
            return False

    def _evict_entries(self) -> None:
        """Evict entries based on cache strategy."""
        if len(self.cache) <= self.config.max_size:
            return

        entries_to_remove = len(self.cache) - self.config.max_size

        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used entries
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].accessed_at
            )
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used entries
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )
        elif self.config.strategy == CacheStrategy.TTL:
            # Remove expired entries first, then by access time
            current_time = time.time()
            expired_entries = [
                (k, v) for k, v in self.cache.items()
                if v.ttl_seconds and (current_time - v.created_at) > v.ttl_seconds
            ]
            if expired_entries:
                sorted_entries = expired_entries
            else:
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: x[1].accessed_at
                )
        else:  # ADAPTIVE
            # Combine LRU and LFU with weights
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (
                    0.7 * x[1].accessed_at +  # LRU weight
                    0.3 * (1.0 / max(x[1].access_count, 1))  # LFU weight
                )
            )

        # Remove entries
        for i in range(min(entries_to_remove, len(sorted_entries))):
            key, entry = sorted_entries[i]
            del self.cache[key]
            self.stats.evictions += 1

            # Remove file cache if exists
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                cache_file.unlink()

        logger.info(f"Evicted {entries_to_remove} cache entries")

    async def get(
        self,
        pdf_content: bytes,
        config: Dict[str, Any],
        options: Dict[str, Any],
        level: CacheLevel
    ) -> Optional[Any]:
        """Get cached value for VLM operation."""
        if not self.config.enabled:
            return None

        key = self._generate_cache_key(pdf_content, config, options, level)

        # Try in-memory cache first
        if key in self.cache:
            entry = self.cache[key]

            # Check TTL
            if entry.ttl_seconds and (time.time() - entry.created_at) > entry.ttl_seconds:
                del self.cache[key]
                self.stats.misses += 1
                return None

            # Update access info
            entry.accessed_at = time.time()
            entry.access_count += 1
            self.stats.hits += 1

            logger.debug("Cache hit", key=key, level=level.value)
            return entry.value

        # Try Redis cache
        if self.redis_client:
            value = await self._get_from_redis(key)
            if value is not None:
                self.stats.hits += 1
                logger.debug("Redis cache hit", key=key, level=level.value)
                return value

        # Try file cache
        value = await self._get_from_file(key)
        if value is not None:
            self.stats.hits += 1
            logger.debug("File cache hit", key=key, level=level.value)
            return value

        self.stats.misses += 1
        logger.debug("Cache miss", key=key, level=level.value)
        return None

    async def set(
        self,
        pdf_content: bytes,
        config: Dict[str, Any],
        options: Dict[str, Any],
        level: CacheLevel,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set cached value for VLM operation."""
        if not self.config.enabled:
            return False

        key = self._generate_cache_key(pdf_content, config, options, level)
        current_time = time.time()

        # Calculate size
        try:
            size_bytes = len(self._serialize_value(value))
        except Exception:
            size_bytes = 0

        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            level=level,
            created_at=current_time,
            accessed_at=current_time,
            size_bytes=size_bytes,
            ttl_seconds=ttl_seconds or self.config.ttl_seconds,
            metadata={
                "pdf_size": len(pdf_content),
                "config": config,
                "options": options
            }
        )

        # Store in memory cache
        self.cache[key] = entry
        self.stats.total_entries = len(self.cache)
        self.stats.total_size_bytes += size_bytes

        # Store in Redis if available
        if self.redis_client:
            await self._set_to_redis(key, value, entry.ttl_seconds)

        # Store in file cache
        await self._set_to_file(key, value)

        # Evict if necessary
        self._evict_entries()

        logger.debug("Cache set", key=key, level=level.value, size_bytes=size_bytes)
        return True

    async def invalidate(
        self,
        pdf_content: bytes,
        config: Dict[str, Any],
        options: Dict[str, Any],
        level: CacheLevel
    ) -> bool:
        """Invalidate cached value."""
        key = self._generate_cache_key(pdf_content, config, options, level)

        # Remove from memory cache
        if key in self.cache:
            del self.cache[key]
            self.stats.total_entries = len(self.cache)

        # Remove from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete failed: {e}")

        # Remove from file cache
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            cache_file.unlink()

        logger.debug("Cache invalidated", key=key, level=level.value)
        return True

    async def clear(self) -> None:
        """Clear all cached values."""
        # Clear memory cache
        self.cache.clear()
        self.stats.total_entries = 0
        self.stats.total_size_bytes = 0

        # Clear Redis cache
        if self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis flush failed: {e}")

        # Clear file cache
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()

        logger.info("Cache cleared")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests
            self.stats.miss_rate = self.stats.misses / total_requests

        return self.stats

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        return {
            "enabled": self.config.enabled,
            "strategy": self.config.strategy.value,
            "max_size": self.config.max_size,
            "ttl_seconds": self.config.ttl_seconds,
            "memory_limit_mb": self.config.memory_limit_mb,
            "redis_available": self.redis_client is not None,
            "cache_directory": str(self.cache_dir),
            "compression_enabled": self.config.compression_enabled,
            "encryption_enabled": self.config.encryption_enabled,
            "stats": self.get_stats().__dict__,
            "entries_by_level": {
                level.value: len([e for e in self.cache.values() if e.level == level])
                for level in CacheLevel
            }
        }


# Global cache instance
_vlm_cache: Optional[VLMCache] = None


def get_vlm_cache() -> VLMCache:
    """Get global VLM cache instance."""
    global _vlm_cache
    if _vlm_cache is None:
        config = CacheConfig()
        _vlm_cache = VLMCache(config)
    return _vlm_cache


def create_vlm_cache(config: CacheConfig) -> VLMCache:
    """Create a new VLM cache instance."""
    return VLMCache(config)
