"""Lightweight caching layer for Docling VLM artifacts.

The original implementation included Redis integration, encryption, and a
variety of eviction strategies. The goal here is to provide a minimal,
syntactically valid subset that keeps the public surface area intact so the
rest of the codebase can run while the full feature set is rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import time

import structlog

logger = structlog.get_logger(__name__)


class CacheStrategy(str, Enum):
    """Supported cache eviction strategies."""

    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class CacheLevel(str, Enum):
    """Cache level identifiers."""

    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"


@dataclass
class VLMCacheConfig:
    """Configuration values for the cache."""

    enabled: bool = True
    max_size: int = 512
    cache_dir: Path = Path(".vlm_cache")
    strategy: CacheStrategy = CacheStrategy.LRU


@dataclass
class CacheEntry:
    """In-memory cache entry."""

    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 1
    ttl_seconds: Optional[int] = None


@dataclass
class VLMCacheStats:
    """Simple cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0


class VLMCache:
    """Minimal VLM cache implementation with in-memory + file persistence."""

    def __init__(self, config: VLMCacheConfig | None = None) -> None:
        self.config = config or VLMCacheConfig()
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = VLMCacheStats()
        self.cache_dir = self.config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def get(
        self,
        pdf_content: bytes,
        config: dict[str, Any],
        options: dict[str, Any],
        level: CacheLevel,
    ) -> Optional[Any]:
        """Retrieve a cached value if available."""
        if not self.config.enabled:
            return None

        key = self._generate_cache_key(pdf_content, config, options, level)

        entry = self.cache.get(key)
        if entry:
            if entry.ttl_seconds and (time.time() - entry.created_at) > entry.ttl_seconds:
                del self.cache[key]
                self.stats.misses += 1
                return None
            entry.accessed_at = time.time()
            entry.access_count += 1
            self.stats.hits += 1
            return entry.value

        file_value = await self._get_from_file(key)
        if file_value is not None:
            self.stats.hits += 1
            return file_value

        self.stats.misses += 1
        return None

    async def set(
        self,
        pdf_content: bytes,
        config: dict[str, Any],
        options: dict[str, Any],
        value: Any,
        level: CacheLevel,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store a value in the cache."""
        if not self.config.enabled:
            return

        key = self._generate_cache_key(pdf_content, config, options, level)
        now = time.time()
        self.cache[key] = CacheEntry(value=value, created_at=now, accessed_at=now, ttl_seconds=ttl_seconds)
        await self._set_to_file(key, value)
        self._evict_if_needed()

    def summary(self) -> dict[str, Any]:
        """Return a summary of cache statistics."""
        return {
            "enabled": self.config.enabled,
            "size": len(self.cache),
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_cache_key(
        self,
        pdf_content: bytes,
        config: dict[str, Any],
        options: dict[str, Any],
        level: CacheLevel,
    ) -> str:
        payload = {
            "level": level.value,
            "pdf": hashlib.sha256(pdf_content).hexdigest(),
            "config": config,
            "options": options,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    async def _get_from_file(self, key: str) -> Optional[Any]:
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            with cache_file.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("vlm_cache.file_read_failed", key=key, error=str(exc))
            return None

    async def _set_to_file(self, key: str, value: Any) -> None:
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with cache_file.open("w", encoding="utf-8") as handle:
                json.dump(value, handle)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("vlm_cache.file_write_failed", key=key, error=str(exc))

    def _evict_if_needed(self) -> None:
        if len(self.cache) <= self.config.max_size:
            return
        overflow = len(self.cache) - self.config.max_size
        keys = sorted(
            self.cache.items(),
            key=lambda item: item[1].accessed_at,
        )
        for index in range(min(overflow, len(keys))):
            key, _entry = keys[index]
            self.cache.pop(key, None)
            file_path = self.cache_dir / f"{key}.json"
            if file_path.exists():
                file_path.unlink()
            self.stats.evictions += 1


__all__ = [
    "CacheStrategy",
    "CacheLevel",
    "VLMCacheConfig",
    "VLMCacheStats",
    "VLMCache",
]
