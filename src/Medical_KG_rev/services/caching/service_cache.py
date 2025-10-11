"""Service-specific caching implementations for GPU services.

This module provides caching decorators and utilities for gRPC service calls
to improve performance and reduce latency.
"""

from collections.abc import Callable
from typing import Any, TypeVar
import functools
import hashlib
import json
import logging
import time

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

from .cache_manager import CacheConfig, ServiceCacheManager, create_cache_manager


logger = logging.getLogger(__name__)

# Prometheus metrics
cached_operations_total = Counter(
    "cached_operations_total", "Total cached operations", ["service", "operation"]
)
cache_performance_seconds = Histogram(
    "cache_performance_seconds", "Cache operation performance", ["service", "operation"]
)

T = TypeVar("T")


class CachePolicy(BaseModel):
    """Cache policy configuration."""

    enabled: bool = Field(default=True, description="Enable caching for this operation")
    ttl_seconds: int = Field(default=3600, description="Time to live in seconds")
    key_include_params: list[str] = Field(
        default_factory=list, description="Parameters to include in cache key"
    )
    key_exclude_params: list[str] = Field(
        default_factory=list, description="Parameters to exclude from cache key"
    )
    condition: str | None = Field(
        default=None, description="Condition for caching (e.g., 'result.success')"
    )


class ServiceCacheConfig(BaseModel):
    """Configuration for service caching."""

    cache_manager: ServiceCacheManager | None = Field(
        default=None, description="Cache manager instance"
    )
    default_policy: CachePolicy = Field(
        default_factory=CachePolicy, description="Default cache policy"
    )
    policies: dict[str, CachePolicy] = Field(
        default_factory=dict, description="Operation-specific policies"
    )
    cache_type: str = Field(default="memory", description="Cache backend type")
    redis_url: str | None = Field(default=None, description="Redis URL for Redis cache")


class ServiceCache:
    """Service cache implementation for gRPC services."""

    def __init__(self, config: ServiceCacheConfig):
        self.config = config
        self._cache_manager = config.cache_manager or create_cache_manager(
            cache_type=config.cache_type, config=CacheConfig()
        )
        self._operation_stats: dict[str, dict[str, Any]] = {}

    def _generate_cache_key(
        self, service: str, operation: str, params: dict[str, Any], policy: CachePolicy
    ) -> str:
        """Generate a cache key based on service, operation, and parameters."""
        # Filter parameters based on policy
        filtered_params = {}

        if policy.key_include_params:
            # Include only specified parameters
            for param in policy.key_include_params:
                if param in params:
                    filtered_params[param] = params[param]
        else:
            # Include all parameters except excluded ones
            for key, value in params.items():
                if key not in policy.key_exclude_params:
                    filtered_params[key] = value

        # Create key data
        key_data = {"service": service, "operation": operation, "params": filtered_params}

        # Generate hash
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _should_cache(self, result: Any, policy: CachePolicy) -> bool:
        """Determine if a result should be cached based on policy."""
        if not policy.enabled:
            return False

        if policy.condition:
            # Evaluate condition (simplified implementation)
            try:
                # This is a basic implementation - in practice, you'd use a proper expression evaluator
                if policy.condition == "result.success":
                    return hasattr(result, "success") and result.success
                elif policy.condition == "result.error_code == 0":
                    return hasattr(result, "error_code") and result.error_code == 0
                # Add more conditions as needed
            except Exception as e:
                logger.warning(f"Failed to evaluate cache condition '{policy.condition}': {e}")
                return False

        return True

    async def get(self, service: str, operation: str, params: dict[str, Any]) -> Any | None:
        """Get a cached result."""
        policy = self.config.policies.get(operation, self.config.default_policy)

        if not policy.enabled:
            return None

        key = self._generate_cache_key(service, operation, params, policy)

        # Note: Prometheus histogram timing context manager not available
        start_time = time.time()
        result = await self._cache_manager.get(service, operation, params)
        cache_performance_seconds.labels(service=service, operation=operation).observe(
            time.time() - start_time
        )

        if result is not None:
            cached_operations_total.labels(service=service, operation=operation).inc()
            logger.debug(f"Cache hit for {service}.{operation}")

        return result

    async def set(self, service: str, operation: str, params: dict[str, Any], result: Any) -> None:
        """Set a cached result."""
        policy = self.config.policies.get(operation, self.config.default_policy)

        if not self._should_cache(result, policy):
            return

        key = self._generate_cache_key(service, operation, params, policy)

        # Note: Prometheus histogram timing context manager not available
        start_time = time.time()
        await self._cache_manager.set(service, operation, params, result, policy.ttl_seconds)
        cache_performance_seconds.labels(service=service, operation=operation).observe(
            time.time() - start_time
        )

        cached_operations_total.labels(service=service, operation=operation).inc()
        logger.debug(f"Cached result for {service}.{operation}")

    async def invalidate(
        self, service: str, operation: str, params: dict[str, Any] | None = None
    ) -> None:
        """Invalidate cache entries."""
        if params:
            await self._cache_manager.delete(service, operation, params)
        else:
            await self._cache_manager.clear_service(service)

        logger.debug(f"Invalidated cache for {service}.{operation}")

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return await self._cache_manager.get_stats()

    async def close(self) -> None:
        """Close the cache and cleanup resources."""
        await self._cache_manager.close()


def cached_operation(
    service_name: str,
    operation_name: str,
    cache_config: ServiceCacheConfig | None = None,
    policy: CachePolicy | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for caching service operations.

    Args:
    ----
        service_name: Name of the service
        operation_name: Name of the operation
        cache_config: Cache configuration
        policy: Cache policy for this operation

    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Extract parameters for caching
            if args and hasattr(args[0], "__dict__"):
                # If first argument is self, get parameters from kwargs
                params = kwargs
            else:
                # Combine args and kwargs
                params = {}
                if args:
                    params.update({f"arg_{i}": arg for i, arg in enumerate(args)})
                params.update(kwargs)

            # Get cache configuration
            config = cache_config or ServiceCacheConfig()
            cache = ServiceCache(config)

            # Get cache policy
            operation_policy = policy or config.policies.get(operation_name, config.default_policy)

            # Try to get from cache
            if operation_policy.enabled:
                cached_result = await cache.get(service_name, operation_name, params)
                if cached_result is not None:
                    return cached_result

            # Execute the function
            result = await func(*args, **kwargs)

            # Cache the result
            if operation_policy.enabled:
                await cache.set(service_name, operation_name, params, result)

            return result

        return wrapper

    return decorator


class EmbeddingCache:
    """Specialized cache for embedding operations."""

    def __init__(self, cache_config: ServiceCacheConfig):
        self.cache = ServiceCache(cache_config)
        self.service_name = "embedding_service"

    async def get_embedding(
        self, text: str, model: str, config: dict[str, Any] | None = None
    ) -> Any | None:
        """Get cached embedding."""
        params = {"text": text, "model": model, "config": config or {}}
        return await self.cache.get(self.service_name, "generate_embedding", params)

    async def set_embedding(
        self, text: str, model: str, embedding: Any, config: dict[str, Any] | None = None
    ) -> None:
        """Set cached embedding."""
        params = {"text": text, "model": model, "config": config or {}}
        await self.cache.set(self.service_name, "generate_embedding", params, embedding)

    async def invalidate_embeddings(self, model: str | None = None) -> None:
        """Invalidate embedding cache."""
        if model:
            params = {"model": model}
            await self.cache.invalidate(self.service_name, "generate_embedding", params)
        else:
            await self.cache.invalidate(self.service_name, "generate_embedding")


class RerankingCache:
    """Specialized cache for reranking operations."""

    def __init__(self, cache_config: ServiceCacheConfig):
        self.cache = ServiceCache(cache_config)
        self.service_name = "reranking_service"

    async def get_reranking(
        self, query: str, documents: list[str], model: str, config: dict[str, Any] | None = None
    ) -> Any | None:
        """Get cached reranking result."""
        params = {"query": query, "documents": documents, "model": model, "config": config or {}}
        return await self.cache.get(self.service_name, "rerank", params)

    async def set_reranking(
        self,
        query: str,
        documents: list[str],
        model: str,
        result: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Set cached reranking result."""
        params = {"query": query, "documents": documents, "model": model, "config": config or {}}
        await self.cache.set(self.service_name, "rerank", params, result)

    async def invalidate_reranking(self, model: str | None = None) -> None:
        """Invalidate reranking cache."""
        if model:
            params = {"model": model}
            await self.cache.invalidate(self.service_name, "rerank", params)
        else:
            await self.cache.invalidate(self.service_name, "rerank")


class DoclingVLMCache:
    """Specialized cache for Docling VLM operations."""

    def __init__(self, cache_config: ServiceCacheConfig):
        self.cache = ServiceCache(cache_config)
        self.service_name = "docling_vlm_service"

    async def get_processing_result(
        self, pdf_content: bytes, config: dict[str, Any] | None = None
    ) -> Any | None:
        """Get cached VLM processing result."""
        # Use hash of PDF content for caching
        pdf_hash = hashlib.sha256(pdf_content).hexdigest()
        params = {"pdf_hash": pdf_hash, "config": config or {}}
        return await self.cache.get(self.service_name, "process_pdf", params)

    async def set_processing_result(
        self, pdf_content: bytes, result: Any, config: dict[str, Any] | None = None
    ) -> None:
        """Set cached VLM processing result."""
        # Use hash of PDF content for caching
        pdf_hash = hashlib.sha256(pdf_content).hexdigest()
        params = {"pdf_hash": pdf_hash, "config": config or {}}
        await self.cache.set(self.service_name, "process_pdf", params, result)

    async def invalidate_processing_results(self) -> None:
        """Invalidate VLM processing cache."""
        await self.cache.invalidate(self.service_name, "process_pdf")


def create_service_cache_config(
    cache_type: str = "memory", redis_url: str | None = None
) -> ServiceCacheConfig:
    """Create a service cache configuration."""
    return ServiceCacheConfig(
        cache_type=cache_type,
        redis_url=redis_url,
        default_policy=CachePolicy(
            enabled=True,
            ttl_seconds=3600,
            key_exclude_params=["pdf_content", "large_data"],  # Exclude large binary data
        ),
        policies={
            "generate_embedding": CachePolicy(
                enabled=True,
                ttl_seconds=7200,  # Longer TTL for embeddings
                key_include_params=["text", "model", "config"],
            ),
            "rerank": CachePolicy(
                enabled=True,
                ttl_seconds=1800,  # Shorter TTL for reranking
                key_include_params=["query", "documents", "model", "config"],
            ),
            "process_pdf": CachePolicy(
                enabled=True, ttl_seconds=3600, key_include_params=["pdf_hash", "config"]
            ),
        },
    )
