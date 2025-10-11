"""Cache integration for gRPC service clients.

This module provides integration between the caching system and gRPC service clients
to enable transparent caching of service calls.
"""

import logging
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from .service_cache import (
    DoclingVLMCache,
    EmbeddingCache,
    RerankingCache,
    ServiceCacheConfig,
    create_service_cache_config,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheIntegrationConfig(BaseModel):
    """Configuration for cache integration."""

    enabled: bool = Field(default=True, description="Enable cache integration")
    cache_type: str = Field(default="memory", description="Cache backend type")
    redis_url: str | None = Field(default=None, description="Redis URL for Redis cache")
    service_configs: dict[str, ServiceCacheConfig] = Field(
        default_factory=dict, description="Service-specific configurations"
    )


class CacheIntegratedClient:
    """Base class for cache-integrated gRPC clients."""

    def __init__(self, cache_config: CacheIntegrationConfig | None = None):
        self.cache_config = cache_config or CacheIntegrationConfig()
        self._cache_instances: dict[str, Any] = {}
        self._initialized = False

    async def _initialize_caches(self) -> None:
        """Initialize cache instances for all services."""
        if self._initialized:
            return

        if not self.cache_config.enabled:
            logger.info("Cache integration disabled")
            self._initialized = True
            return

        try:
            # Create default cache configuration
            default_config = create_service_cache_config(
                cache_type=self.cache_config.cache_type, redis_url=self.cache_config.redis_url
            )

            # Initialize service-specific caches
            self._cache_instances["embedding"] = EmbeddingCache(default_config)
            self._cache_instances["reranking"] = RerankingCache(default_config)
            self._cache_instances["docling_vlm"] = DoclingVLMCache(default_config)

            logger.info(
                f"Initialized cache integration with {self.cache_config.cache_type} backend"
            )
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize cache integration: {e}")
            self._initialized = True  # Mark as initialized to prevent retries

    async def _get_cache(self, service_name: str) -> Any | None:
        """Get cache instance for a service."""
        await self._initialize_caches()
        return self._cache_instances.get(service_name)

    async def close_caches(self) -> None:
        """Close all cache instances."""
        for cache in self._cache_instances.values():
            if hasattr(cache, "close"):
                await cache.close()
        self._cache_instances.clear()


class CacheIntegratedEmbeddingClient(CacheIntegratedClient):
    """Cache-integrated embedding client."""

    def __init__(self, embedding_client, cache_config: CacheIntegrationConfig | None = None):
        super().__init__(cache_config)
        self.embedding_client = embedding_client

    async def generate_embedding(
        self, text: str, model: str, config: dict[str, Any] | None = None
    ) -> Any:
        """Generate embedding with caching."""
        cache = await self._get_cache("embedding")

        if cache:
            # Try to get from cache
            cached_result = await cache.get_embedding(text, model, config)
            if cached_result is not None:
                logger.debug(f"Cache hit for embedding: {model}")
                return cached_result

        # Generate embedding
        result = await self.embedding_client.generate_embedding(text, model, config)

        # Cache the result
        if cache:
            await cache.set_embedding(text, model, result, config)
            logger.debug(f"Cached embedding result: {model}")

        return result

    async def generate_embeddings_batch(
        self, texts: list[str], model: str, config: dict[str, Any] | None = None
    ) -> list[Any]:
        """Generate embeddings batch with caching."""
        cache = await self._get_cache("embedding")
        results = []
        uncached_texts = []
        uncached_indices = []

        if cache:
            # Check cache for each text
            for i, text in enumerate(texts):
                cached_result = await cache.get_embedding(text, model, config)
                if cached_result is not None:
                    results.append(cached_result)
                    logger.debug(f"Cache hit for embedding batch item {i}: {model}")
                else:
                    results.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            # No cache, process all texts
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            results = [None] * len(texts)

        # Generate embeddings for uncached texts
        if uncached_texts:
            uncached_results = await self.embedding_client.generate_embeddings_batch(
                uncached_texts, model, config
            )

            # Cache results and update results list
            if cache:
                for i, result in enumerate(uncached_results):
                    text = uncached_texts[i]
                    await cache.set_embedding(text, model, result, config)
                    logger.debug(f"Cached embedding batch result {i}: {model}")

            # Update results list
            for i, result in zip(uncached_indices, uncached_results, strict=False):
                results[i] = result

        return results

    async def invalidate_embedding_cache(self, model: str | None = None) -> None:
        """Invalidate embedding cache."""
        cache = await self._get_cache("embedding")
        if cache:
            await cache.invalidate_embeddings(model)
            logger.info(f"Invalidated embedding cache for model: {model}")


class CacheIntegratedRerankingClient(CacheIntegratedClient):
    """Cache-integrated reranking client."""

    def __init__(self, reranking_client, cache_config: CacheIntegrationConfig | None = None):
        super().__init__(cache_config)
        self.reranking_client = reranking_client

    async def rerank_batch(
        self, query: str, documents: list[str], model: str, config: dict[str, Any] | None = None
    ) -> Any:
        """Rerank documents with caching."""
        cache = await self._get_cache("reranking")

        if cache:
            # Try to get from cache
            cached_result = await cache.get_reranking(query, documents, model, config)
            if cached_result is not None:
                logger.debug(f"Cache hit for reranking: {model}")
                return cached_result

        # Perform reranking
        result = await self.reranking_client.rerank_batch(query, documents, model, config)

        # Cache the result
        if cache:
            await cache.set_reranking(query, documents, model, result, config)
            logger.debug(f"Cached reranking result: {model}")

        return result

    async def rerank_multiple_batches(
        self,
        queries: list[str],
        documents_list: list[list[str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Rerank multiple batches with caching."""
        cache = await self._get_cache("reranking")
        results = []
        uncached_queries = []
        uncached_documents = []
        uncached_indices = []

        if cache:
            # Check cache for each query-documents pair
            for i, (query, documents) in enumerate(zip(queries, documents_list, strict=False)):
                cached_result = await cache.get_reranking(query, documents, model, config)
                if cached_result is not None:
                    results.append(cached_result)
                    logger.debug(f"Cache hit for reranking batch item {i}: {model}")
                else:
                    results.append(None)
                    uncached_queries.append(query)
                    uncached_documents.append(documents)
                    uncached_indices.append(i)
        else:
            # No cache, process all queries
            uncached_queries = queries
            uncached_documents = documents_list
            uncached_indices = list(range(len(queries)))
            results = [None] * len(queries)

        # Perform reranking for uncached queries
        if uncached_queries:
            uncached_results = await self.reranking_client.rerank_multiple_batches(
                uncached_queries, uncached_documents, model, config
            )

            # Cache results and update results list
            if cache:
                for i, result in enumerate(uncached_results):
                    query = uncached_queries[i]
                    documents = uncached_documents[i]
                    await cache.set_reranking(query, documents, model, result, config)
                    logger.debug(f"Cached reranking batch result {i}: {model}")

            # Update results list
            for i, result in zip(uncached_indices, uncached_results, strict=False):
                results[i] = result

        return results

    async def invalidate_reranking_cache(self, model: str | None = None) -> None:
        """Invalidate reranking cache."""
        cache = await self._get_cache("reranking")
        if cache:
            await cache.invalidate_reranking(model)
            logger.info(f"Invalidated reranking cache for model: {model}")


class CacheIntegratedDoclingVLMClient(CacheIntegratedClient):
    """Cache-integrated Docling VLM client."""

    def __init__(self, docling_vlm_client, cache_config: CacheIntegrationConfig | None = None):
        super().__init__(cache_config)
        self.docling_vlm_client = docling_vlm_client

    async def process_pdf(self, pdf_content: bytes, config: dict[str, Any] | None = None) -> Any:
        """Process PDF with caching."""
        cache = await self._get_cache("docling_vlm")

        if cache:
            # Try to get from cache
            cached_result = await cache.get_processing_result(pdf_content, config)
            if cached_result is not None:
                logger.debug("Cache hit for VLM processing")
                return cached_result

        # Process PDF
        result = await self.docling_vlm_client.process_pdf(pdf_content, config)

        # Cache the result
        if cache:
            await cache.set_processing_result(pdf_content, result, config)
            logger.debug("Cached VLM processing result")

        return result

    async def process_pdf_batch(
        self, pdf_contents: list[bytes], config: dict[str, Any] | None = None
    ) -> list[Any]:
        """Process PDF batch with caching."""
        cache = await self._get_cache("docling_vlm")
        results = []
        uncached_contents = []
        uncached_indices = []

        if cache:
            # Check cache for each PDF
            for i, pdf_content in enumerate(pdf_contents):
                cached_result = await cache.get_processing_result(pdf_content, config)
                if cached_result is not None:
                    results.append(cached_result)
                    logger.debug(f"Cache hit for VLM processing batch item {i}")
                else:
                    results.append(None)
                    uncached_contents.append(pdf_content)
                    uncached_indices.append(i)
        else:
            # No cache, process all PDFs
            uncached_contents = pdf_contents
            uncached_indices = list(range(len(pdf_contents)))
            results = [None] * len(pdf_contents)

        # Process uncached PDFs
        if uncached_contents:
            uncached_results = await self.docling_vlm_client.process_pdf_batch(
                uncached_contents, config
            )

            # Cache results and update results list
            if cache:
                for i, result in enumerate(uncached_results):
                    pdf_content = uncached_contents[i]
                    await cache.set_processing_result(pdf_content, result, config)
                    logger.debug(f"Cached VLM processing batch result {i}")

            # Update results list
            for i, result in zip(uncached_indices, uncached_results, strict=False):
                results[i] = result

        return results

    async def invalidate_vlm_cache(self) -> None:
        """Invalidate VLM processing cache."""
        cache = await self._get_cache("docling_vlm")
        if cache:
            await cache.invalidate_processing_results()
            logger.info("Invalidated VLM processing cache")


def create_cache_integrated_clients(
    embedding_client=None,
    reranking_client=None,
    docling_vlm_client=None,
    cache_config: CacheIntegrationConfig | None = None,
) -> dict[str, Any]:
    """Create cache-integrated clients for all services."""
    clients = {}

    if embedding_client:
        clients["embedding"] = CacheIntegratedEmbeddingClient(embedding_client, cache_config)

    if reranking_client:
        clients["reranking"] = CacheIntegratedRerankingClient(reranking_client, cache_config)

    if docling_vlm_client:
        clients["docling_vlm"] = CacheIntegratedDoclingVLMClient(docling_vlm_client, cache_config)

    return clients


async def cleanup_cache_integration(clients: dict[str, Any]) -> None:
    """Cleanup cache integration for all clients."""
    for client in clients.values():
        if hasattr(client, "close_caches"):
            await client.close_caches()
