"""Hybrid Retrieval Service.

This module implements a hybrid retrieval system that combines BM25, SPLADE-v3,
and Qwen3 dense retrieval strategies for improved accuracy and coverage.
"""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from Medical_KG_rev.config.retrieval_config import BM25Config, Qwen3Config, SPLADEConfig
from Medical_KG_rev.services.retrieval.bm25_service import BM25Service
from Medical_KG_rev.services.retrieval.qwen3_service import Qwen3Service
from Medical_KG_rev.services.retrieval.splade_service import SPLADEService
from Medical_KG_rev.services.vector_store.stores.bm25_index import BM25Index
from Medical_KG_rev.services.vector_store.stores.qwen3_index import Qwen3Index
from Medical_KG_rev.services.vector_store.stores.splade_index import SPLADEImpactIndex
from Medical_KG_rev.storage.chunk_store import ChunkStore

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """Single retrieval result with metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., description="Relevance score")
    method: str = Field(..., description="Retrieval method that produced this result")
    chunk_text: str = Field(..., description="Chunk text content")
    section_path: str | None = Field(None, description="Document section path")
    page_number: int | None = Field(None, description="Page number")
    bbox: dict[str, float] | None = Field(None, description="Bounding box coordinates")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HybridRetrievalResult(BaseModel):
    """Hybrid retrieval result with fused scores."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    fused_score: float = Field(..., description="Fused relevance score")
    individual_scores: dict[str, float] = Field(..., description="Scores from each method")
    chunk_text: str = Field(..., description="Chunk text content")
    section_path: str | None = Field(None, description="Document section path")
    page_number: int | None = Field(None, description="Page number")
    bbox: dict[str, float] | None = Field(None, description="Bounding box coordinates")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HybridRetrievalService:
    """Hybrid retrieval service combining BM25, SPLADE, and Qwen3 strategies.

    This service implements parallel retrieval across multiple strategies and
    fuses results using reciprocal rank fusion (RRF) for improved accuracy.
    """

    def __init__(
        self,
        bm25_config: BM25Config,
        splade_config: SPLADEConfig,
        qwen3_config: Qwen3Config,
        chunk_store: ChunkStore,
        bm25_index: BM25Index,
        splade_index: SPLADEImpactIndex,
        qwen3_index: Qwen3Index,
        fusion_k: int = 60,
        max_results: int = 100,
    ):
        """Initialize the hybrid retrieval service.

        Args:
            bm25_config: BM25 configuration
            splade_config: SPLADE configuration
            qwen3_config: Qwen3 configuration
            chunk_store: Chunk store for retrieving chunk details
            bm25_index: BM25 index for lexical retrieval
            splade_index: SPLADE impact index for sparse retrieval
            qwen3_index: Qwen3 index for dense retrieval
            fusion_k: RRF fusion parameter (higher = more weight to top results)
            max_results: Maximum number of results to return

        """
        self.bm25_config = bm25_config
        self.splade_config = splade_config
        self.qwen3_config = qwen3_config
        self.chunk_store = chunk_store
        self.bm25_index = bm25_index
        self.splade_index = splade_index
        self.qwen3_index = qwen3_index
        self.fusion_k = fusion_k
        self.max_results = max_results

        # Initialize retrieval services
        self.bm25_service: Any = None
        self.splade_service: Any = None
        self.qwen3_service: Any = None

        logger.info("HybridRetrievalService initialized")

    async def initialize(self) -> None:
        """Initialize all retrieval services."""
        try:
            # Initialize BM25 service
            self.bm25_service = BM25Service(
                field_boosts=getattr(self.bm25_config, "field_boosts", {}),
                k1=getattr(self.bm25_config, "k1", 1.2),
                b=getattr(self.bm25_config, "b", 0.75),
                delta=getattr(self.bm25_config, "delta", 0.0),
            )

            # Initialize SPLADE service
            self.splade_service = SPLADEService(
                model_name=getattr(self.splade_config, "model_name", "naver/splade-v3"),
                sparsity_threshold=getattr(self.splade_config, "sparsity_threshold", 0.1),
            )

            # Initialize Qwen3 service
            self.qwen3_service = Qwen3Service(
                model_name=getattr(self.qwen3_config, "model_name", "Qwen/Qwen2.5-7B-Instruct"),
                batch_size=getattr(self.qwen3_config, "batch_size", 8),
            )

            logger.info("All retrieval services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize retrieval services: {e}")
            raise

    async def search(
        self,
        query: str,
        k: int = 10,
        methods: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridRetrievalResult]:
        """Perform hybrid search across multiple retrieval strategies.

        Args:
            query: Search query text
            k: Number of results to return
            methods: List of methods to use (default: all)
            filters: Additional filters to apply

        Returns:
            List of hybrid retrieval results with fused scores

        """
        if methods is None:
            methods = ["bm25", "splade", "qwen3"]

        logger.info(f"Performing hybrid search for query: '{query[:100]}...'")

        try:
            # Execute parallel retrieval across methods
            retrieval_tasks = []

            if "bm25" in methods:
                retrieval_tasks.append(self._bm25_search(query, k))

            if "splade" in methods:
                retrieval_tasks.append(self._splade_search(query, k))

            if "qwen3" in methods:
                retrieval_tasks.append(self._qwen3_search(query, k))

            # Wait for all retrieval methods to complete
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

            # Process results and handle exceptions
            method_results: dict[str, list[RetrievalResult]] = {}
            for i, result in enumerate(retrieval_results):
                method = methods[i]
                if isinstance(result, Exception):
                    logger.error(f"Retrieval method {method} failed: {result}")
                    method_results[method] = []
                else:
                    method_results[method] = result

            # Fuse results using reciprocal rank fusion
            fused_results = self._fuse_results(method_results, k)

            # Apply filters if provided
            if filters:
                fused_results = self._apply_filters(fused_results, filters)

            # Retrieve chunk details for results
            enriched_results = await self._enrich_results(fused_results)

            logger.info(f"Hybrid search completed, returning {len(enriched_results)} results")
            return enriched_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    async def _bm25_search(self, query: str, k: int) -> list[RetrievalResult]:
        """Perform BM25 search."""
        try:
            results = self.bm25_service.search(query, k)
            return [
                RetrievalResult(
                    chunk_id=result["chunk_id"],
                    score=result["score"],
                    method="bm25",
                    chunk_text="",  # Will be filled by enrichment
                    section_path=None,
                    page_number=None,
                    bbox=None,
                    metadata=result.get("metadata", {}),
                )
                for result in results
            ]
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    async def _splade_search(self, query: str, k: int) -> list[RetrievalResult]:
        """Perform SPLADE search."""
        try:
            results = self.splade_service.search(query, k)
            return [
                RetrievalResult(
                    chunk_id=result["chunk_id"],
                    score=result["score"],
                    method="splade",
                    chunk_text="",  # Will be filled by enrichment
                    section_path=None,
                    page_number=None,
                    bbox=None,
                    metadata=result.get("metadata", {}),
                )
                for result in results
            ]
        except Exception as e:
            logger.error(f"SPLADE search failed: {e}")
            return []

    async def _qwen3_search(self, query: str, k: int) -> list[RetrievalResult]:
        """Perform Qwen3 dense search."""
        try:
            results = self.qwen3_service.search(query, k)
            return [
                RetrievalResult(
                    chunk_id=result["chunk_id"],
                    score=result["score"],
                    method="qwen3",
                    chunk_text="",  # Will be filled by enrichment
                    section_path=None,
                    page_number=None,
                    bbox=None,
                    metadata=result.get("metadata", {}),
                )
                for result in results
            ]
        except Exception as e:
            logger.error(f"Qwen3 search failed: {e}")
            return []

    def _fuse_results(
        self,
        method_results: dict[str, list[RetrievalResult]],
        k: int,
    ) -> list[tuple[str, float, dict[str, float]]]:
        """Fuse results using reciprocal rank fusion (RRF).

        Args:
            method_results: Results from each retrieval method
            k: Number of results to return

        Returns:
            List of (chunk_id, fused_score, individual_scores) tuples

        """
        # Collect all unique chunk IDs
        all_chunk_ids = set()
        for results in method_results.values():
            for result in results:
                all_chunk_ids.add(result.chunk_id)

        # Calculate RRF scores
        rrf_scores = {}
        individual_scores = {}

        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            chunk_individual_scores = {}

            for method, results in method_results.items():
                # Find rank of this chunk in this method's results
                rank = None
                score = 0.0

                for i, result in enumerate(results):
                    if result.chunk_id == chunk_id:
                        rank = i + 1  # 1-based ranking
                        score = result.score
                        break

                if rank is not None:
                    # RRF formula: 1 / (k + rank)
                    rrf_score += 1.0 / (self.fusion_k + rank)
                    chunk_individual_scores[method] = score
                else:
                    chunk_individual_scores[method] = 0.0

            rrf_scores[chunk_id] = rrf_score
            individual_scores[chunk_id] = chunk_individual_scores

        # Sort by RRF score and return top k
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            (chunk_id, score, individual_scores[chunk_id]) for chunk_id, score in sorted_results
        ]

    def _apply_filters(
        self,
        results: list[tuple[str, float, dict[str, float]]],
        filters: dict[str, Any],
    ) -> list[tuple[str, float, dict[str, float]]]:
        """Apply filters to results."""
        # TODO: Implement filtering logic based on filters
        # For now, return results as-is
        return results

    async def _enrich_results(
        self,
        results: list[tuple[str, float, dict[str, float]]],
    ) -> list[HybridRetrievalResult]:
        """Enrich results with chunk details from chunk store."""
        enriched_results = []

        for chunk_id, fused_score, individual_scores in results:
            try:
                # Retrieve chunk details from chunk store
                chunk = self.chunk_store.get_chunk(chunk_id)

                if chunk:
                    enriched_result = HybridRetrievalResult(
                        chunk_id=chunk_id,
                        fused_score=fused_score,
                        individual_scores=individual_scores,
                        chunk_text=getattr(chunk, "contextualized_text", "")
                        or getattr(chunk, "content_only_text", ""),
                        section_path=getattr(chunk, "section_path", None),
                        page_number=getattr(chunk, "page_number", None),
                        bbox=getattr(chunk, "bbox", None),
                        metadata=getattr(chunk, "metadata", {}) or {},
                    )
                    enriched_results.append(enriched_result)
                else:
                    logger.warning(f"Chunk {chunk_id} not found in chunk store")

            except Exception as e:
                logger.error(f"Failed to enrich result for chunk {chunk_id}: {e}")
                continue

        return enriched_results

    async def health_check(self) -> dict[str, Any]:
        """Check health of all retrieval services."""
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": None,
        }

        try:
            # Check BM25 service
            if self.bm25_service:
                bm25_health = self.bm25_service.health_check()
                health_status["components"]["bm25"] = bm25_health

            # Check SPLADE service
            if self.splade_service:
                splade_health = self.splade_service.health_check()
                health_status["components"]["splade"] = splade_health

            # Check Qwen3 service
            if self.qwen3_service:
                qwen3_health = self.qwen3_service.health_check()
                health_status["components"]["qwen3"] = qwen3_health

            # Check chunk store
            chunk_store_health = self.chunk_store.health_check()
            health_status["components"]["chunk_store"] = chunk_store_health

            # Determine overall status
            component_statuses = []
            for comp in health_status["components"].values():
                if isinstance(comp, dict):
                    component_statuses.append(comp.get("status", "unknown"))
                else:
                    component_statuses.append("unknown")

            if "unhealthy" in component_statuses:
                health_status["status"] = "unhealthy"
            elif "degraded" in component_statuses:
                health_status["status"] = "degraded"

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the hybrid retrieval service."""
        stats = {
            "service": "hybrid_retrieval",
            "methods": ["bm25", "splade", "qwen3"],
            "fusion_k": self.fusion_k,
            "max_results": self.max_results,
            "components": {},
        }

        try:
            # Get BM25 stats
            if self.bm25_service:
                stats["components"]["bm25"] = self.bm25_service.get_stats()

            # Get SPLADE stats
            if self.splade_service:
                stats["components"]["splade"] = self.splade_service.get_stats()

            # Get Qwen3 stats
            if self.qwen3_service:
                stats["components"]["qwen3"] = self.qwen3_service.get_stats()

            # Get chunk store stats
            if hasattr(self.chunk_store, "get_stats"):
                stats["components"]["chunk_store"] = self.chunk_store.get_stats()
            else:
                stats["components"]["chunk_store"] = {"status": "unknown"}

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            stats["error"] = str(e)

        return stats
