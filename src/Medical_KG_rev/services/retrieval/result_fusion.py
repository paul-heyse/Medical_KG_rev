"""Fuse retrieval results from lexical, sparse, and dense components.

Key Responsibilities:
    - Provide configurable result fusion strategies (RRF, weighted, comb methods).
    - Normalize and threshold scores across heterogeneous retrieval outputs.
    - Produce ranked `FusionResult` objects with contribution metadata.

Collaborators:
    - Upstream: Retrieval components producing method-specific results.
    - Downstream: Rerankers, gateways, or orchestration pipelines consuming fused rankings.

Side Effects:
    - Emits structured logs; otherwise pure in-memory processing.

Thread Safety:
    - Thread-safe: Does not maintain mutable shared state beyond configuration.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import logging
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class FusionResult(BaseModel):
    """Single retrieval result after fusion across multiple methods.

    Attributes:
        chunk_id: Unique identifier of the retrieved chunk.
        fused_score: Combined relevance score after fusion.
        individual_scores: Scores contributed by each retrieval strategy.
        rank: Final ranking position assigned post-fusion.
        method_contributions: Per-method contribution values (e.g., RRF weights).
        metadata: Additional metadata associated with the chunk.
    """

    chunk_id: str = Field(..., description="Chunk identifier")
    fused_score: float = Field(..., description="Fused relevance score")
    individual_scores: dict[str, float] = Field(..., description="Scores from each method")
    rank: int = Field(..., description="Final rank after fusion")
    method_contributions: dict[str, float] = Field(
        default_factory=dict, description="Contribution from each method"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Result metadata")


class FusionConfig(BaseModel):
    """Configuration options controlling result fusion behavior."""

    fusion_method: str = Field(
        default="rrf", description="Fusion method (rrf, weighted, comb_sum, comb_max)"
    )
    fusion_k: int = Field(default=60, description="RRF fusion parameter")
    method_weights: dict[str, float] = Field(
        default_factory=dict, description="Weights for each method"
    )
    normalize_scores: bool = Field(default=True, description="Normalize scores before fusion")
    min_score_threshold: float = Field(default=0.0, description="Minimum score threshold")
    max_results: int = Field(default=100, description="Maximum number of results to return")


class ResultFusion:
    """Combine retrieval method outputs into a unified ranking.

    Attributes:
        config: :class:`FusionConfig` controlling fusion behavior.
    """

    def __init__(self, config: FusionConfig):
        """Initialize the result fusion system.

        Args:
            config: Fusion configuration describing algorithms and weights.
        """
        self.config = config

        # Set default method weights if not provided
        if not self.config.method_weights:
            self.config.method_weights = {
                "bm25": 0.33,
                "splade": 0.33,
                "qwen3": 0.34,
            }

        logger.info(f"ResultFusion initialized with method: {self.config.fusion_method}")

    def fuse_results(
        self,
        method_results: dict[str, list[tuple[str, float]]],
        k: int = 10,
    ) -> list[FusionResult]:
        """Fuse results from multiple retrieval methods.

        Args:
            method_results: Mapping of method name to pairs of ``(chunk_id, score)``.
            k: Maximum number of fused results to return.

        Returns:
            List of :class:`FusionResult` objects sorted by fused score.
        """
        logger.info(f"Fusing results from {len(method_results)} methods")

        try:
            # Normalize scores if configured
            if self.config.normalize_scores:
                method_results = self._normalize_scores(method_results)

            # Apply fusion method
            if self.config.fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(method_results, k)
            elif self.config.fusion_method == "weighted":
                fused_results = self._weighted_fusion(method_results, k)
            elif self.config.fusion_method == "comb_sum":
                fused_results = self._comb_sum_fusion(method_results, k)
            elif self.config.fusion_method == "comb_max":
                fused_results = self._comb_max_fusion(method_results, k)
            else:
                raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")

            # Apply minimum score threshold
            fused_results = self._apply_score_threshold(fused_results)

            # Limit results
            fused_results = fused_results[:k]

            # Add ranks
            for i, result in enumerate(fused_results):
                result.rank = i + 1

            logger.info(f"Fusion completed, returning {len(fused_results)} results")
            return fused_results

        except Exception as e:
            logger.error(f"Failed to fuse results: {e}")
            raise

    def _normalize_scores(
        self,
        method_results: dict[str, list[tuple[str, float]]],
    ) -> dict[str, list[tuple[str, float]]]:
        """Normalize scores within each method to the ``[0, 1]`` range."""
        normalized_results = {}

        for method, results in method_results.items():
            if not results:
                normalized_results[method] = []
                continue

            # Find min and max scores
            scores = [score for _, score in results]
            min_score = min(scores)
            max_score = max(scores)

            # Normalize scores to [0, 1] range
            if max_score > min_score:
                normalized_results[method] = [
                    (chunk_id, (score - min_score) / (max_score - min_score))
                    for chunk_id, score in results
                ]
            else:
                # All scores are the same
                normalized_results[method] = [(chunk_id, 1.0) for chunk_id, _ in results]

        return normalized_results

    def _reciprocal_rank_fusion(
        self,
        method_results: dict[str, list[tuple[str, float]]],
        k: int,
    ) -> list[FusionResult]:
        """Implement reciprocal rank fusion (RRF)."""
        # Collect all unique chunk IDs
        all_chunk_ids = set()
        for results in method_results.values():
            for chunk_id, _ in results:
                all_chunk_ids.add(chunk_id)

        # Calculate RRF scores
        rrf_scores = {}
        individual_scores = {}
        method_contributions = {}

        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            chunk_individual_scores = {}
            chunk_contributions = {}

            for method, results in method_results.items():
                # Find rank of this chunk in this method's results
                rank = None
                score = 0.0

                for i, (result_chunk_id, result_score) in enumerate(results):
                    if result_chunk_id == chunk_id:
                        rank = i + 1  # 1-based ranking
                        score = result_score
                        break

                if rank is not None:
                    # RRF formula: 1 / (k + rank)
                    contribution = 1.0 / (self.config.fusion_k + rank)
                    rrf_score += contribution
                    chunk_individual_scores[method] = score
                    chunk_contributions[method] = contribution
                else:
                    chunk_individual_scores[method] = 0.0
                    chunk_contributions[method] = 0.0

            rrf_scores[chunk_id] = rrf_score
            individual_scores[chunk_id] = chunk_individual_scores
            method_contributions[chunk_id] = chunk_contributions

        # Create fusion results
        fusion_results = []
        for chunk_id, fused_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            fusion_result = FusionResult(
                chunk_id=chunk_id,
                fused_score=fused_score,
                individual_scores=individual_scores[chunk_id],
                rank=0,  # Will be set later
                method_contributions=method_contributions[chunk_id],
                metadata={"fusion_method": "rrf", "fusion_k": self.config.fusion_k},
            )
            fusion_results.append(fusion_result)

        return fusion_results

    def _weighted_fusion(
        self,
        method_results: dict[str, list[tuple[str, float]]],
        k: int,
    ) -> list[FusionResult]:
        """Implement weighted fusion."""
        # Collect all unique chunk IDs
        all_chunk_ids = set()
        for results in method_results.values():
            for chunk_id, _ in results:
                all_chunk_ids.add(chunk_id)

        # Calculate weighted scores
        weighted_scores = {}
        individual_scores = {}
        method_contributions = {}

        for chunk_id in all_chunk_ids:
            weighted_score = 0.0
            chunk_individual_scores = {}
            chunk_contributions = {}

            for method, results in method_results.items():
                # Find score of this chunk in this method's results
                score = 0.0
                for result_chunk_id, result_score in results:
                    if result_chunk_id == chunk_id:
                        score = result_score
                        break

                weight = self.config.method_weights.get(method, 0.0)
                contribution = score * weight
                weighted_score += contribution

                chunk_individual_scores[method] = score
                chunk_contributions[method] = contribution

            weighted_scores[chunk_id] = weighted_score
            individual_scores[chunk_id] = chunk_individual_scores
            method_contributions[chunk_id] = chunk_contributions

        # Create fusion results
        fusion_results = []
        for chunk_id, fused_score in sorted(
            weighted_scores.items(), key=lambda x: x[1], reverse=True
        ):
            fusion_result = FusionResult(
                chunk_id=chunk_id,
                fused_score=fused_score,
                individual_scores=individual_scores[chunk_id],
                rank=0,  # Will be set later
                method_contributions=method_contributions[chunk_id],
                metadata={
                    "fusion_method": "weighted",
                    "method_weights": self.config.method_weights,
                },
            )
            fusion_results.append(fusion_result)

        return fusion_results

    def _comb_sum_fusion(
        self,
        method_results: dict[str, list[tuple[str, float]]],
        k: int,
    ) -> list[FusionResult]:
        """Implement combination sum fusion."""
        # Collect all unique chunk IDs
        all_chunk_ids = set()
        for results in method_results.values():
            for chunk_id, _ in results:
                all_chunk_ids.add(chunk_id)

        # Calculate sum scores
        sum_scores = {}
        individual_scores = {}
        method_contributions = {}

        for chunk_id in all_chunk_ids:
            sum_score = 0.0
            chunk_individual_scores = {}
            chunk_contributions = {}

            for method, results in method_results.items():
                # Find score of this chunk in this method's results
                score = 0.0
                for result_chunk_id, result_score in results:
                    if result_chunk_id == chunk_id:
                        score = result_score
                        break

                sum_score += score
                chunk_individual_scores[method] = score
                chunk_contributions[method] = score

            sum_scores[chunk_id] = sum_score
            individual_scores[chunk_id] = chunk_individual_scores
            method_contributions[chunk_id] = chunk_contributions

        # Create fusion results
        fusion_results = []
        for chunk_id, fused_score in sorted(sum_scores.items(), key=lambda x: x[1], reverse=True):
            fusion_result = FusionResult(
                chunk_id=chunk_id,
                fused_score=fused_score,
                individual_scores=individual_scores[chunk_id],
                rank=0,  # Will be set later
                method_contributions=method_contributions[chunk_id],
                metadata={"fusion_method": "comb_sum"},
            )
            fusion_results.append(fusion_result)

        return fusion_results

    def _comb_max_fusion(
        self,
        method_results: dict[str, list[tuple[str, float]]],
        k: int,
    ) -> list[FusionResult]:
        """Implement combination max fusion."""
        # Collect all unique chunk IDs
        all_chunk_ids = set()
        for results in method_results.values():
            for chunk_id, _ in results:
                all_chunk_ids.add(chunk_id)

        # Calculate max scores
        max_scores = {}
        individual_scores = {}
        method_contributions = {}

        for chunk_id in all_chunk_ids:
            max_score = 0.0
            chunk_individual_scores = {}
            chunk_contributions = {}

            for method, results in method_results.items():
                # Find score of this chunk in this method's results
                score = 0.0
                for result_chunk_id, result_score in results:
                    if result_chunk_id == chunk_id:
                        score = result_score
                        break

                max_score = max(max_score, score)
                chunk_individual_scores[method] = score
                chunk_contributions[method] = score

            max_scores[chunk_id] = max_score
            individual_scores[chunk_id] = chunk_individual_scores
            method_contributions[chunk_id] = chunk_contributions

        # Create fusion results
        fusion_results = []
        for chunk_id, fused_score in sorted(max_scores.items(), key=lambda x: x[1], reverse=True):
            fusion_result = FusionResult(
                chunk_id=chunk_id,
                fused_score=fused_score,
                individual_scores=individual_scores[chunk_id],
                rank=0,  # Will be set later
                method_contributions=method_contributions[chunk_id],
                metadata={"fusion_method": "comb_max"},
            )
            fusion_results.append(fusion_result)

        return fusion_results

    def _apply_score_threshold(self, results: list[FusionResult]) -> list[FusionResult]:
        """Apply minimum score threshold."""
        if self.config.min_score_threshold <= 0:
            return results

        filtered_results = [
            result for result in results if result.fused_score >= self.config.min_score_threshold
        ]

        return filtered_results

    def explain_fusion(self, fusion_results: list[FusionResult]) -> dict[str, Any]:
        """Explain the fusion process and results."""
        explanation = {
            "fusion_method": self.config.fusion_method,
            "fusion_k": self.config.fusion_k,
            "method_weights": self.config.method_weights,
            "normalize_scores": self.config.normalize_scores,
            "min_score_threshold": self.config.min_score_threshold,
            "num_results": len(fusion_results),
            "top_results": [
                {
                    "chunk_id": result.chunk_id,
                    "fused_score": result.fused_score,
                    "individual_scores": result.individual_scores,
                    "method_contributions": result.method_contributions,
                }
                for result in fusion_results[:5]
            ],
        }

        return explanation

    def get_fusion_stats(self) -> dict[str, Any]:
        """Get statistics about the fusion system."""
        stats = {
            "fusion_method": self.config.fusion_method,
            "fusion_k": self.config.fusion_k,
            "method_weights": self.config.method_weights,
            "normalize_scores": self.config.normalize_scores,
            "min_score_threshold": self.config.min_score_threshold,
            "max_results": self.config.max_results,
        }

        return stats
