"""SPLADE aggregation for document retrieval."""

from __future__ import annotations

import logging
import time
from typing import Any

import structlog

from Medical_KG_rev.services.retrieval.splade_service import SPLADESegment, SPLADEVector

logger = structlog.get_logger(__name__)


class SPLADEAggregator:
    """Aggregator for SPLADE vectors."""

    def __init__(self, sparsity_threshold: float = 0.01) -> None:
        """Initialize the SPLADE aggregator."""
        self.logger = logger
        self.sparsity_threshold = sparsity_threshold

    def aggregate_segments(
        self,
        segment_vectors: list[SPLADEVector],
        chunk_id: str,
        aggregation_method: str = "max",
    ) -> SPLADEVector:
        """Aggregate segment vectors into a single vector."""
        start_time = time.perf_counter()

        try:
            if not segment_vectors:
                self.logger.warning(
                    "No segment vectors provided for aggregation", extra={"chunk_id": chunk_id}
                )
                return SPLADEVector(
                    terms={},
                    tokenizer_name="naver/splade-v3",
                    model_name="naver/splade-v3",
                    sparsity_threshold=self.sparsity_threshold,
                )

            # Get the first vector as reference
            reference_vector = segment_vectors[0]

            # Aggregate terms
            aggregated_terms = {}

            if aggregation_method == "max":
                aggregated_terms = self._aggregate_max(segment_vectors)
            elif aggregation_method == "mean":
                aggregated_terms = self._aggregate_mean(segment_vectors)
            elif aggregation_method == "sum":
                aggregated_terms = self._aggregate_sum(segment_vectors)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

            # Apply sparsity threshold
            filtered_terms = {
                term: score for term, score in aggregated_terms.items()
                if score >= self.sparsity_threshold
            }

            duration_ms = (time.perf_counter() - start_time) * 1000

            self.logger.debug(
                "splade.aggregation.completed",
                chunk_id=chunk_id,
                segment_count=len(segment_vectors),
                aggregated_terms=len(aggregated_terms),
                filtered_terms=len(filtered_terms),
                duration_ms=duration_ms,
                aggregation_method=aggregation_method,
            )

            return SPLADEVector(
                terms=filtered_terms,
                tokenizer_name=reference_vector.tokenizer_name,
                model_name=reference_vector.model_name,
                sparsity_threshold=self.sparsity_threshold,
            )

        except Exception as e:
            self.logger.error(f"SPLADE aggregation failed for chunk {chunk_id}: {e}")
            raise

    def _aggregate_max(self, segment_vectors: list[SPLADEVector]) -> dict[str, float]:
        """Aggregate using maximum values."""
        aggregated = {}

        for vector in segment_vectors:
            for term, score in vector.terms.items():
                if term not in aggregated:
                    aggregated[term] = score
                else:
                    aggregated[term] = max(aggregated[term], score)

        return aggregated

    def _aggregate_mean(self, segment_vectors: list[SPLADEVector]) -> dict[str, float]:
        """Aggregate using mean values."""
        aggregated = {}
        term_counts = {}

        for vector in segment_vectors:
            for term, score in vector.terms.items():
                if term not in aggregated:
                    aggregated[term] = score
                    term_counts[term] = 1
                else:
                    aggregated[term] += score
                    term_counts[term] += 1

        # Calculate means
        for term in aggregated:
            aggregated[term] /= term_counts[term]

        return aggregated

    def _aggregate_sum(self, segment_vectors: list[SPLADEVector]) -> dict[str, float]:
        """Aggregate using sum values."""
        aggregated = {}

        for vector in segment_vectors:
            for term, score in vector.terms.items():
                if term not in aggregated:
                    aggregated[term] = score
                else:
                    aggregated[term] += score

        return aggregated

    def aggregate_documents(
        self,
        document_vectors: list[SPLADEVector],
        document_id: str,
        aggregation_method: str = "max",
    ) -> SPLADEVector:
        """Aggregate document vectors."""
        start_time = time.perf_counter()

        try:
            if not document_vectors:
                self.logger.warning(
                    "No document vectors provided for aggregation", extra={"document_id": document_id}
                )
                return SPLADEVector(
                    terms={},
                    tokenizer_name="naver/splade-v3",
                    model_name="naver/splade-v3",
                    sparsity_threshold=self.sparsity_threshold,
                )

            # Get the first vector as reference
            reference_vector = document_vectors[0]

            # Aggregate terms
            aggregated_terms = {}

            if aggregation_method == "max":
                aggregated_terms = self._aggregate_max(document_vectors)
            elif aggregation_method == "mean":
                aggregated_terms = self._aggregate_mean(document_vectors)
            elif aggregation_method == "sum":
                aggregated_terms = self._aggregate_sum(document_vectors)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

            # Apply sparsity threshold
            filtered_terms = {
                term: score for term, score in aggregated_terms.items()
                if score >= self.sparsity_threshold
            }

            duration_ms = (time.perf_counter() - start_time) * 1000

            self.logger.debug(
                "splade.document.aggregation.completed",
                document_id=document_id,
                document_count=len(document_vectors),
                aggregated_terms=len(aggregated_terms),
                filtered_terms=len(filtered_terms),
                duration_ms=duration_ms,
                aggregation_method=aggregation_method,
            )

            return SPLADEVector(
                terms=filtered_terms,
                tokenizer_name=reference_vector.tokenizer_name,
                model_name=reference_vector.model_name,
                sparsity_threshold=self.sparsity_threshold,
            )

        except Exception as e:
            self.logger.error(f"SPLADE document aggregation failed for document {document_id}: {e}")
            raise

    def health_check(self) -> dict[str, Any]:
        """Check aggregator health."""
        return {
            "aggregator": "splade",
            "status": "healthy",
            "sparsity_threshold": self.sparsity_threshold,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "sparsity_threshold": self.sparsity_threshold,
        }


class SPLADEAggregatorFactory:
    """Factory for creating SPLADE aggregators."""

    @staticmethod
    def create(sparsity_threshold: float = 0.01) -> SPLADEAggregator:
        """Create a SPLADE aggregator instance."""
        return SPLADEAggregator(sparsity_threshold)

    @staticmethod
    def create_with_config(config: dict[str, Any]) -> SPLADEAggregator:
        """Create a SPLADE aggregator with configuration."""
        sparsity_threshold = config.get("sparsity_threshold", 0.01)
        return SPLADEAggregator(sparsity_threshold)


# Global SPLADE aggregator instance
_splade_aggregator: SPLADEAggregator | None = None


def get_splade_aggregator() -> SPLADEAggregator:
    """Get the global SPLADE aggregator instance."""
    global _splade_aggregator

    if _splade_aggregator is None:
        _splade_aggregator = SPLADEAggregatorFactory.create()

    return _splade_aggregator


def create_splade_aggregator(sparsity_threshold: float = 0.01) -> SPLADEAggregator:
    """Create a new SPLADE aggregator instance."""
    return SPLADEAggregatorFactory.create(sparsity_threshold)
