"""SPLADE Rep-Max aggregation utilities.

This module implements Rep-Max aggregation for SPLADE vectors, which merges
segment vectors by taking the maximum weight per term to create one
learned-sparse vector per chunk.
"""

import logging

from Medical_KG_rev.services.retrieval.splade_service import SPLADESegment, SPLADEVector
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
SPLADE_AGGREGATION_SECONDS = Histogram(
    "splade_aggregation_seconds",
    "Time spent on SPLADE aggregation operations",
    ["operation", "status"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

SPLADE_AGGREGATION_OPERATIONS = Counter(
    "splade_aggregation_operations_total",
    "Total number of SPLADE aggregation operations",
    ["operation", "status"],
)

SPLADE_TERMS_PROCESSED = Counter(
    "splade_terms_processed_total", "Total number of terms processed in aggregation", ["operation"]
)


class SPLADEAggregator:
    """Aggregator for SPLADE vectors using Rep-Max aggregation.

    Rep-Max aggregation merges segment vectors by taking the maximum weight
    per term, creating one learned-sparse vector per chunk while preserving
    the learned sparsity patterns from the SPLADE model.
    """

    def __init__(
        self,
        sparsity_threshold: float = 0.01,
        quantization_scale: int = 1000,
        max_terms_per_chunk: int = 10000,
        preserve_term_order: bool = False,
    ):
        """Initialize SPLADE aggregator.

        Args:
            sparsity_threshold: Minimum weight threshold for sparsity control
            quantization_scale: Scale factor for weight quantization
            max_terms_per_chunk: Maximum number of terms to keep per chunk
            preserve_term_order: Whether to preserve term order in output

        """
        self.sparsity_threshold = sparsity_threshold
        self.quantization_scale = quantization_scale
        self.max_terms_per_chunk = max_terms_per_chunk
        self.preserve_term_order = preserve_term_order

        logger.info(
            "Initialized SPLADE aggregator",
            extra={
                "sparsity_threshold": self.sparsity_threshold,
                "quantization_scale": self.quantization_scale,
                "max_terms_per_chunk": self.max_terms_per_chunk,
                "preserve_term_order": self.preserve_term_order,
            },
        )

    def _extract_terms_from_segment(self, segment: SPLADESegment) -> dict[int, float]:
        """Extract terms from a segment (placeholder for actual SPLADE encoding).

        Args:
            segment: Segment to extract terms from

        Returns:
            Dictionary mapping term IDs to weights

        """
        # This is a placeholder - in real implementation, this would
        # call the SPLADE model to encode the segment
        # For now, return empty dict as segments don't contain encoded terms
        return {}

    def rep_max_aggregate(
        self,
        segment_vectors: list[SPLADEVector],
        chunk_id: str,
    ) -> SPLADEVector:
        """Perform Rep-Max aggregation on segment vectors.

        Rep-Max aggregation merges segment vectors by taking the maximum weight
        per term, creating one learned-sparse vector per chunk.

        Args:
            segment_vectors: List of SPLADE vectors from segments
            chunk_id: Identifier for the chunk

        Returns:
            Aggregated SPLADE vector

        """
        import time

        start_time = time.perf_counter()

        try:
            if not segment_vectors:
                logger.warning(
                    "No segment vectors provided for aggregation", extra={"chunk_id": chunk_id}
                )
                return SPLADEVector(
                    terms={},
                    tokenizer_name="naver/splade-v3",
                    model_name="naver/splade-v3",
                    sparsity_threshold=self.sparsity_threshold,
                    quantization_scale=self.quantization_scale,
                )

            # Rep-Max aggregation: take maximum weight per term
            aggregated_terms = {}
            total_terms_processed = 0

            for vector in segment_vectors:
                for term_id, weight in vector.terms.items():
                    total_terms_processed += 1

                    if term_id not in aggregated_terms:
                        aggregated_terms[term_id] = weight
                    else:
                        # Take maximum weight (Rep-Max)
                        aggregated_terms[term_id] = max(aggregated_terms[term_id], weight)

            # Apply sparsity threshold
            filtered_terms = {
                term_id: weight
                for term_id, weight in aggregated_terms.items()
                if weight >= self.sparsity_threshold
            }

            # Sort by weight and limit to max_terms_per_chunk
            if len(filtered_terms) > self.max_terms_per_chunk:
                sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
                filtered_terms = dict(sorted_terms[: self.max_terms_per_chunk])

            # Quantize weights
            quantized_terms = {}
            for term_id, weight in filtered_terms.items():
                quantized_weight = int(weight * self.quantization_scale)
                if quantized_weight > 0:  # Only keep positive quantized weights
                    quantized_terms[term_id] = quantized_weight

            # Create aggregated vector
            aggregated_vector = SPLADEVector(
                terms=quantized_terms,
                tokenizer_name=segment_vectors[0].tokenizer_name,
                model_name=segment_vectors[0].model_name,
                sparsity_threshold=self.sparsity_threshold,
                quantization_scale=self.quantization_scale,
            )

            processing_time = time.perf_counter() - start_time

            SPLADE_AGGREGATION_SECONDS.labels(operation="rep_max", status="ok").observe(
                processing_time
            )
            SPLADE_AGGREGATION_OPERATIONS.labels(operation="rep_max", status="ok").inc()
            SPLADE_TERMS_PROCESSED.labels(operation="rep_max").inc(total_terms_processed)

            logger.info(
                "Rep-Max aggregation completed",
                extra={
                    "chunk_id": chunk_id,
                    "input_vectors": len(segment_vectors),
                    "total_terms_processed": total_terms_processed,
                    "filtered_terms": len(filtered_terms),
                    "quantized_terms": len(quantized_terms),
                    "processing_time_seconds": processing_time,
                },
            )

            return aggregated_vector

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_AGGREGATION_SECONDS.labels(operation="rep_max", status="error").observe(
                processing_time
            )
            SPLADE_AGGREGATION_OPERATIONS.labels(operation="rep_max", status="error").inc()

            logger.error(
                "Rep-Max aggregation failed",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def score_max_aggregate(
        self,
        segment_vectors: list[SPLADEVector],
        chunk_id: str,
    ) -> SPLADEVector:
        """Perform Score-Max aggregation on segment vectors.

        Score-Max aggregation is an alternative to Rep-Max that takes the
        maximum score per term across segments.

        Args:
            segment_vectors: List of SPLADE vectors from segments
            chunk_id: Identifier for the chunk

        Returns:
            Aggregated SPLADE vector

        """
        import time

        start_time = time.perf_counter()

        try:
            if not segment_vectors:
                logger.warning(
                    "No segment vectors provided for aggregation", extra={"chunk_id": chunk_id}
                )
                return SPLADEVector(
                    terms={},
                    tokenizer_name="naver/splade-v3",
                    model_name="naver/splade-v3",
                    sparsity_threshold=self.sparsity_threshold,
                    quantization_scale=self.quantization_scale,
                )

            # Score-Max aggregation: take maximum score per term
            aggregated_terms = {}
            total_terms_processed = 0

            for vector in segment_vectors:
                for term_id, weight in vector.terms.items():
                    total_terms_processed += 1

                    if term_id not in aggregated_terms:
                        aggregated_terms[term_id] = weight
                    else:
                        # Take maximum score (Score-Max)
                        aggregated_terms[term_id] = max(aggregated_terms[term_id], weight)

            # Apply sparsity threshold
            filtered_terms = {
                term_id: weight
                for term_id, weight in aggregated_terms.items()
                if weight >= self.sparsity_threshold
            }

            # Sort by weight and limit to max_terms_per_chunk
            if len(filtered_terms) > self.max_terms_per_chunk:
                sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
                filtered_terms = dict(sorted_terms[: self.max_terms_per_chunk])

            # Quantize weights
            quantized_terms = {}
            for term_id, weight in filtered_terms.items():
                quantized_weight = int(weight * self.quantization_scale)
                if quantized_weight > 0:
                    quantized_terms[term_id] = quantized_weight

            # Create aggregated vector
            aggregated_vector = SPLADEVector(
                terms=quantized_terms,
                tokenizer_name=segment_vectors[0].tokenizer_name,
                model_name=segment_vectors[0].model_name,
                sparsity_threshold=self.sparsity_threshold,
                quantization_scale=self.quantization_scale,
            )

            processing_time = time.perf_counter() - start_time

            SPLADE_AGGREGATION_SECONDS.labels(operation="score_max", status="ok").observe(
                processing_time
            )
            SPLADE_AGGREGATION_OPERATIONS.labels(operation="score_max", status="ok").inc()
            SPLADE_TERMS_PROCESSED.labels(operation="score_max").inc(total_terms_processed)

            logger.info(
                "Score-Max aggregation completed",
                extra={
                    "chunk_id": chunk_id,
                    "input_vectors": len(segment_vectors),
                    "total_terms_processed": total_terms_processed,
                    "filtered_terms": len(filtered_terms),
                    "quantized_terms": len(quantized_terms),
                    "processing_time_seconds": processing_time,
                },
            )

            return aggregated_vector

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_AGGREGATION_SECONDS.labels(operation="score_max", status="error").observe(
                processing_time
            )
            SPLADE_AGGREGATION_OPERATIONS.labels(operation="score_max", status="error").inc()

            logger.error(
                "Score-Max aggregation failed",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def weighted_average_aggregate(
        self,
        segment_vectors: list[SPLADEVector],
        chunk_id: str,
        weights: list[float] | None = None,
    ) -> SPLADEVector:
        """Perform weighted average aggregation on segment vectors.

        Args:
            segment_vectors: List of SPLADE vectors from segments
            chunk_id: Identifier for the chunk
            weights: Optional weights for each segment (defaults to equal weights)

        Returns:
            Aggregated SPLADE vector

        """
        import time

        start_time = time.perf_counter()

        try:
            if not segment_vectors:
                logger.warning(
                    "No segment vectors provided for aggregation", extra={"chunk_id": chunk_id}
                )
                return SPLADEVector(
                    terms={},
                    tokenizer_name="naver/splade-v3",
                    model_name="naver/splade-v3",
                    sparsity_threshold=self.sparsity_threshold,
                    quantization_scale=self.quantization_scale,
                )

            # Use equal weights if not provided
            if weights is None:
                weights = [1.0] * len(segment_vectors)

            if len(weights) != len(segment_vectors):
                raise ValueError("Number of weights must match number of segment vectors")

            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1.0] * len(segment_vectors)
                total_weight = len(segment_vectors)

            normalized_weights = [w / total_weight for w in weights]

            # Weighted average aggregation
            aggregated_terms = {}
            total_terms_processed = 0

            for vector, weight in zip(segment_vectors, normalized_weights, strict=False):
                for term_id, term_weight in vector.terms.items():
                    total_terms_processed += 1

                    if term_id not in aggregated_terms:
                        aggregated_terms[term_id] = term_weight * weight
                    else:
                        aggregated_terms[term_id] += term_weight * weight

            # Apply sparsity threshold
            filtered_terms = {
                term_id: weight
                for term_id, weight in aggregated_terms.items()
                if weight >= self.sparsity_threshold
            }

            # Sort by weight and limit to max_terms_per_chunk
            if len(filtered_terms) > self.max_terms_per_chunk:
                sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
                filtered_terms = dict(sorted_terms[: self.max_terms_per_chunk])

            # Quantize weights
            quantized_terms = {}
            for term_id, weight in filtered_terms.items():
                quantized_weight = int(weight * self.quantization_scale)
                if quantized_weight > 0:
                    quantized_terms[term_id] = quantized_weight

            # Create aggregated vector
            aggregated_vector = SPLADEVector(
                terms=quantized_terms,
                tokenizer_name=segment_vectors[0].tokenizer_name,
                model_name=segment_vectors[0].model_name,
                sparsity_threshold=self.sparsity_threshold,
                quantization_scale=self.quantization_scale,
            )

            processing_time = time.perf_counter() - start_time

            SPLADE_AGGREGATION_SECONDS.labels(operation="weighted_avg", status="ok").observe(
                processing_time
            )
            SPLADE_AGGREGATION_OPERATIONS.labels(operation="weighted_avg", status="ok").inc()
            SPLADE_TERMS_PROCESSED.labels(operation="weighted_avg").inc(total_terms_processed)

            logger.info(
                "Weighted average aggregation completed",
                extra={
                    "chunk_id": chunk_id,
                    "input_vectors": len(segment_vectors),
                    "total_terms_processed": total_terms_processed,
                    "filtered_terms": len(filtered_terms),
                    "quantized_terms": len(quantized_terms),
                    "processing_time_seconds": processing_time,
                },
            )

            return aggregated_vector

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_AGGREGATION_SECONDS.labels(operation="weighted_avg", status="error").observe(
                processing_time
            )
            SPLADE_AGGREGATION_OPERATIONS.labels(operation="weighted_avg", status="error").inc()

            logger.error(
                "Weighted average aggregation failed",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def validate_aggregation_result(
        self,
        aggregated_vector: SPLADEVector,
        chunk_id: str,
    ) -> list[str]:
        """Validate aggregation result.

        Args:
            aggregated_vector: Aggregated vector to validate
            chunk_id: Identifier for the chunk

        Returns:
            List of validation error messages

        """
        errors = []

        # Check sparsity threshold
        for term_id, weight in aggregated_vector.terms.items():
            if weight < self.sparsity_threshold * self.quantization_scale:
                errors.append(
                    f"Term {term_id} in chunk {chunk_id} below sparsity threshold: "
                    f"{weight} < {self.sparsity_threshold * self.quantization_scale}"
                )

        # Check maximum terms
        if len(aggregated_vector.terms) > self.max_terms_per_chunk:
            errors.append(
                f"Chunk {chunk_id} exceeds max terms: "
                f"{len(aggregated_vector.terms)} > {self.max_terms_per_chunk}"
            )

        # Check quantization
        for term_id, weight in aggregated_vector.terms.items():
            if weight <= 0:
                errors.append(
                    f"Term {term_id} in chunk {chunk_id} has non-positive weight: {weight}"
                )

            if weight > self.quantization_scale:
                errors.append(
                    f"Term {term_id} in chunk {chunk_id} exceeds quantization scale: "
                    f"{weight} > {self.quantization_scale}"
                )

        return errors

    def get_aggregation_stats(
        self,
        aggregated_vector: SPLADEVector,
        chunk_id: str,
    ) -> dict:
        """Get statistics about aggregation result.

        Args:
            aggregated_vector: Aggregated vector to analyze
            chunk_id: Identifier for the chunk

        Returns:
            Dictionary with aggregation statistics

        """
        if not aggregated_vector.terms:
            return {
                "chunk_id": chunk_id,
                "total_terms": 0,
                "avg_weight": 0,
                "min_weight": 0,
                "max_weight": 0,
                "sparsity_ratio": 1.0,
            }

        weights = list(aggregated_vector.terms.values())

        return {
            "chunk_id": chunk_id,
            "total_terms": len(aggregated_vector.terms),
            "avg_weight": sum(weights) / len(weights),
            "min_weight": min(weights),
            "max_weight": max(weights),
            "sparsity_ratio": len(aggregated_vector.terms) / self.max_terms_per_chunk,
            "weight_distribution": {
                "under_100": len([w for w in weights if w < 100]),
                "100_500": len([w for w in weights if 100 <= w < 500]),
                "500_1000": len([w for w in weights if 500 <= w < 1000]),
                "over_1000": len([w for w in weights if w >= 1000]),
            },
        }
