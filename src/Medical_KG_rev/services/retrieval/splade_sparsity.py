"""SPLADE sparsity control and quantization utilities.

This module implements sparsity control and quantization for SPLADE vectors,
including threshold application, term capping, and weight quantization
for efficient storage and retrieval.
"""

import logging

import numpy as np

from Medical_KG_rev.services.retrieval.splade_service import SPLADEVector
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
SPLADE_SPARSITY_SECONDS = Histogram(
    "splade_sparsity_seconds",
    "Time spent on SPLADE sparsity operations",
    ["operation", "status"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

SPLADE_SPARSITY_OPERATIONS = Counter(
    "splade_sparsity_operations_total",
    "Total number of SPLADE sparsity operations",
    ["operation", "status"],
)

SPLADE_TERMS_FILTERED = Counter(
    "splade_terms_filtered_total",
    "Total number of terms filtered by sparsity",
    ["operation", "reason"],
)


class SPLADESparsityController:
    """Controller for SPLADE vector sparsity and quantization.

    This class handles sparsity control, term capping, and weight quantization
    for SPLADE vectors to ensure efficient storage and retrieval.
    """

    def __init__(
        self,
        sparsity_threshold: float = 0.01,
        quantization_scale: int = 1000,
        max_terms_per_chunk: int = 10000,
        quantization_method: str = "round",
        preserve_top_k: bool = True,
    ):
        """Initialize SPLADE sparsity controller.

        Args:
            sparsity_threshold: Minimum weight threshold for sparsity control
            quantization_scale: Scale factor for weight quantization
            max_terms_per_chunk: Maximum number of terms to keep per chunk
            quantization_method: Method for quantization ("round", "floor", "ceil")
            preserve_top_k: Whether to preserve top-k terms regardless of threshold

        """
        self.sparsity_threshold = sparsity_threshold
        self.quantization_scale = quantization_scale
        self.max_terms_per_chunk = max_terms_per_chunk
        self.quantization_method = quantization_method
        self.preserve_top_k = preserve_top_k

        logger.info(
            "Initialized SPLADE sparsity controller",
            extra={
                "sparsity_threshold": self.sparsity_threshold,
                "quantization_scale": self.quantization_scale,
                "max_terms_per_chunk": self.max_terms_per_chunk,
                "quantization_method": self.quantization_method,
                "preserve_top_k": self.preserve_top_k,
            },
        )

    def apply_sparsity_threshold(
        self,
        vector: SPLADEVector,
        chunk_id: str,
    ) -> SPLADEVector:
        """Apply sparsity threshold to filter out low-weight terms.

        Args:
            vector: SPLADE vector to apply sparsity threshold to
            chunk_id: Identifier for the chunk

        Returns:
            SPLADE vector with sparsity threshold applied

        """
        import time

        start_time = time.perf_counter()

        try:
            original_terms = len(vector.terms)

            # Filter terms by sparsity threshold
            filtered_terms = {
                term_id: weight
                for term_id, weight in vector.terms.items()
                if weight >= self.sparsity_threshold
            }

            filtered_count = original_terms - len(filtered_terms)

            processing_time = time.perf_counter() - start_time

            SPLADE_SPARSITY_SECONDS.labels(operation="threshold", status="ok").observe(
                processing_time
            )
            SPLADE_SPARSITY_OPERATIONS.labels(operation="threshold", status="ok").inc()
            SPLADE_TERMS_FILTERED.labels(operation="threshold", reason="below_threshold").inc(
                filtered_count
            )

            logger.info(
                "Sparsity threshold applied",
                extra={
                    "chunk_id": chunk_id,
                    "original_terms": original_terms,
                    "filtered_terms": len(filtered_terms),
                    "filtered_out": filtered_count,
                    "processing_time_seconds": processing_time,
                },
            )

            return SPLADEVector(
                terms=filtered_terms,
                tokenizer_name=vector.tokenizer_name,
                model_name=vector.model_name,
                sparsity_threshold=self.sparsity_threshold,
                quantization_scale=vector.quantization_scale,
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_SPARSITY_SECONDS.labels(operation="threshold", status="error").observe(
                processing_time
            )
            SPLADE_SPARSITY_OPERATIONS.labels(operation="threshold", status="error").inc()

            logger.error(
                "Sparsity threshold application failed",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def cap_max_terms(
        self,
        vector: SPLADEVector,
        chunk_id: str,
    ) -> SPLADEVector:
        """Cap the number of terms to max_terms_per_chunk.

        Args:
            vector: SPLADE vector to cap terms for
            chunk_id: Identifier for the chunk

        Returns:
            SPLADE vector with capped terms

        """
        import time

        start_time = time.perf_counter()

        try:
            if len(vector.terms) <= self.max_terms_per_chunk:
                processing_time = time.perf_counter() - start_time
                SPLADE_SPARSITY_SECONDS.labels(operation="cap", status="ok").observe(
                    processing_time
                )
                SPLADE_SPARSITY_OPERATIONS.labels(operation="cap", status="ok").inc()
                return vector

            # Sort terms by weight and keep top-k
            sorted_terms = sorted(vector.terms.items(), key=lambda x: x[1], reverse=True)

            capped_terms = dict(sorted_terms[: self.max_terms_per_chunk])
            capped_count = len(vector.terms) - len(capped_terms)

            processing_time = time.perf_counter() - start_time

            SPLADE_SPARSITY_SECONDS.labels(operation="cap", status="ok").observe(processing_time)
            SPLADE_SPARSITY_OPERATIONS.labels(operation="cap", status="ok").inc()
            SPLADE_TERMS_FILTERED.labels(operation="cap", reason="max_terms").inc(capped_count)

            logger.info(
                "Terms capped to max_terms_per_chunk",
                extra={
                    "chunk_id": chunk_id,
                    "original_terms": len(vector.terms),
                    "capped_terms": len(capped_terms),
                    "capped_out": capped_count,
                    "processing_time_seconds": processing_time,
                },
            )

            return SPLADEVector(
                terms=capped_terms,
                tokenizer_name=vector.tokenizer_name,
                model_name=vector.model_name,
                sparsity_threshold=vector.sparsity_threshold,
                quantization_scale=vector.quantization_scale,
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_SPARSITY_SECONDS.labels(operation="cap", status="error").observe(processing_time)
            SPLADE_SPARSITY_OPERATIONS.labels(operation="cap", status="error").inc()

            logger.error(
                "Term capping failed",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def quantize_weights(
        self,
        vector: SPLADEVector,
        chunk_id: str,
    ) -> SPLADEVector:
        """Quantize weights to fixed-point integers.

        Args:
            vector: SPLADE vector to quantize
            chunk_id: Identifier for the chunk

        Returns:
            SPLADE vector with quantized weights

        """
        import time

        start_time = time.perf_counter()

        try:
            quantized_terms = {}

            for term_id, weight in vector.terms.items():
                # Apply quantization method
                if self.quantization_method == "round":
                    quantized_weight = int(round(weight * self.quantization_scale))
                elif self.quantization_method == "floor":
                    quantized_weight = int(np.floor(weight * self.quantization_scale))
                elif self.quantization_method == "ceil":
                    quantized_weight = int(np.ceil(weight * self.quantization_scale))
                else:
                    raise ValueError(f"Unknown quantization method: {self.quantization_method}")

                # Only keep positive quantized weights
                if quantized_weight > 0:
                    quantized_terms[term_id] = quantized_weight

            filtered_count = len(vector.terms) - len(quantized_terms)

            processing_time = time.perf_counter() - start_time

            SPLADE_SPARSITY_SECONDS.labels(operation="quantize", status="ok").observe(
                processing_time
            )
            SPLADE_SPARSITY_OPERATIONS.labels(operation="quantize", status="ok").inc()
            SPLADE_TERMS_FILTERED.labels(operation="quantize", reason="zero_weight").inc(
                filtered_count
            )

            logger.info(
                "Weights quantized",
                extra={
                    "chunk_id": chunk_id,
                    "original_terms": len(vector.terms),
                    "quantized_terms": len(quantized_terms),
                    "filtered_out": filtered_count,
                    "quantization_method": self.quantization_method,
                    "processing_time_seconds": processing_time,
                },
            )

            return SPLADEVector(
                terms=quantized_terms,
                tokenizer_name=vector.tokenizer_name,
                model_name=vector.model_name,
                sparsity_threshold=vector.sparsity_threshold,
                quantization_scale=self.quantization_scale,
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_SPARSITY_SECONDS.labels(operation="quantize", status="error").observe(
                processing_time
            )
            SPLADE_SPARSITY_OPERATIONS.labels(operation="quantize", status="error").inc()

            logger.error(
                "Weight quantization failed",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def apply_full_sparsity_control(
        self,
        vector: SPLADEVector,
        chunk_id: str,
    ) -> SPLADEVector:
        """Apply full sparsity control pipeline.

        This applies sparsity threshold, term capping, and quantization
        in sequence for complete sparsity control.

        Args:
            vector: SPLADE vector to apply sparsity control to
            chunk_id: Identifier for the chunk

        Returns:
            SPLADE vector with full sparsity control applied

        """
        import time

        start_time = time.perf_counter()

        try:
            original_terms = len(vector.terms)

            # Step 1: Apply sparsity threshold
            vector = self.apply_sparsity_threshold(vector, chunk_id)

            # Step 2: Cap max terms
            vector = self.cap_max_terms(vector, chunk_id)

            # Step 3: Quantize weights
            vector = self.quantize_weights(vector, chunk_id)

            final_terms = len(vector.terms)
            total_filtered = original_terms - final_terms

            processing_time = time.perf_counter() - start_time

            SPLADE_SPARSITY_SECONDS.labels(operation="full_control", status="ok").observe(
                processing_time
            )
            SPLADE_SPARSITY_OPERATIONS.labels(operation="full_control", status="ok").inc()

            logger.info(
                "Full sparsity control applied",
                extra={
                    "chunk_id": chunk_id,
                    "original_terms": original_terms,
                    "final_terms": final_terms,
                    "total_filtered": total_filtered,
                    "sparsity_ratio": final_terms / original_terms if original_terms > 0 else 0,
                    "processing_time_seconds": processing_time,
                },
            )

            return vector

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            SPLADE_SPARSITY_SECONDS.labels(operation="full_control", status="error").observe(
                processing_time
            )
            SPLADE_SPARSITY_OPERATIONS.labels(operation="full_control", status="error").inc()

            logger.error(
                "Full sparsity control failed",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise

    def validate_sparsity_result(
        self,
        vector: SPLADEVector,
        chunk_id: str,
    ) -> list[str]:
        """Validate sparsity control result.

        Args:
            vector: SPLADE vector to validate
            chunk_id: Identifier for the chunk

        Returns:
            List of validation error messages

        """
        errors = []

        # Check sparsity threshold
        for term_id, weight in vector.terms.items():
            if weight < self.sparsity_threshold * self.quantization_scale:
                errors.append(
                    f"Term {term_id} in chunk {chunk_id} below sparsity threshold: "
                    f"{weight} < {self.sparsity_threshold * self.quantization_scale}"
                )

        # Check maximum terms
        if len(vector.terms) > self.max_terms_per_chunk:
            errors.append(
                f"Chunk {chunk_id} exceeds max terms: "
                f"{len(vector.terms)} > {self.max_terms_per_chunk}"
            )

        # Check quantization
        for term_id, weight in vector.terms.items():
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

    def get_sparsity_stats(
        self,
        vector: SPLADEVector,
        chunk_id: str,
    ) -> dict:
        """Get statistics about sparsity control result.

        Args:
            vector: SPLADE vector to analyze
            chunk_id: Identifier for the chunk

        Returns:
            Dictionary with sparsity statistics

        """
        if not vector.terms:
            return {
                "chunk_id": chunk_id,
                "total_terms": 0,
                "avg_weight": 0,
                "min_weight": 0,
                "max_weight": 0,
                "sparsity_ratio": 1.0,
                "quantization_efficiency": 0.0,
            }

        weights = list(vector.terms.values())

        return {
            "chunk_id": chunk_id,
            "total_terms": len(vector.terms),
            "avg_weight": sum(weights) / len(weights),
            "min_weight": min(weights),
            "max_weight": max(weights),
            "sparsity_ratio": len(vector.terms) / self.max_terms_per_chunk,
            "quantization_efficiency": len(vector.terms) / self.max_terms_per_chunk,
            "weight_distribution": {
                "under_100": len([w for w in weights if w < 100]),
                "100_500": len([w for w in weights if 100 <= w < 500]),
                "500_1000": len([w for w in weights if 500 <= w < 1000]),
                "over_1000": len([w for w in weights if w >= 1000]),
            },
            "sparsity_threshold": self.sparsity_threshold,
            "quantization_scale": self.quantization_scale,
            "max_terms_per_chunk": self.max_terms_per_chunk,
        }

    def round_trip_test(
        self,
        vector: SPLADEVector,
        chunk_id: str,
    ) -> dict:
        """Test round-trip quantization accuracy.

        Args:
            vector: SPLADE vector to test
            chunk_id: Identifier for the chunk

        Returns:
            Dictionary with round-trip test results

        """
        try:
            # Store original weights
            original_weights = vector.terms.copy()

            # Apply quantization
            quantized_vector = self.quantize_weights(vector, chunk_id)

            # Dequantize weights
            dequantized_weights = {}
            for term_id, quantized_weight in quantized_vector.terms.items():
                dequantized_weight = quantized_weight / self.quantization_scale
                dequantized_weights[term_id] = dequantized_weight

            # Calculate accuracy metrics
            mse = 0.0
            mae = 0.0
            max_error = 0.0
            total_terms = 0

            for term_id in original_weights:
                if term_id in dequantized_weights:
                    original_weight = original_weights[term_id]
                    dequantized_weight = dequantized_weights[term_id]

                    error = abs(original_weight - dequantized_weight)
                    mse += error**2
                    mae += error
                    max_error = max(max_error, error)
                    total_terms += 1

            if total_terms > 0:
                mse /= total_terms
                mae /= total_terms

            return {
                "chunk_id": chunk_id,
                "total_terms": total_terms,
                "mse": mse,
                "mae": mae,
                "max_error": max_error,
                "quantization_method": self.quantization_method,
                "quantization_scale": self.quantization_scale,
            }

        except Exception as e:
            logger.error(
                "Round-trip test failed",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                },
            )
            return {
                "chunk_id": chunk_id,
                "error": str(e),
            }
