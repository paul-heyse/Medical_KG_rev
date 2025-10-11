"""SPLADE Query Processor

This module implements SPLADE query processing with sparse vector generation,
query segmentation, and impact scoring for learned sparse retrieval.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from Medical_KG_rev.services.retrieval.splade_aggregation import SPLADEAggregator
from Medical_KG_rev.services.retrieval.splade_segmentation import SPLADESegmenter
from Medical_KG_rev.services.retrieval.splade_service import SPLADEService
from Medical_KG_rev.services.retrieval.splade_sparsity import SPLADESparsityController

logger = logging.getLogger(__name__)


class SPLADEQuery(BaseModel):
    """SPLADE query representation."""

    query_text: str = Field(..., description="Original query text")
    segments: list[str] = Field(default_factory=list, description="Query segments")
    sparse_vector: dict[int, float] = Field(
        default_factory=dict, description="Sparse vector representation"
    )
    impact_scores: dict[int, float] = Field(
        default_factory=dict, description="Impact scores by term"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Query metadata")


class SPLADEQueryResult(BaseModel):
    """SPLADE query result."""

    chunk_id: str = Field(..., description="Chunk identifier")
    score: float = Field(..., description="SPLADE similarity score")
    matched_terms: list[int] = Field(default_factory=list, description="Matched term IDs")
    term_contributions: dict[int, float] = Field(
        default_factory=dict, description="Term contribution scores"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Result metadata")


class SPLADEQueryProcessor:
    """SPLADE query processor for learned sparse retrieval.

    This processor handles query preprocessing, segmentation, sparse vector
    generation, and impact scoring for SPLADE-based retrieval.
    """

    def __init__(
        self,
        splade_service: SPLADEService,
        segmenter: SPLADESegmenter,
        aggregator: SPLADEAggregator,
        sparsity_controller: SPLADESparsityController,
        max_query_length: int = 512,
        enable_query_expansion: bool = True,
    ):
        """Initialize the SPLADE query processor.

        Args:
            splade_service: SPLADE service for model operations
            segmenter: Query segmenter for long queries
            aggregator: Vector aggregator for segment combination
            sparsity_controller: Sparsity controller for vector optimization
            max_query_length: Maximum query length in tokens
            enable_query_expansion: Enable query expansion

        """
        self.splade_service = splade_service
        self.segmenter = segmenter
        self.aggregator = aggregator
        self.sparsity_controller = sparsity_controller
        self.max_query_length = max_query_length
        self.enable_query_expansion = enable_query_expansion

        logger.info("SPLADEQueryProcessor initialized")

    def process_query(self, query_text: str) -> SPLADEQuery:
        """Process a query for SPLADE retrieval.

        Args:
            query_text: Raw query text

        Returns:
            Processed SPLADE query with sparse vector representation

        """
        logger.info(f"Processing SPLADE query: '{query_text[:100]}...'")

        try:
            # Preprocess query text
            processed_text = self._preprocess_query(query_text)

            # Segment query if necessary
            segments = self._segment_query(processed_text)

            # Generate sparse vectors for each segment
            segment_vectors = []
            for segment in segments:
                vector = self._generate_segment_vector(segment)
                if vector:
                    segment_vectors.append(vector)

            # Aggregate segment vectors
            aggregated_vector = self._aggregate_vectors(segment_vectors)

            # Apply sparsity control
            sparse_vector = self._apply_sparsity_control(aggregated_vector)

            # Extract impact scores
            impact_scores = self._extract_impact_scores(sparse_vector)

            # Create SPLADE query
            splade_query = SPLADEQuery(
                query_text=query_text,
                segments=segments,
                sparse_vector=sparse_vector,
                impact_scores=impact_scores,
                metadata={
                    "original_query": query_text,
                    "processed_query": processed_text,
                    "num_segments": len(segments),
                    "num_terms": len(sparse_vector),
                    "expansion_enabled": self.enable_query_expansion,
                },
            )

            logger.info(f"SPLADE query processed successfully with {len(sparse_vector)} terms")
            return splade_query

        except Exception as e:
            logger.error(f"Failed to process SPLADE query: {e}")
            raise

    def _preprocess_query(self, query_text: str) -> str:
        """Preprocess query text for SPLADE processing."""
        # Convert to lowercase
        processed = query_text.lower()

        # Remove extra whitespace
        processed = " ".join(processed.split())

        # Handle special characters
        processed = self._normalize_special_characters(processed)

        # Expand medical abbreviations if enabled
        if self.enable_query_expansion:
            processed = self._expand_medical_abbreviations(processed)

        return processed

    def _segment_query(self, query_text: str) -> list[str]:
        """Segment query into manageable pieces."""
        # Check if query needs segmentation
        token_count = self._count_tokens(query_text)

        if token_count <= self.max_query_length:
            return [query_text]

        # Use segmenter to split long queries
        segments = self.segmenter.segment_chunk_for_splade(query_text)

        # Ensure segments are not empty
        segments = [seg for seg in segments if seg.strip()]

        return segments

    def _generate_segment_vector(self, segment: str) -> dict[int, float] | None:
        """Generate sparse vector for a query segment."""
        try:
            # Use SPLADE service to encode segment
            result = self.splade_service.process_batch([segment])

            if result and len(result) > 0:
                # Extract sparse vector from result
                sparse_vector = result[0].get("sparse_vector", {})
                return sparse_vector

            return None

        except Exception as e:
            logger.error(f"Failed to generate segment vector: {e}")
            return None

    def _aggregate_vectors(self, segment_vectors: list[dict[int, float]]) -> dict[int, float]:
        """Aggregate multiple segment vectors into a single vector."""
        if not segment_vectors:
            return {}

        if len(segment_vectors) == 1:
            return segment_vectors[0]

        # Use aggregator to combine vectors
        try:
            aggregated = self.aggregator.aggregate_splade_segments(segment_vectors)
            return aggregated
        except Exception as e:
            logger.error(f"Failed to aggregate vectors: {e}")
            # Fallback to simple max aggregation
            return self._simple_max_aggregation(segment_vectors)

    def _simple_max_aggregation(self, vectors: list[dict[int, float]]) -> dict[int, float]:
        """Simple max aggregation as fallback."""
        aggregated = {}

        for vector in vectors:
            for term_id, score in vector.items():
                if term_id in aggregated:
                    aggregated[term_id] = max(aggregated[term_id], score)
                else:
                    aggregated[term_id] = score

        return aggregated

    def _apply_sparsity_control(self, vector: dict[int, float]) -> dict[int, float]:
        """Apply sparsity control to the vector."""
        try:
            controlled = self.sparsity_controller.apply_sparsity_threshold(vector)
            return controlled
        except Exception as e:
            logger.error(f"Failed to apply sparsity control: {e}")
            # Fallback to simple thresholding
            return self._simple_sparsity_control(vector)

    def _simple_sparsity_control(
        self, vector: dict[int, float], threshold: float = 0.1
    ) -> dict[int, float]:
        """Simple sparsity control as fallback."""
        return {term_id: score for term_id, score in vector.items() if score >= threshold}

    def _extract_impact_scores(self, sparse_vector: dict[int, float]) -> dict[int, float]:
        """Extract impact scores from sparse vector."""
        # For SPLADE, impact scores are the same as vector values
        # but we can apply additional processing here
        impact_scores = {}

        for term_id, score in sparse_vector.items():
            # Apply impact score transformation if needed
            impact_score = self._transform_impact_score(score)
            impact_scores[term_id] = impact_score

        return impact_scores

    def _transform_impact_score(self, score: float) -> float:
        """Transform raw score to impact score."""
        # Simple transformation - can be customized
        return max(0.0, score)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using SPLADE tokenizer."""
        try:
            # Use SPLADE service tokenizer
            tokenizer = self.splade_service.tokenizer
            if tokenizer:
                tokens = tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            else:
                # Fallback to word count
                return len(text.split())
        except Exception as e:
            logger.error(f"Failed to count tokens: {e}")
            return len(text.split())

    def _normalize_special_characters(self, text: str) -> str:
        """Normalize special characters in text."""
        # Replace common special characters
        replacements = {
            "&": " and ",
            "|": " or ",
            "!": " not ",
            "(": " ",
            ")": " ",
            "[": " ",
            "]": " ",
            "{": " ",
            "}": " ",
            '"': " ",
            "'": " ",
            "`": " ",
        }

        normalized = text
        for old_char, new_char in replacements.items():
            normalized = normalized.replace(old_char, new_char)

        # Remove multiple spaces
        normalized = " ".join(normalized.split())

        return normalized

    def _expand_medical_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations in text."""
        # Common medical abbreviations
        abbreviations = {
            "htn": "hypertension",
            "dm": "diabetes mellitus",
            "mi": "myocardial infarction",
            "copd": "chronic obstructive pulmonary disease",
            "bmi": "body mass index",
            "bp": "blood pressure",
            "hr": "heart rate",
            "temp": "temperature",
            "wt": "weight",
            "ht": "height",
        }

        expanded = text
        for abbrev, expansion in abbreviations.items():
            # Replace standalone abbreviations
            expanded = expanded.replace(f" {abbrev} ", f" {expansion} ")
            expanded = expanded.replace(f" {abbrev}.", f" {expansion}.")

        return expanded

    def score_query_against_chunk(
        self,
        splade_query: SPLADEQuery,
        chunk_vector: dict[int, float],
    ) -> float:
        """Score a SPLADE query against a chunk vector.

        Args:
            splade_query: Processed SPLADE query
            chunk_vector: Chunk's sparse vector

        Returns:
            Similarity score

        """
        try:
            # Calculate dot product between query and chunk vectors
            score = 0.0

            for term_id, query_score in splade_query.sparse_vector.items():
                if term_id in chunk_vector:
                    chunk_score = chunk_vector[term_id]
                    score += query_score * chunk_score

            return score

        except Exception as e:
            logger.error(f"Failed to score query against chunk: {e}")
            return 0.0

    def explain_query(self, splade_query: SPLADEQuery) -> dict[str, Any]:
        """Explain how a SPLADE query will be processed."""
        explanation = {
            "original_query": splade_query.query_text,
            "segments": splade_query.segments,
            "num_terms": len(splade_query.sparse_vector),
            "top_terms": self._get_top_terms(splade_query.sparse_vector, 10),
            "impact_scores": splade_query.impact_scores,
            "metadata": splade_query.metadata,
        }

        return explanation

    def _get_top_terms(self, sparse_vector: dict[int, float], k: int) -> list[tuple[int, float]]:
        """Get top k terms by score."""
        sorted_terms = sorted(sparse_vector.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:k]

    def get_query_stats(self) -> dict[str, Any]:
        """Get statistics about query processing."""
        stats = {
            "processor": "splade_query_processor",
            "max_query_length": self.max_query_length,
            "query_expansion_enabled": self.enable_query_expansion,
            "sparsity_controller": (
                self.sparsity_controller.get_stats()
                if hasattr(self.sparsity_controller, "get_stats")
                else {}
            ),
        }

        return stats
