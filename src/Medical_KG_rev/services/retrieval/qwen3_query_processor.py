"""Qwen3 Query Processor

This module implements Qwen3 dense query processing with embedding generation,
query preprocessing, and similarity search for semantic retrieval.
"""

from typing import Any
import logging

from pydantic import BaseModel, Field
import numpy as np

from Medical_KG_rev.services.retrieval.qwen3_contextualized import HttpClient
from Medical_KG_rev.services.retrieval.qwen3_service import Qwen3Service


logger = logging.getLogger(__name__)


class Qwen3Query(BaseModel):
    """Qwen3 query representation."""

    query_text: str = Field(..., description="Original query text")
    contextualized_text: str = Field(..., description="Contextualized query text")
    embedding: list[float] = Field(..., description="Query embedding vector")
    embedding_dimension: int = Field(..., description="Embedding dimension")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Query metadata")


class Qwen3QueryResult(BaseModel):
    """Qwen3 query result."""

    chunk_id: str = Field(..., description="Chunk identifier")
    score: float = Field(..., description="Cosine similarity score")
    embedding_similarity: float = Field(..., description="Raw embedding similarity")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Result metadata")


class Qwen3QueryProcessor:
    """Qwen3 query processor for dense semantic retrieval.

    This processor handles query preprocessing, contextualization, embedding
    generation, and similarity search for Qwen3-based retrieval.
    """

    def __init__(
        self,
        qwen3_service: Qwen3Service,
        contextualized_processor: Qwen3ContextualizedProcessor,
        enable_query_expansion: bool = True,
        enable_contextualization: bool = True,
        max_query_length: int = 512,
    ):
        """Initialize the Qwen3 query processor.

        Args:
        ----
            qwen3_service: Qwen3 service for embedding generation
            contextualized_processor: Processor for query contextualization
            enable_query_expansion: Enable query expansion
            enable_contextualization: Enable query contextualization
            max_query_length: Maximum query length in tokens

        """
        self.qwen3_service = qwen3_service
        self.contextualized_processor = contextualized_processor
        self.enable_query_expansion = enable_query_expansion
        self.enable_contextualization = enable_contextualization
        self.max_query_length = max_query_length

        logger.info("Qwen3QueryProcessor initialized")

    def process_query(self, query_text: str) -> Qwen3Query:
        """Process a query for Qwen3 retrieval.

        Args:
        ----
            query_text: Raw query text

        Returns:
        -------
            Processed Qwen3 query with embedding vector

        """
        logger.info(f"Processing Qwen3 query: '{query_text[:100]}...'")

        try:
            # Preprocess query text
            processed_text = self._preprocess_query(query_text)

            # Contextualize query if enabled
            if self.enable_contextualization:
                contextualized_text = self._contextualize_query(processed_text)
            else:
                contextualized_text = processed_text

            # Generate embedding
            embedding = self._generate_embedding(contextualized_text)

            # Validate embedding
            if not embedding or len(embedding) == 0:
                raise ValueError("Failed to generate embedding")

            # Create Qwen3 query
            qwen3_query = Qwen3Query(
                query_text=query_text,
                contextualized_text=contextualized_text,
                embedding=embedding,
                embedding_dimension=len(embedding),
                metadata={
                    "original_query": query_text,
                    "processed_query": processed_text,
                    "contextualization_enabled": self.enable_contextualization,
                    "expansion_enabled": self.enable_query_expansion,
                },
            )

            logger.info(f"Qwen3 query processed successfully with {len(embedding)}-dim embedding")
            return qwen3_query

        except Exception as e:
            logger.error(f"Failed to process Qwen3 query: {e}")
            raise

    def _preprocess_query(self, query_text: str) -> str:
        """Preprocess query text for Qwen3 processing."""
        # Convert to lowercase
        processed = query_text.lower()

        # Remove extra whitespace
        processed = " ".join(processed.split())

        # Handle special characters
        processed = self._normalize_special_characters(processed)

        # Expand medical abbreviations if enabled
        if self.enable_query_expansion:
            processed = self._expand_medical_abbreviations(processed)

        # Truncate if too long
        if len(processed) > self.max_query_length:
            processed = processed[: self.max_query_length]

        return processed

    def _contextualize_query(self, query_text: str) -> str:
        """Contextualize query for better embedding generation."""
        try:
            # Use contextualized processor to enhance query
            contextualized = self.contextualized_processor.get_contextualized_text(query_text)
            return contextualized
        except Exception as e:
            logger.error(f"Failed to contextualize query: {e}")
            # Fallback to original query
            return query_text

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        try:
            # Use Qwen3 service to generate embedding
            result = self.qwen3_service.generate_embedding(text)

            if result and isinstance(result, list):
                return result
            else:
                raise ValueError("Invalid embedding result")

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

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

    def calculate_similarity(
        self,
        query_embedding: list[float],
        chunk_embedding: list[float],
    ) -> float:
        """Calculate cosine similarity between query and chunk embeddings.

        Args:
        ----
            query_embedding: Query embedding vector
            chunk_embedding: Chunk embedding vector

        Returns:
        -------
            Cosine similarity score

        """
        try:
            # Convert to numpy arrays
            query_vec = np.array(query_embedding)
            chunk_vec = np.array(chunk_embedding)

            # Ensure same dimension
            if len(query_vec) != len(chunk_vec):
                logger.warning(
                    f"Dimension mismatch: query={len(query_vec)}, chunk={len(chunk_vec)}"
                )
                return 0.0

            # Calculate cosine similarity
            dot_product = np.dot(query_vec, chunk_vec)
            query_norm = np.linalg.norm(query_vec)
            chunk_norm = np.linalg.norm(chunk_vec)

            if query_norm == 0 or chunk_norm == 0:
                return 0.0

            similarity = dot_product / (query_norm * chunk_norm)

            # Ensure similarity is in [0, 1] range
            similarity = max(0.0, min(1.0, similarity))

            return float(similarity)

        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

    def score_query_against_chunk(
        self,
        qwen3_query: Qwen3Query,
        chunk_embedding: list[float],
    ) -> float:
        """Score a Qwen3 query against a chunk embedding.

        Args:
        ----
            qwen3_query: Processed Qwen3 query
            chunk_embedding: Chunk's embedding vector

        Returns:
        -------
            Similarity score

        """
        try:
            # Calculate cosine similarity
            similarity = self.calculate_similarity(qwen3_query.embedding, chunk_embedding)

            # Apply any additional scoring logic here
            score = self._apply_scoring_enhancements(similarity, qwen3_query, chunk_embedding)

            return score

        except Exception as e:
            logger.error(f"Failed to score query against chunk: {e}")
            return 0.0

    def _apply_scoring_enhancements(
        self,
        base_similarity: float,
        qwen3_query: Qwen3Query,
        chunk_embedding: list[float],
    ) -> float:
        """Apply additional scoring enhancements."""
        # For now, return base similarity
        # Can be enhanced with query-specific or chunk-specific adjustments
        return base_similarity

    def batch_score_queries(
        self,
        queries: list[Qwen3Query],
        chunk_embeddings: dict[str, list[float]],
    ) -> dict[str, list[tuple[str, float]]]:
        """Score multiple queries against multiple chunk embeddings.

        Args:
        ----
            queries: List of Qwen3 queries
            chunk_embeddings: Dictionary of chunk_id -> embedding

        Returns:
        -------
            Dictionary of query_id -> list of (chunk_id, score) tuples

        """
        results = {}

        for i, query in enumerate(queries):
            query_id = f"query_{i}"
            query_results = []

            for chunk_id, chunk_embedding in chunk_embeddings.items():
                try:
                    score = self.score_query_against_chunk(query, chunk_embedding)
                    query_results.append((chunk_id, score))
                except Exception as e:
                    logger.error(f"Failed to score query {i} against chunk {chunk_id}: {e}")
                    continue

            # Sort by score descending
            query_results.sort(key=lambda x: x[1], reverse=True)
            results[query_id] = query_results

        return results

    def explain_query(self, qwen3_query: Qwen3Query) -> dict[str, Any]:
        """Explain how a Qwen3 query will be processed."""
        explanation = {
            "original_query": qwen3_query.query_text,
            "contextualized_query": qwen3_query.contextualized_text,
            "embedding_dimension": qwen3_query.embedding_dimension,
            "embedding_norm": (
                np.linalg.norm(qwen3_query.embedding) if qwen3_query.embedding else 0.0
            ),
            "metadata": qwen3_query.metadata,
        }

        return explanation

    def get_query_stats(self) -> dict[str, Any]:
        """Get statistics about query processing."""
        stats = {
            "processor": "qwen3_query_processor",
            "max_query_length": self.max_query_length,
            "query_expansion_enabled": self.enable_query_expansion,
            "contextualization_enabled": self.enable_contextualization,
            "embedding_dimension": 4096,  # Qwen3 default
        }

        return stats
