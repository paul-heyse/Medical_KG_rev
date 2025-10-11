"""BM25 service for structured medical document retrieval.

This module implements BM25 retrieval with structured fields optimized for
medical document search, including title, section headers, paragraphs,
captions, and table text with appropriate field boosts.
"""

from collections import defaultdict
from typing import Any
import logging
import math
import re
import time

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

from Medical_KG_rev.config.settings import get_settings


logger = logging.getLogger(__name__)

# Prometheus metrics
BM25_PROCESSING_SECONDS = Histogram(
    "bm25_processing_seconds",
    "Time spent on BM25 operations",
    ["operation", "status"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

BM25_OPERATIONS_TOTAL = Counter(
    "bm25_operations_total", "Total number of BM25 operations", ["operation", "status"]
)

BM25_INDEX_SIZE_BYTES = Histogram(
    "bm25_index_size_bytes",
    "Size of BM25 index in bytes",
    ["operation"],
    buckets=[1024, 10240, 102400, 1024000, 10240000, 102400000],
)


class BM25Field(BaseModel):
    """BM25 field configuration."""

    name: str = Field(..., description="Field name")
    boost: float = Field(default=1.0, description="Field boost factor")
    analyzer: str = Field(default="standard", description="Analyzer type")
    preserve_medical_terms: bool = Field(default=False, description="Preserve medical terminology")


class BM25Document(BaseModel):
    """BM25 document with structured fields."""

    chunk_id: str = Field(..., description="Chunk identifier")
    title: str = Field(default="", description="Document title")
    section_headers: str = Field(default="", description="Section headers")
    paragraph: str = Field(default="", description="Main paragraph text")
    caption: str = Field(default="", description="Caption text")
    table_text: str = Field(default="", description="Table text content")
    footnote: str = Field(default="", description="Footnote text")
    refs_text: str = Field(default="", description="References text")


class BM25Query(BaseModel):
    """BM25 query with field-specific terms."""

    query_text: str = Field(..., description="Original query text")
    title_terms: list[str] = Field(default_factory=list, description="Title query terms")
    section_terms: list[str] = Field(default_factory=list, description="Section query terms")
    paragraph_terms: list[str] = Field(default_factory=list, description="Paragraph query terms")
    caption_terms: list[str] = Field(default_factory=list, description="Caption query terms")
    table_terms: list[str] = Field(default_factory=list, description="Table query terms")
    footnote_terms: list[str] = Field(default_factory=list, description="Footnote query terms")


class BM25Result(BaseModel):
    """BM25 search result."""

    chunk_id: str = Field(..., description="Chunk identifier")
    score: float = Field(..., description="BM25 relevance score")
    field_scores: dict[str, float] = Field(default_factory=dict, description="Per-field scores")
    matched_terms: dict[str, list[str]] = Field(
        default_factory=dict, description="Matched terms per field"
    )


class BM25ProcessingError(Exception):
    """Exception raised during BM25 processing."""

    pass


class BM25IndexError(Exception):
    """Exception raised during BM25 indexing."""

    pass


class BM25Service:
    """BM25 service for structured medical document retrieval.

    This service implements BM25 retrieval with structured fields optimized
    for medical document search, including proper field boosts and medical
    terminology preservation.
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        delta: float = 1.0,
        field_boosts: dict[str, float] | None = None,
        preserve_medical_terms: bool = True,
    ):
        """Initialize BM25 service.

        Args:
        ----
            k1: BM25 parameter controlling term frequency scaling
            b: BM25 parameter controlling document length normalization
            delta: BM25 parameter for term frequency floor
            field_boosts: Field boost factors
            preserve_medical_terms: Whether to preserve medical terminology

        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.preserve_medical_terms = preserve_medical_terms

        # Default field boosts for medical documents
        self.field_boosts = field_boosts or {
            "title": 3.0,
            "section_headers": 2.5,
            "paragraph": 1.0,
            "caption": 2.0,
            "table_text": 1.5,
            "footnote": 0.5,
            "refs_text": 0.1,
        }

        # Field configurations
        self.fields = {
            "title": BM25Field(
                name="title", boost=self.field_boosts["title"], preserve_medical_terms=True
            ),
            "section_headers": BM25Field(
                name="section_headers",
                boost=self.field_boosts["section_headers"],
                preserve_medical_terms=True,
            ),
            "paragraph": BM25Field(
                name="paragraph", boost=self.field_boosts["paragraph"], preserve_medical_terms=False
            ),
            "caption": BM25Field(
                name="caption", boost=self.field_boosts["caption"], preserve_medical_terms=True
            ),
            "table_text": BM25Field(
                name="table_text",
                boost=self.field_boosts["table_text"],
                preserve_medical_terms=False,
            ),
            "footnote": BM25Field(
                name="footnote", boost=self.field_boosts["footnote"], preserve_medical_terms=False
            ),
            "refs_text": BM25Field(
                name="refs_text", boost=self.field_boosts["refs_text"], preserve_medical_terms=False
            ),
        }

        # Document collection statistics
        self.doc_count = 0
        self.avg_doc_length = 0.0
        self.field_lengths: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.term_frequencies: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.document_frequencies: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Settings
        self._settings = get_settings()

        logger.info(
            "Initialized BM25 service",
            extra={
                "k1": self.k1,
                "b": self.b,
                "delta": self.delta,
                "field_boosts": self.field_boosts,
                "preserve_medical_terms": self.preserve_medical_terms,
            },
        )

    def _tokenize_text(self, text: str, preserve_medical: bool = False) -> list[str]:
        """Tokenize text with optional medical term preservation.

        Args:
        ----
            text: Text to tokenize
            preserve_medical: Whether to preserve medical terminology

        Returns:
        -------
            List of tokens

        """
        if not text:
            return []

        # Basic text preprocessing
        text = text.lower().strip()

        if preserve_medical:
            # Preserve medical terms and units
            # Keep medical abbreviations and units intact
            medical_patterns = [
                r"\b\d+\.?\d*\s*(mg|ml|kg|g|mcg|μg|μl|μmol|mmol|iu|units?)\b",  # Units
                r"\b\d+\.?\d*\s*(mg/kg|ml/kg|g/kg|mcg/kg)\b",  # Dosage units
                r"\b\d+\.?\d*\s*(mg/dl|mg/l|μg/dl|μg/l|mmol/l|iu/ml)\b",  # Concentration units
                r"\b\d+\.?\d*\s*(bpm|hr|min|sec|h|d|wk|mo|yr)\b",  # Time units
                r"\b\d+\.?\d*\s*(°c|°f|kpa|mmhg|cmh2o)\b",  # Physical units
                r"\b\d+\.?\d*\s*(mm|cm|m|in|ft)\b",  # Length units
                r"\b\d+\.?\d*\s*(ml|dl|l|gal|qt|pt)\b",  # Volume units
                r"\b\d+\.?\d*\s*(g|kg|lb|oz)\b",  # Weight units
                r"\b\d+\.?\d*\s*(%|percent)\b",  # Percentage
                r"\b\d+\.?\d*\s*(x|×)\s*\d+\.?\d*\b",  # Multipliers
            ]

            # Replace medical patterns with placeholders
            placeholders = {}
            for i, pattern in enumerate(medical_patterns):
                matches = re.findall(pattern, text)
                for match in matches:
                    placeholder = f"__MEDICAL_{i}_{len(placeholders)}__"
                    placeholders[placeholder] = match
                    text = text.replace(match, placeholder, 1)

        # Tokenize by splitting on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", text)

        if preserve_medical:
            # Restore medical terms
            for placeholder, original in placeholders.items():
                tokens = [
                    token.replace(placeholder, original) if placeholder in token else token
                    for token in tokens
                ]

        # Filter out very short tokens and common stop words
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "or",
            "but",
            "not",
            "this",
            "these",
            "they",
            "them",
            "their",
            "there",
            "then",
            "than",
            "so",
            "if",
            "can",
            "could",
            "would",
            "should",
            "may",
            "might",
            "must",
            "shall",
        }

        filtered_tokens = [token for token in tokens if len(token) > 1 and token not in stop_words]

        return filtered_tokens

    def _calculate_field_length(self, field_name: str, tokens: list[str]) -> int:
        """Calculate field length for BM25 scoring.

        Args:
        ----
            field_name: Name of the field
            tokens: List of tokens in the field

        Returns:
        -------
            Field length

        """
        return len(tokens)

    def _calculate_term_frequency(self, tokens: list[str]) -> dict[str, int]:
        """Calculate term frequency for a list of tokens.

        Args:
        ----
            tokens: List of tokens

        Returns:
        -------
            Dictionary mapping terms to their frequencies

        """
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        return dict(tf)

    def add_document(self, document: BM25Document) -> None:
        """Add a document to the BM25 index.

        Args:
        ----
            document: BM25 document to add

        """
        start_time = time.perf_counter()

        try:
            # Process each field
            for field_name, field_config in self.fields.items():
                field_text = getattr(document, field_name, "")
                tokens = self._tokenize_text(field_text, field_config.preserve_medical_terms)

                # Calculate field length
                field_length = self._calculate_field_length(field_name, tokens)
                self.field_lengths[document.chunk_id][field_name] = field_length

                # Calculate term frequencies
                term_freqs = self._calculate_term_frequency(tokens)
                for term, freq in term_freqs.items():
                    self.term_frequencies[field_name][document.chunk_id][term] = freq
                    self.document_frequencies[field_name][term] += 1

            # Update document count and average length
            self.doc_count += 1

            # Calculate average document length across all fields
            total_length = sum(
                self.field_lengths[document.chunk_id][field_name] for field_name in self.fields
            )
            self.avg_doc_length = (
                self.avg_doc_length * (self.doc_count - 1) + total_length
            ) / self.doc_count

            processing_time = time.perf_counter() - start_time

            BM25_PROCESSING_SECONDS.labels(operation="add_document", status="ok").observe(
                processing_time
            )
            BM25_OPERATIONS_TOTAL.labels(operation="add_document", status="ok").inc()

            logger.info(
                "Document added to BM25 index",
                extra={
                    "chunk_id": document.chunk_id,
                    "processing_time_seconds": processing_time,
                },
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            BM25_PROCESSING_SECONDS.labels(operation="add_document", status="error").observe(
                processing_time
            )
            BM25_OPERATIONS_TOTAL.labels(operation="add_document", status="error").inc()

            logger.error(
                "Failed to add document to BM25 index",
                extra={
                    "chunk_id": document.chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise BM25IndexError(f"Failed to add document: {e}")

    def _calculate_bm25_score(
        self,
        term: str,
        field_name: str,
        chunk_id: str,
        query_term_freq: int,
    ) -> float:
        """Calculate BM25 score for a term in a field.

        Args:
        ----
            term: Term to score
            field_name: Name of the field
            chunk_id: Document identifier
            query_term_freq: Frequency of term in query

        Returns:
        -------
            BM25 score

        """
        # Get term frequency in document field
        doc_term_freq = self.term_frequencies[field_name][chunk_id].get(term, 0)
        if doc_term_freq == 0:
            return 0.0

        # Get document frequency
        doc_freq = self.document_frequencies[field_name].get(term, 0)
        if doc_freq == 0:
            return 0.0

        # Get field length
        field_length = self.field_lengths[chunk_id].get(field_name, 0)
        if field_length == 0:
            return 0.0

        # Calculate BM25 components
        # Term frequency component
        tf_component = (doc_term_freq * (self.k1 + 1)) / (
            doc_term_freq + self.k1 * (1 - self.b + self.b * (field_length / self.avg_doc_length))
        )

        # Inverse document frequency component
        idf_component = self.delta + math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

        # Query term frequency component
        query_tf_component = query_term_freq * (self.k1 + 1) / (query_term_freq + self.k1)

        # Calculate final score
        score = tf_component * idf_component * query_tf_component

        return score

    def search(self, query_text: str, top_k: int = 10) -> list[BM25Result]:
        """Search documents using BM25 scoring.

        Args:
        ----
            query_text: Query text
            top_k: Number of top results to return

        Returns:
        -------
            List of BM25 results sorted by score

        """
        start_time = time.perf_counter()

        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query_text, preserve_medical=True)
            query_term_freqs = self._calculate_term_frequency(query_tokens)

            # Calculate scores for each document
            scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            matched_terms: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

            for chunk_id in self.field_lengths:
                for field_name, field_config in self.fields.items():
                    field_score = 0.0
                    field_matched_terms = []

                    for term, query_freq in query_term_freqs.items():
                        term_score = self._calculate_bm25_score(
                            term, field_name, chunk_id, query_freq
                        )
                        if term_score > 0:
                            field_score += term_score
                            field_matched_terms.append(term)

                    # Apply field boost
                    boosted_score = field_score * field_config.boost
                    scores[chunk_id][field_name] = boosted_score
                    matched_terms[chunk_id][field_name] = field_matched_terms

            # Aggregate scores and create results
            results = []
            for chunk_id in scores:
                total_score = sum(scores[chunk_id].values())
                if total_score > 0:
                    result = BM25Result(
                        chunk_id=chunk_id,
                        score=total_score,
                        field_scores=scores[chunk_id],
                        matched_terms=matched_terms[chunk_id],
                    )
                    results.append(result)

            # Sort by score and return top-k
            results.sort(key=lambda x: x.score, reverse=True)
            top_results = results[:top_k]

            processing_time = time.perf_counter() - start_time

            BM25_PROCESSING_SECONDS.labels(operation="search", status="ok").observe(processing_time)
            BM25_OPERATIONS_TOTAL.labels(operation="search", status="ok").inc()

            logger.info(
                "BM25 search completed",
                extra={
                    "query_text": query_text,
                    "results_count": len(top_results),
                    "processing_time_seconds": processing_time,
                },
            )

            return top_results

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            BM25_PROCESSING_SECONDS.labels(operation="search", status="error").observe(
                processing_time
            )
            BM25_OPERATIONS_TOTAL.labels(operation="search", status="error").inc()

            logger.error(
                "BM25 search failed",
                extra={
                    "query_text": query_text,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                },
            )
            raise BM25ProcessingError(f"Search failed: {e}")

    def get_document(self, chunk_id: str) -> BM25Document | None:
        """Get a document from the index.

        Args:
        ----
            chunk_id: Document identifier

        Returns:
        -------
            BM25 document if found, None otherwise

        """
        if chunk_id not in self.field_lengths:
            return None

        # Reconstruct document from index data
        # This is a simplified reconstruction - in practice, you'd store the original text
        document = BM25Document(chunk_id=chunk_id)

        for field_name in self.fields:
            # Get field length to indicate content presence
            field_length = self.field_lengths[chunk_id].get(field_name, 0)
            if field_length > 0:
                # In a real implementation, you'd store the original text
                # For now, we'll use a placeholder
                setattr(document, field_name, f"[{field_name} content - {field_length} tokens]")

        return document

    def remove_document(self, chunk_id: str) -> bool:
        """Remove a document from the index.

        Args:
        ----
            chunk_id: Document identifier

        Returns:
        -------
            True if document was removed, False if not found

        """
        if chunk_id not in self.field_lengths:
            return False

        try:
            # Remove from field lengths
            del self.field_lengths[chunk_id]

            # Remove from term frequencies
            for field_name in self.fields:
                if chunk_id in self.term_frequencies[field_name]:
                    del self.term_frequencies[field_name][chunk_id]

            # Update document count
            self.doc_count = max(0, self.doc_count - 1)

            logger.info("Document removed from BM25 index", extra={"chunk_id": chunk_id})
            return True

        except Exception as e:
            logger.error(
                "Failed to remove document from BM25 index",
                extra={"chunk_id": chunk_id, "error": str(e)},
            )
            return False

    def clear_index(self) -> None:
        """Clear all documents from the index."""
        self.doc_count = 0
        self.avg_doc_length = 0.0
        self.field_lengths.clear()
        self.term_frequencies.clear()
        self.document_frequencies.clear()

        logger.info("BM25 index cleared")

    def get_index_stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns
        -------
            Dictionary with index statistics

        """
        total_terms = sum(len(field_terms) for field_terms in self.document_frequencies.values())

        return {
            "doc_count": self.doc_count,
            "avg_doc_length": self.avg_doc_length,
            "total_terms": total_terms,
            "field_boosts": self.field_boosts,
            "bm25_params": {
                "k1": self.k1,
                "b": self.b,
                "delta": self.delta,
            },
            "preserve_medical_terms": self.preserve_medical_terms,
        }

    def health_check(self) -> dict[str, Any]:
        """Check BM25 service health.

        Returns
        -------
            Health status information

        """
        try:
            stats = self.get_index_stats()

            return {
                "status": "healthy",
                "doc_count": stats["doc_count"],
                "total_terms": stats["total_terms"],
                "avg_doc_length": stats["avg_doc_length"],
                "field_boosts": stats["field_boosts"],
                "bm25_params": stats["bm25_params"],
            }

        except Exception as e:
            logger.error("BM25 health check failed", extra={"error": str(e)})
            return {
                "status": "unhealthy",
                "error": str(e),
            }
