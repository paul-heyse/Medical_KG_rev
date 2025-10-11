"""BM25 Query Processor

This module implements BM25 query processing with medical domain optimizations,
including multi-field queries, term expansion, and medical terminology support.
"""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from Medical_KG_rev.services.retrieval.bm25_analyzers import BM25Analyzer
from Medical_KG_rev.services.retrieval.bm25_field_mapping import BM25FieldMapper

logger = logging.getLogger(__name__)


class BM25Query(BaseModel):
    """BM25 query representation."""

    query_text: str = Field(..., description="Original query text")
    expanded_terms: dict[str, list[str]] = Field(
        default_factory=dict, description="Expanded terms by field"
    )
    field_queries: dict[str, str] = Field(
        default_factory=dict, description="Field-specific queries"
    )
    boost_factors: dict[str, float] = Field(
        default_factory=dict, description="Boost factors by field"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Query metadata")


class BM25QueryResult(BaseModel):
    """BM25 query result."""

    chunk_id: str = Field(..., description="Chunk identifier")
    score: float = Field(..., description="BM25 relevance score")
    field_scores: dict[str, float] = Field(default_factory=dict, description="Scores by field")
    matched_terms: list[str] = Field(default_factory=list, description="Matched terms")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Result metadata")


class BM25QueryProcessor:
    """BM25 query processor with medical domain optimizations.

    This processor handles query preprocessing, term expansion, multi-field
    query generation, and medical terminology support for BM25 retrieval.
    """

    def __init__(
        self,
        field_mapper: BM25FieldMapper,
        analyzer: BM25Analyzer,
        enable_medical_expansion: bool = True,
        enable_synonym_expansion: bool = True,
        max_expansion_terms: int = 5,
    ):
        """Initialize the BM25 query processor.

        Args:
            field_mapper: Field mapper for chunk structure
            analyzer: Text analyzer for tokenization and normalization
            enable_medical_expansion: Enable medical terminology expansion
            enable_synonym_expansion: Enable synonym expansion
            max_expansion_terms: Maximum number of expansion terms per query term

        """
        self.field_mapper = field_mapper
        self.analyzer = analyzer
        self.enable_medical_expansion = enable_medical_expansion
        self.enable_synonym_expansion = enable_synonym_expansion
        self.max_expansion_terms = max_expansion_terms

        # Medical terminology patterns
        self.medical_patterns = {
            "dosage": r"\d+\s*(mg|g|ml|mcg|units?)\b",
            "frequency": r"\d+\s*(times?|x)\s*(per|daily|weekly|monthly)\b",
            "condition": r"\b(diabetes|hypertension|depression|anxiety|cancer|heart disease)\b",
            "medication": r"\b(insulin|metformin|aspirin|ibuprofen|acetaminophen)\b",
            "procedure": r"\b(surgery|biopsy|scan|test|examination)\b",
        }

        # Common medical abbreviations and expansions
        self.medical_abbreviations = {
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

        logger.info("BM25QueryProcessor initialized")

    def process_query(self, query_text: str) -> BM25Query:
        """Process a query for BM25 retrieval.

        Args:
            query_text: Raw query text

        Returns:
            Processed BM25 query with field-specific queries and expansions

        """
        logger.info(f"Processing BM25 query: '{query_text[:100]}...'")

        try:
            # Preprocess query text
            processed_text = self._preprocess_query(query_text)

            # Extract medical terms and patterns
            medical_terms = self._extract_medical_terms(processed_text)

            # Expand terms using medical terminology
            expanded_terms = self._expand_medical_terms(medical_terms)

            # Generate field-specific queries
            field_queries = self._generate_field_queries(processed_text, expanded_terms)

            # Set boost factors based on field importance
            boost_factors = self._calculate_boost_factors(field_queries)

            # Create BM25 query
            bm25_query = BM25Query(
                query_text=query_text,
                expanded_terms=expanded_terms,
                field_queries=field_queries,
                boost_factors=boost_factors,
                metadata={
                    "original_query": query_text,
                    "processed_query": processed_text,
                    "medical_terms": medical_terms,
                    "expansion_enabled": self.enable_medical_expansion,
                },
            )

            logger.info(f"BM25 query processed successfully with {len(field_queries)} fields")
            return bm25_query

        except Exception as e:
            logger.error(f"Failed to process BM25 query: {e}")
            raise

    def _preprocess_query(self, query_text: str) -> str:
        """Preprocess query text for BM25 processing."""
        # Convert to lowercase
        processed = query_text.lower()

        # Remove extra whitespace
        processed = re.sub(r"\s+", " ", processed).strip()

        # Handle medical abbreviations
        if self.enable_medical_expansion:
            for abbrev, expansion in self.medical_abbreviations.items():
                # Replace standalone abbreviations
                processed = re.sub(r"\b" + abbrev + r"\b", expansion, processed)

        # Remove special characters but preserve medical terms
        processed = re.sub(r"[^\w\s\-/]", " ", processed)

        # Normalize medical units
        processed = self._normalize_medical_units(processed)

        return processed

    def _extract_medical_terms(self, query_text: str) -> list[str]:
        """Extract medical terms from query text."""
        medical_terms = []

        for pattern_name, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    medical_terms.extend([m for m in match if m])
                else:
                    medical_terms.append(match)

        # Also extract individual words that might be medical terms
        words = query_text.split()
        for word in words:
            if len(word) > 3 and word.isalpha():
                # Check if word might be a medical term
                if self._is_potential_medical_term(word):
                    medical_terms.append(word)

        return list(set(medical_terms))  # Remove duplicates

    def _expand_medical_terms(self, medical_terms: list[str]) -> dict[str, list[str]]:
        """Expand medical terms using terminology knowledge."""
        expanded_terms = {}

        for term in medical_terms:
            expansions = [term]  # Include original term

            # Add medical abbreviation expansions
            if term in self.medical_abbreviations:
                expansions.append(self.medical_abbreviations[term])

            # Add synonym expansions (placeholder for MeSH/UMLS integration)
            if self.enable_synonym_expansion:
                synonyms = self._get_medical_synonyms(term)
                expansions.extend(synonyms)

            # Limit expansions
            expansions = expansions[: self.max_expansion_terms]
            expanded_terms[term] = expansions

        return expanded_terms

    def _generate_field_queries(
        self, query_text: str, expanded_terms: dict[str, list[str]]
    ) -> dict[str, str]:
        """Generate field-specific queries with appropriate boosts."""
        field_queries = {}

        # Get field configuration from field mapper
        field_config = self.field_mapper.field_config

        # Generate queries for each field
        for field_name, field_config in field_config.items():
            field_query = self._build_field_query(
                query_text, expanded_terms, field_name, field_config
            )
            if field_query:
                field_queries[field_name] = field_query

        return field_queries

    def _build_field_query(
        self,
        query_text: str,
        expanded_terms: dict[str, list[str]],
        field_name: str,
        field_config: dict[str, Any],
    ) -> str | None:
        """Build a query for a specific field."""
        # Analyze query text for this field
        analyzed_terms = self.analyzer.analyze(query_text, field_name)

        if not analyzed_terms:
            return None

        # Build query with expansions
        query_parts = []

        for term in analyzed_terms:
            if term in expanded_terms:
                # Use expanded terms
                expanded = expanded_terms[term]
                if len(expanded) > 1:
                    # Create OR query for expansions
                    expansion_query = " OR ".join(f'"{exp}"' for exp in expanded)
                    query_parts.append(f"({expansion_query})")
                else:
                    query_parts.append(f'"{expanded[0]}"')
            else:
                query_parts.append(f'"{term}"')

        if not query_parts:
            return None

        # Combine terms with AND
        field_query = " AND ".join(query_parts)

        # Add field-specific modifiers
        if field_config.get("preserve_exact", False):
            # For fields that preserve exact matches (like titles)
            field_query = f"({field_query})"

        return field_query

    def _calculate_boost_factors(self, field_queries: dict[str, str]) -> dict[str, float]:
        """Calculate boost factors for each field based on importance."""
        boost_factors = {}

        # Default boost factors based on field importance
        default_boosts = {
            "title": 3.0,
            "section_headers": 2.5,
            "paragraph": 1.0,
            "caption": 2.0,
            "table_text": 1.5,
            "footnote": 0.5,
            "refs_text": 0.1,
        }

        for field_name, query in field_queries.items():
            if query:  # Only boost fields with actual queries
                boost_factors[field_name] = default_boosts.get(field_name, 1.0)

        return boost_factors

    def _normalize_medical_units(self, text: str) -> str:
        """Normalize medical units in text."""
        # Normalize common unit variations
        unit_mappings = {
            "mg/dl": "mg per dl",
            "mg/kg": "mg per kg",
            "ml/min": "ml per min",
            "units/kg": "units per kg",
            "iu/ml": "iu per ml",
        }

        normalized = text
        for old_unit, new_unit in unit_mappings.items():
            normalized = re.sub(r"\b" + re.escape(old_unit) + r"\b", new_unit, normalized)

        return normalized

    def _is_potential_medical_term(self, word: str) -> bool:
        """Check if a word might be a medical term."""
        # Simple heuristics for medical terms
        medical_suffixes = ["itis", "osis", "emia", "uria", "oma", "pathy", "therapy"]
        medical_prefixes = ["hyper", "hypo", "anti", "pro", "pre", "post"]

        word_lower = word.lower()

        # Check suffixes
        for suffix in medical_suffixes:
            if word_lower.endswith(suffix):
                return True

        # Check prefixes
        for prefix in medical_prefixes:
            if word_lower.startswith(prefix):
                return True

        # Check for medical term patterns
        if re.match(r"^[a-z]+[a-z]{2,}$", word_lower):
            # Word with at least 3 characters
            return True

        return False

    def _get_medical_synonyms(self, term: str) -> list[str]:
        """Get medical synonyms for a term (placeholder for MeSH/UMLS integration)."""
        # Placeholder implementation
        # In a real system, this would integrate with MeSH/UMLS APIs

        synonym_mappings = {
            "diabetes": ["diabetes mellitus", "diabetic", "diabetes type 2"],
            "hypertension": ["high blood pressure", "elevated blood pressure"],
            "depression": ["major depressive disorder", "clinical depression"],
            "cancer": ["neoplasm", "malignancy", "tumor"],
            "heart": ["cardiac", "cardiovascular", "myocardial"],
        }

        return synonym_mappings.get(term.lower(), [])

    def explain_query(self, bm25_query: BM25Query) -> dict[str, Any]:
        """Explain how a BM25 query will be processed."""
        explanation = {
            "original_query": bm25_query.query_text,
            "field_queries": bm25_query.field_queries,
            "boost_factors": bm25_query.boost_factors,
            "expanded_terms": bm25_query.expanded_terms,
            "metadata": bm25_query.metadata,
        }

        return explanation

    def get_query_stats(self) -> dict[str, Any]:
        """Get statistics about query processing."""
        stats = {
            "processor": "bm25_query_processor",
            "medical_expansion_enabled": self.enable_medical_expansion,
            "synonym_expansion_enabled": self.enable_synonym_expansion,
            "max_expansion_terms": self.max_expansion_terms,
            "medical_patterns": len(self.medical_patterns),
            "medical_abbreviations": len(self.medical_abbreviations),
        }

        return stats
