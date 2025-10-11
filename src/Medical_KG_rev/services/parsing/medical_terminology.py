"""Medical terminology support service for biomedical document processing.

This module provides medical terminology capabilities including:
- MeSH/UMLS synonym filter for BM25 index only
- Keep learned-sparse and dense encoders on original text
- Add controlled vocabulary expansions for medical terms
- Add terminology validation and quality checks
- Document terminology handling in processing pipeline
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any
import logging
import re


logger = logging.getLogger(__name__)


class TerminologySource(Enum):
    """Sources of medical terminology."""

    MESH = "mesh"  # Medical Subject Headings
    UMLS = "umls"  # Unified Medical Language System
    SNOMED = "snomed"  # SNOMED CT
    ICD10 = "icd10"  # ICD-10
    ICD11 = "icd11"  # ICD-11
    RX_NORM = "rx_norm"  # RxNorm
    LOINC = "loinc"  # LOINC
    CUSTOM = "custom"  # Custom terminology


class TerminologyType(Enum):
    """Types of terminology processing."""

    SYNONYM_EXPANSION = "synonym_expansion"  # Expand synonyms for BM25
    NORMALIZATION = "normalization"  # Normalize terms
    VALIDATION = "validation"  # Validate terms
    MAPPING = "mapping"  # Map between terminologies


@dataclass
class MedicalTerm:
    """Represents a medical term."""

    term: str
    normalized_term: str
    synonyms: list[str]
    source: TerminologySource
    concept_id: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = None


@dataclass
class TerminologyResult:
    """Result of terminology processing."""

    original_text: str
    processed_text: str
    expanded_terms: list[str]
    normalized_terms: list[str]
    validation_results: list[bool]
    terminology_applied: list[str]
    confidence_score: float


class MedicalTerminologyProcessor:
    """Medical terminology processor for biomedical document processing.

    Handles:
    - MeSH/UMLS synonym filter for BM25 index only
    - Keep learned-sparse and dense encoders on original text
    - Add controlled vocabulary expansions for medical terms
    - Add terminology validation and quality checks
    - Document terminology handling in processing pipeline
    """

    def __init__(self, enable_synonym_expansion: bool = True, enable_validation: bool = True):
        """Initialize the medical terminology processor.

        Args:
        ----
            enable_synonym_expansion: Whether to enable synonym expansion
            enable_validation: Whether to enable terminology validation

        """
        self.enable_synonym_expansion = enable_synonym_expansion
        self.enable_validation = enable_validation
        self._init_terminology_data()
        self._init_patterns()

    def _init_terminology_data(self) -> None:
        """Initialize terminology data structures."""
        # Medical term synonyms (simplified for demonstration)
        self.medical_synonyms = {
            # Common medical terms
            "heart attack": ["myocardial infarction", "MI", "acute myocardial infarction"],
            "stroke": ["cerebrovascular accident", "CVA", "brain attack"],
            "diabetes": ["diabetes mellitus", "DM", "diabetic"],
            "hypertension": ["high blood pressure", "HTN", "elevated blood pressure"],
            "cancer": ["neoplasm", "tumor", "malignancy", "carcinoma"],
            "pneumonia": ["lung infection", "respiratory infection"],
            "asthma": ["bronchial asthma", "reactive airway disease"],
            "arthritis": ["joint inflammation", "rheumatoid arthritis", "RA"],
            "depression": ["major depressive disorder", "MDD", "clinical depression"],
            "anxiety": ["anxiety disorder", "generalized anxiety disorder", "GAD"],
            # Medical procedures
            "surgery": ["operation", "surgical procedure", "intervention"],
            "biopsy": ["tissue sample", "histological examination"],
            "chemotherapy": ["chemo", "cancer treatment", "antineoplastic therapy"],
            "radiotherapy": ["radiation therapy", "radiation treatment", "RT"],
            "transplant": ["transplantation", "organ transplant", "graft"],
            # Medical conditions
            "infection": ["infectious disease", "pathogen", "microbial infection"],
            "inflammation": ["inflammatory response", "inflammatory condition"],
            "allergy": ["allergic reaction", "hypersensitivity", "allergic response"],
            "fracture": ["broken bone", "bone break", "fractured bone"],
            "bleeding": ["hemorrhage", "blood loss", "bleeding disorder"],
            # Medical specialties
            "cardiology": ["heart medicine", "cardiovascular medicine"],
            "oncology": ["cancer medicine", "tumor medicine"],
            "neurology": ["brain medicine", "nervous system medicine"],
            "dermatology": ["skin medicine", "dermatological medicine"],
            "pediatrics": ["children's medicine", "pediatric medicine"],
        }

        # Medical term normalization rules
        self.normalization_rules = {
            # Common abbreviations
            r"\bMI\b": "myocardial infarction",
            r"\bCVA\b": "cerebrovascular accident",
            r"\bDM\b": "diabetes mellitus",
            r"\bHTN\b": "hypertension",
            r"\bRA\b": "rheumatoid arthritis",
            r"\bMDD\b": "major depressive disorder",
            r"\bGAD\b": "generalized anxiety disorder",
            r"\bRT\b": "radiotherapy",
            r"\bICU\b": "intensive care unit",
            r"\bER\b": "emergency room",
            r"\bOR\b": "operating room",
            r"\bMRI\b": "magnetic resonance imaging",
            r"\bCT\b": "computed tomography",
            r"\bPET\b": "positron emission tomography",
            r"\bECG\b": "electrocardiogram",
            r"\bEEG\b": "electroencephalogram",
            r"\bCBC\b": "complete blood count",
            r"\bBMP\b": "basic metabolic panel",
            r"\bCMP\b": "comprehensive metabolic panel",
        }

        # Medical term validation patterns
        self.validation_patterns = {
            # Drug names (generic and brand)
            r"\b[A-Z][a-z]+(?:cin|mycin|pam|zole|pril|sartan|olol|pine|zine)\b": "drug_name",
            # Medical conditions
            r"\b[A-Z][a-z]+(?:itis|osis|emia|uria|pathy|plasia|trophy)\b": "medical_condition",
            # Medical procedures
            r"\b[A-Z][a-z]+(?:ectomy|otomy|oscopy|plasty|graphy|scopy)\b": "medical_procedure",
            # Anatomical terms
            r"\b[A-Z][a-z]+(?:cardia|pulmonary|hepatic|renal|gastric|intestinal)\b": "anatomical_term",
            # Medical units
            r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|IU|U|mEq|mol|mmol|μmol|nmol|pmol)\b": "medical_unit",
        }

    def _init_patterns(self) -> None:
        """Initialize regex patterns for terminology processing."""
        # Medical term detection patterns
        self.term_patterns = [
            # Medical abbreviations
            r"\b[A-Z]{2,4}\b",
            # Medical terms with common suffixes
            r"\b\w+(?:itis|osis|emia|uria|pathy|plasia|trophy)\b",
            # Medical terms with common prefixes
            r"\b(?:cardio|neuro|dermato|pediatric|onco|gyneco|ortho|ophthalmo)\w+\b",
            # Drug names
            r"\b\w+(?:cin|mycin|pam|zole|pril|sartan|olol|pine|zine)\b",
            # Medical procedures
            r"\b\w+(?:ectomy|otomy|oscopy|plasty|graphy|scopy)\b",
        ]

        # Stop words for medical text
        self.medical_stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "within",
            "without",
            "under",
            "over",
            "around",
            "near",
            "far",
            "here",
            "there",
            "where",
            "when",
            "why",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "mine",
            "yours",
            "hers",
            "ours",
            "theirs",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "shall",
            "am",
        }

    def process_text(
        self, text: str, terminology_type: TerminologyType = TerminologyType.SYNONYM_EXPANSION
    ) -> TerminologyResult:
        """Process text with medical terminology support.

        Args:
        ----
            text: Input text to process
            terminology_type: Type of terminology processing to apply

        Returns:
        -------
            TerminologyResult with processed text and metadata

        """
        if not text or not text.strip():
            return TerminologyResult(
                original_text=text,
                processed_text=text,
                expanded_terms=[],
                normalized_terms=[],
                validation_results=[],
                terminology_applied=[],
                confidence_score=1.0,
            )

        original_text = text
        processed_text = text
        expanded_terms = []
        normalized_terms = []
        validation_results = []
        terminology_applied = []

        try:
            # Step 1: Extract medical terms
            medical_terms = self._extract_medical_terms(text)

            # Step 2: Apply terminology processing based on type
            if (
                terminology_type == TerminologyType.SYNONYM_EXPANSION
                and self.enable_synonym_expansion
            ):
                processed_text, expanded = self._expand_synonyms(text, medical_terms)
                expanded_terms = expanded
                terminology_applied.append("synonym_expansion")

            elif terminology_type == TerminologyType.NORMALIZATION:
                processed_text, normalized = self._normalize_terms(text, medical_terms)
                normalized_terms = normalized
                terminology_applied.append("normalization")

            elif terminology_type == TerminologyType.VALIDATION and self.enable_validation:
                validation_results = self._validate_terms(medical_terms)
                terminology_applied.append("validation")

            elif terminology_type == TerminologyType.MAPPING:
                processed_text, mapped = self._map_terms(text, medical_terms)
                terminology_applied.append("mapping")

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                original_text, processed_text, terminology_applied
            )

            return TerminologyResult(
                original_text=original_text,
                processed_text=processed_text,
                expanded_terms=expanded_terms,
                normalized_terms=normalized_terms,
                validation_results=validation_results,
                terminology_applied=terminology_applied,
                confidence_score=confidence_score,
            )

        except Exception as e:
            logger.error(f"Error processing medical terminology: {e}")
            return TerminologyResult(
                original_text=original_text,
                processed_text=original_text,
                expanded_terms=[],
                normalized_terms=[],
                validation_results=[],
                terminology_applied=["error"],
                confidence_score=0.0,
            )

    def _extract_medical_terms(self, text: str) -> list[str]:
        """Extract medical terms from text."""
        terms = []

        for pattern in self.term_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)

        # Remove duplicates and filter out stop words
        unique_terms = list(set(terms))
        medical_terms = [
            term for term in unique_terms if term.lower() not in self.medical_stop_words
        ]

        return medical_terms

    def _expand_synonyms(self, text: str, medical_terms: list[str]) -> tuple[str, list[str]]:
        """Expand synonyms for medical terms."""
        expanded_text = text
        expanded_terms = []

        for term in medical_terms:
            term_lower = term.lower()
            if term_lower in self.medical_synonyms:
                synonyms = self.medical_synonyms[term_lower]
                expanded_terms.extend(synonyms)

                # Add synonyms to text
                synonym_text = " ".join(synonyms)
                expanded_text = expanded_text.replace(term, f"{term} {synonym_text}")

        return expanded_text, expanded_terms

    def _normalize_terms(self, text: str, medical_terms: list[str]) -> tuple[str, list[str]]:
        """Normalize medical terms."""
        normalized_text = text
        normalized_terms = []

        for pattern, replacement in self.normalization_rules.items():
            if re.search(pattern, normalized_text):
                normalized_text = re.sub(pattern, replacement, normalized_text)
                normalized_terms.append(f"{pattern} -> {replacement}")

        return normalized_text, normalized_terms

    def _validate_terms(self, medical_terms: list[str]) -> list[bool]:
        """Validate medical terms."""
        validation_results = []

        for term in medical_terms:
            is_valid = False

            for pattern, term_type in self.validation_patterns.items():
                if re.search(pattern, term, re.IGNORECASE):
                    is_valid = True
                    break

            validation_results.append(is_valid)

        return validation_results

    def _map_terms(self, text: str, medical_terms: list[str]) -> tuple[str, list[str]]:
        """Map medical terms between terminologies."""
        mapped_text = text
        mapped_terms = []

        # Simple mapping example (in practice, this would use external APIs)
        for term in medical_terms:
            term_lower = term.lower()
            if term_lower in self.medical_synonyms:
                # Map to first synonym as canonical form
                canonical = self.medical_synonyms[term_lower][0]
                mapped_text = mapped_text.replace(term, canonical)
                mapped_terms.append(f"{term} -> {canonical}")

        return mapped_text, mapped_terms

    def _calculate_confidence_score(
        self, original: str, processed: str, applied: list[str]
    ) -> float:
        """Calculate confidence score for terminology processing."""
        if not applied or "error" in applied:
            return 0.0

        # Base confidence
        confidence = 1.0

        # Reduce confidence for aggressive changes
        if "synonym_expansion" in applied:
            # Check if text length increased significantly
            length_ratio = len(processed) / len(original)
            if length_ratio > 1.5:
                confidence -= 0.2

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def create_bm25_synonym_filter(self, text: str) -> str:
        """Create synonym filter for BM25 index only.

        This method expands synonyms for BM25 retrieval while keeping
        original text for learned-sparse and dense encoders.

        Args:
        ----
            text: Input text for BM25 processing

        Returns:
        -------
            Text with expanded synonyms for BM25

        """
        result = self.process_text(text, TerminologyType.SYNONYM_EXPANSION)
        return result.processed_text

    def validate_medical_terminology(self, result: TerminologyResult) -> bool:
        """Validate medical terminology processing result.

        Args:
        ----
            result: TerminologyResult to validate

        Returns:
        -------
            True if validation passes, False otherwise

        """
        if result.confidence_score < 0.5:
            return False

        # Check for reasonable text length preservation
        length_ratio = len(result.processed_text) / len(result.original_text)
        if length_ratio < 0.5 or length_ratio > 3.0:
            return False

        # Check for medical term preservation
        original_terms = self._extract_medical_terms(result.original_text)
        processed_terms = self._extract_medical_terms(result.processed_text)

        # Ensure key medical terms are preserved
        for term in original_terms:
            if len(term) > 3:  # Skip very short terms
                if term.lower() not in result.processed_text.lower():
                    return False

        return True

    def get_terminology_stats(self) -> dict[str, Any]:
        """Get terminology processing statistics."""
        return {
            "enable_synonym_expansion": self.enable_synonym_expansion,
            "enable_validation": self.enable_validation,
            "medical_synonyms": len(self.medical_synonyms),
            "normalization_rules": len(self.normalization_rules),
            "validation_patterns": len(self.validation_patterns),
            "term_patterns": len(self.term_patterns),
            "medical_stop_words": len(self.medical_stop_words),
        }


def process_medical_terminology(
    text: str,
    terminology_type: TerminologyType = TerminologyType.SYNONYM_EXPANSION,
    enable_synonym_expansion: bool = True,
    enable_validation: bool = True,
) -> TerminologyResult:
    """Convenience function for medical terminology processing.

    Args:
    ----
        text: Text to process
        terminology_type: Type of processing
        enable_synonym_expansion: Whether to enable synonym expansion
        enable_validation: Whether to enable validation

    Returns:
    -------
        TerminologyResult

    """
    processor = MedicalTerminologyProcessor(enable_synonym_expansion, enable_validation)
    return processor.process_text(text, terminology_type)
