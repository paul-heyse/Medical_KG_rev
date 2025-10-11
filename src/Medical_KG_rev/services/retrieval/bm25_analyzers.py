"""BM25 analyzers and tokenizers for medical document processing.

This module implements specialized analyzers and tokenizers for BM25
indexing with medical terminology preservation and MeSH/UMLS support.
"""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnalyzerConfig(BaseModel):
    """Analyzer configuration."""

    name: str = Field(..., description="Analyzer name")
    lowercase: bool = Field(default=True, description="Convert to lowercase")
    remove_stopwords: bool = Field(default=True, description="Remove stop words")
    stemming: bool = Field(default=False, description="Apply stemming")
    preserve_medical_terms: bool = Field(default=True, description="Preserve medical terminology")
    mesh_synonyms: bool = Field(default=False, description="Use MeSH synonyms")
    umls_synonyms: bool = Field(default=False, description="Use UMLS synonyms")


class TokenizerConfig(BaseModel):
    """Tokenizer configuration."""

    name: str = Field(..., description="Tokenizer name")
    min_token_length: int = Field(default=2, description="Minimum token length")
    max_token_length: int = Field(default=50, description="Maximum token length")
    preserve_numbers: bool = Field(default=True, description="Preserve numbers")
    preserve_units: bool = Field(default=True, description="Preserve units")
    preserve_medical_terms: bool = Field(default=True, description="Preserve medical terms")


class BM25Analyzer:
    """BM25 analyzer for medical document processing.

    This analyzer implements specialized text processing for medical
    documents with terminology preservation and synonym expansion.
    """

    def __init__(self, config: AnalyzerConfig):
        """Initialize BM25 analyzer.

        Args:
            config: Analyzer configuration

        """
        self.config = config

        # Medical terminology patterns
        self.medical_patterns = [
            r"\b\d+\.?\d*\s*(mg|ml|kg|g|mcg|μg|μl|μmol|mmol|iu|units?)\b",
            r"\b\d+\.?\d*\s*(mg/kg|ml/kg|g/kg|mcg/kg)\b",
            r"\b\d+\.?\d*\s*(mg/dl|mg/l|μg/dl|μg/l|mmol/l|iu/ml)\b",
            r"\b\d+\.?\d*\s*(bpm|hr|min|sec|h|d|wk|mo|yr)\b",
            r"\b\d+\.?\d*\s*(°c|°f|kpa|mmhg|cmh2o)\b",
            r"\b\d+\.?\d*\s*(mm|cm|m|in|ft)\b",
            r"\b\d+\.?\d*\s*(ml|dl|l|gal|qt|pt)\b",
            r"\b\d+\.?\d*\s*(g|kg|lb|oz)\b",
            r"\b\d+\.?\d*\s*(%|percent)\b",
            r"\b\d+\.?\d*\s*(x|×)\s*\d+\.?\d*\b",
        ]

        # Compile patterns
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.medical_patterns
        ]

        # Stop words
        self.stop_words = {
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
            "have",
            "had",
            "been",
            "being",
            "do",
            "does",
            "did",
            "ought",
        }

        # MeSH synonyms (simplified - in practice, load from MeSH database)
        self.mesh_synonyms = {
            "heart attack": ["myocardial infarction", "mi", "cardiac infarction"],
            "diabetes": ["diabetes mellitus", "dm", "diabetic"],
            "hypertension": ["high blood pressure", "htn", "elevated blood pressure"],
            "cancer": ["neoplasm", "tumor", "malignancy", "carcinoma"],
            "stroke": ["cerebrovascular accident", "cva", "brain attack"],
            "pneumonia": ["lung infection", "respiratory infection"],
            "fracture": ["broken bone", "bone break", "crack"],
            "surgery": ["operation", "procedure", "surgical intervention"],
            "medication": ["drug", "medicine", "pharmaceutical", "therapeutic"],
            "treatment": ["therapy", "intervention", "management"],
        }

        # UMLS synonyms (simplified - in practice, load from UMLS database)
        self.umls_synonyms = {
            "patient": ["subject", "individual", "person", "case"],
            "disease": ["disorder", "condition", "illness", "syndrome"],
            "symptom": ["sign", "manifestation", "indication"],
            "diagnosis": ["assessment", "evaluation", "determination"],
            "prognosis": ["outcome", "prediction", "forecast"],
            "therapy": ["treatment", "intervention", "management"],
            "dosage": ["dose", "amount", "quantity", "strength"],
            "administration": ["delivery", "route", "method"],
            "contraindication": ["warning", "precaution", "restriction"],
            "indication": ["purpose", "use", "application"],
        }

        logger.info(
            "Initialized BM25 analyzer",
            extra={
                "name": self.config.name,
                "lowercase": self.config.lowercase,
                "remove_stopwords": self.config.remove_stopwords,
                "stemming": self.config.stemming,
                "preserve_medical_terms": self.config.preserve_medical_terms,
                "mesh_synonyms": self.config.mesh_synonyms,
                "umls_synonyms": self.config.umls_synonyms,
            },
        )

    def _preserve_medical_terms(self, text: str) -> tuple[str, dict[str, str]]:
        """Preserve medical terms in text.

        Args:
            text: Input text

        Returns:
            Tuple of (processed_text, placeholders)

        """
        placeholders = {}
        processed_text = text

        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(processed_text)
            for j, match in enumerate(matches):
                placeholder = f"__MEDICAL_{i}_{j}__"
                placeholders[placeholder] = match
                processed_text = processed_text.replace(match, placeholder, 1)

        return processed_text, placeholders

    def _restore_medical_terms(self, text: str, placeholders: dict[str, str]) -> str:
        """Restore medical terms in text.

        Args:
            text: Processed text with placeholders
            placeholders: Dictionary mapping placeholders to original terms

        Returns:
            Text with restored medical terms

        """
        restored_text = text
        for placeholder, original in placeholders.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text

    def _apply_synonyms(self, tokens: list[str]) -> list[str]:
        """Apply MeSH and UMLS synonyms.

        Args:
            tokens: List of tokens

        Returns:
            List of tokens with synonyms

        """
        if not (self.config.mesh_synonyms or self.config.umls_synonyms):
            return tokens

        expanded_tokens = []

        for token in tokens:
            expanded_tokens.append(token)

            # Check MeSH synonyms
            if self.config.mesh_synonyms:
                for term, synonyms in self.mesh_synonyms.items():
                    if token.lower() in term.lower() or term.lower() in token.lower():
                        expanded_tokens.extend(synonyms)

            # Check UMLS synonyms
            if self.config.umls_synonyms:
                for term, synonyms in self.umls_synonyms.items():
                    if token.lower() in term.lower() or term.lower() in token.lower():
                        expanded_tokens.extend(synonyms)

        return expanded_tokens

    def _apply_stemming(self, tokens: list[str]) -> list[str]:
        """Apply stemming to tokens.

        Args:
            tokens: List of tokens

        Returns:
            List of stemmed tokens

        """
        if not self.config.stemming:
            return tokens

        # Simple stemming implementation
        # In practice, use a proper stemming library like NLTK or spaCy
        stemmed_tokens = []

        for token in tokens:
            # Simple suffix removal
            if token.endswith("ing"):
                stemmed_tokens.append(token[:-3])
            elif token.endswith("ed"):
                stemmed_tokens.append(token[:-2])
            elif token.endswith("s") and len(token) > 3:
                stemmed_tokens.append(token[:-1])
            else:
                stemmed_tokens.append(token)

        return stemmed_tokens

    def analyze(self, text: str) -> list[str]:
        """Analyze text and return tokens.

        Args:
            text: Input text

        Returns:
            List of analyzed tokens

        """
        if not text:
            return []

        try:
            # Preserve medical terms if configured
            if self.config.preserve_medical_terms:
                processed_text, placeholders = self._preserve_medical_terms(text)
            else:
                processed_text, placeholders = text, {}

            # Convert to lowercase if configured
            if self.config.lowercase:
                processed_text = processed_text.lower()

            # Tokenize
            tokens = re.findall(r"\b\w+\b", processed_text)

            # Filter by length
            tokens = [token for token in tokens if len(token) >= 2]

            # Remove stop words if configured
            if self.config.remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words]

            # Apply stemming if configured
            tokens = self._apply_stemming(tokens)

            # Apply synonyms if configured
            tokens = self._apply_synonyms(tokens)

            # Restore medical terms
            if self.config.preserve_medical_terms:
                tokens = [self._restore_medical_terms(token, placeholders) for token in tokens]

            # Remove duplicates while preserving order
            seen = set()
            unique_tokens = []
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    unique_tokens.append(token)

            return unique_tokens

        except Exception as e:
            logger.error(
                "Failed to analyze text",
                extra={
                    "text_length": len(text),
                    "error": str(e),
                },
            )
            raise

    def get_analyzer_stats(self) -> dict[str, Any]:
        """Get analyzer statistics.

        Returns:
            Dictionary with analyzer statistics

        """
        return {
            "name": self.config.name,
            "lowercase": self.config.lowercase,
            "remove_stopwords": self.config.remove_stopwords,
            "stemming": self.config.stemming,
            "preserve_medical_terms": self.config.preserve_medical_terms,
            "mesh_synonyms": self.config.mesh_synonyms,
            "umls_synonyms": self.config.umls_synonyms,
            "stop_words_count": len(self.stop_words),
            "mesh_synonyms_count": len(self.mesh_synonyms),
            "umls_synonyms_count": len(self.umls_synonyms),
            "medical_patterns_count": len(self.medical_patterns),
        }


class BM25Tokenizer:
    """BM25 tokenizer for medical document processing.

    This tokenizer implements specialized tokenization for medical
    documents with unit preservation and medical term handling.
    """

    def __init__(self, config: TokenizerConfig):
        """Initialize BM25 tokenizer.

        Args:
            config: Tokenizer configuration

        """
        self.config = config

        # Medical unit patterns
        self.unit_patterns = [
            r"\b\d+\.?\d*\s*(mg|ml|kg|g|mcg|μg|μl|μmol|mmol|iu|units?)\b",
            r"\b\d+\.?\d*\s*(mg/kg|ml/kg|g/kg|mcg/kg)\b",
            r"\b\d+\.?\d*\s*(mg/dl|mg/l|μg/dl|μg/l|mmol/l|iu/ml)\b",
            r"\b\d+\.?\d*\s*(bpm|hr|min|sec|h|d|wk|mo|yr)\b",
            r"\b\d+\.?\d*\s*(°c|°f|kpa|mmhg|cmh2o)\b",
            r"\b\d+\.?\d*\s*(mm|cm|m|in|ft)\b",
            r"\b\d+\.?\d*\s*(ml|dl|l|gal|qt|pt)\b",
            r"\b\d+\.?\d*\s*(g|kg|lb|oz)\b",
            r"\b\d+\.?\d*\s*(%|percent)\b",
            r"\b\d+\.?\d*\s*(x|×)\s*\d+\.?\d*\b",
        ]

        # Compile patterns
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.unit_patterns
        ]

        logger.info(
            "Initialized BM25 tokenizer",
            extra={
                "name": self.config.name,
                "min_token_length": self.config.min_token_length,
                "max_token_length": self.config.max_token_length,
                "preserve_numbers": self.config.preserve_numbers,
                "preserve_units": self.config.preserve_units,
                "preserve_medical_terms": self.config.preserve_medical_terms,
            },
        )

    def _preserve_units(self, text: str) -> tuple[str, dict[str, str]]:
        """Preserve units in text.

        Args:
            text: Input text

        Returns:
            Tuple of (processed_text, placeholders)

        """
        placeholders = {}
        processed_text = text

        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(processed_text)
            for j, match in enumerate(matches):
                placeholder = f"__UNIT_{i}_{j}__"
                placeholders[placeholder] = match
                processed_text = processed_text.replace(match, placeholder, 1)

        return processed_text, placeholders

    def _restore_units(self, text: str, placeholders: dict[str, str]) -> str:
        """Restore units in text.

        Args:
            text: Processed text with placeholders
            placeholders: Dictionary mapping placeholders to original units

        Returns:
            Text with restored units

        """
        restored_text = text
        for placeholder, original in placeholders.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text.

        Args:
            text: Input text

        Returns:
            List of tokens

        """
        if not text:
            return []

        try:
            # Preserve units if configured
            if self.config.preserve_units:
                processed_text, placeholders = self._preserve_units(text)
            else:
                processed_text, placeholders = text, {}

            # Tokenize by splitting on whitespace and punctuation
            tokens = re.findall(r"\b\w+\b", processed_text)

            # Filter by length
            tokens = [
                token
                for token in tokens
                if self.config.min_token_length <= len(token) <= self.config.max_token_length
            ]

            # Preserve numbers if configured
            if not self.config.preserve_numbers:
                tokens = [token for token in tokens if not token.isdigit()]

            # Restore units
            if self.config.preserve_units:
                tokens = [self._restore_units(token, placeholders) for token in tokens]

            return tokens

        except Exception as e:
            logger.error(
                "Failed to tokenize text",
                extra={
                    "text_length": len(text),
                    "error": str(e),
                },
            )
            raise

    def get_tokenizer_stats(self) -> dict[str, Any]:
        """Get tokenizer statistics.

        Returns:
            Dictionary with tokenizer statistics

        """
        return {
            "name": self.config.name,
            "min_token_length": self.config.min_token_length,
            "max_token_length": self.config.max_token_length,
            "preserve_numbers": self.config.preserve_numbers,
            "preserve_units": self.config.preserve_units,
            "preserve_medical_terms": self.config.preserve_medical_terms,
            "unit_patterns_count": len(self.unit_patterns),
        }


class BM25AnalyzerFactory:
    """Factory for creating BM25 analyzers and tokenizers."""

    @staticmethod
    def create_standard_analyzer() -> BM25Analyzer:
        """Create standard analyzer.

        Returns:
            Standard BM25 analyzer

        """
        config = AnalyzerConfig(
            name="standard",
            lowercase=True,
            remove_stopwords=True,
            stemming=False,
            preserve_medical_terms=False,
            mesh_synonyms=False,
            umls_synonyms=False,
        )
        return BM25Analyzer(config)

    @staticmethod
    def create_medical_analyzer() -> BM25Analyzer:
        """Create medical analyzer.

        Returns:
            Medical BM25 analyzer

        """
        config = AnalyzerConfig(
            name="medical",
            lowercase=True,
            remove_stopwords=True,
            stemming=False,
            preserve_medical_terms=True,
            mesh_synonyms=True,
            umls_synonyms=True,
        )
        return BM25Analyzer(config)

    @staticmethod
    def create_title_analyzer() -> BM25Analyzer:
        """Create title analyzer.

        Returns:
            Title BM25 analyzer

        """
        config = AnalyzerConfig(
            name="title",
            lowercase=True,
            remove_stopwords=False,  # Keep all words in titles
            stemming=False,
            preserve_medical_terms=True,
            mesh_synonyms=True,
            umls_synonyms=True,
        )
        return BM25Analyzer(config)

    @staticmethod
    def create_standard_tokenizer() -> BM25Tokenizer:
        """Create standard tokenizer.

        Returns:
            Standard BM25 tokenizer

        """
        config = TokenizerConfig(
            name="standard",
            min_token_length=2,
            max_token_length=50,
            preserve_numbers=True,
            preserve_units=False,
            preserve_medical_terms=False,
        )
        return BM25Tokenizer(config)

    @staticmethod
    def create_medical_tokenizer() -> BM25Tokenizer:
        """Create medical tokenizer.

        Returns:
            Medical BM25 tokenizer

        """
        config = TokenizerConfig(
            name="medical",
            min_token_length=2,
            max_token_length=50,
            preserve_numbers=True,
            preserve_units=True,
            preserve_medical_terms=True,
        )
        return BM25Tokenizer(config)
