"""Sentence splitting utilities for chunking."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from .exceptions import ChunkerConfigurationError

logger = logging.getLogger(__name__)


class SentenceSplitter(ABC):
    """Abstract base class for sentence splitters."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split text into sentences."""
        pass


class NLTKSentenceSplitter(SentenceSplitter):
    """Sentence splitter using NLTK."""

    def __init__(self, language: str = "english") -> None:
        """Initialize the NLTK sentence splitter."""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize

            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            self._sent_tokenize = sent_tokenize
            self.language = language
        except ImportError as exc:
            raise ChunkerConfigurationError(
                "nltk must be installed to use NLTKSentenceSplitter"
            ) from exc

    def split(self, text: str) -> list[str]:
        """Split text into sentences using NLTK."""
        if not text:
            return []

        try:
            sentences = self._sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as exc:
            logger.error("nltk.sentence_split_failed", error=str(exc))
            raise ChunkerConfigurationError(
                "NLTK sentence splitter failed"
            ) from exc


class SpacySentenceSplitter(SentenceSplitter):
    """Sentence splitter using spaCy."""

    def __init__(self, model: str = "en_core_web_sm") -> None:
        """Initialize the spaCy sentence splitter."""
        try:
            import spacy
            self._nlp = spacy.load(model)
        except ImportError as exc:
            raise ChunkerConfigurationError(
                "spacy must be installed to use SpacySentenceSplitter"
            ) from exc
        except OSError as exc:
            raise ChunkerConfigurationError(
                f"spaCy model '{model}' not found. Install with: python -m spacy download {model}"
            ) from exc

    def split(self, text: str) -> list[str]:
        """Split text into sentences using spaCy."""
        if not text:
            return []

        try:
            doc = self._nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            return [s for s in sentences if s]
        except Exception as exc:
            logger.error("spacy.sentence_split_failed", error=str(exc))
            raise ChunkerConfigurationError(
                "spaCy sentence splitter failed"
            ) from exc


class PySBDSentenceSplitter(SentenceSplitter):
    """Sentence splitter using PySBD."""

    def __init__(self, language: str = "en") -> None:
        """Initialize the PySBD sentence splitter."""
        try:
            import pysbd
            self._segmenter = pysbd.Segmenter(language=language, clean=True)
            self.language = language
        except ImportError as exc:
            raise ChunkerConfigurationError(
                "pysbd must be installed to use PySBDSentenceSplitter"
            ) from exc

    def split(self, text: str) -> list[str]:
        """Split text into sentences using PySBD."""
        if not text:
            return []

        try:
            segments = self._segmenter.segment(text)
            return [segment.strip() for segment in segments if segment.strip()]
        except Exception as exc:
            logger.error("pysbd.sentence_split_failed", error=str(exc))
            raise ChunkerConfigurationError(
                "PySBD sentence splitter failed"
            ) from exc


class RegexSentenceSplitter(SentenceSplitter):
    """Simple regex-based sentence splitter."""

    def __init__(self, pattern: str = r'[.!?]+') -> None:
        """Initialize the regex sentence splitter."""
        self.pattern = pattern
        self._regex = re.compile(pattern)

    def split(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        if not text:
            return []

        sentences = self._regex.split(text)
        return [s.strip() for s in sentences if s.strip()]


class SentenceSplitterFactory:
    """Factory for creating sentence splitters."""

    _splitters = {
        "nltk": NLTKSentenceSplitter,
        "spacy": SpacySentenceSplitter,
        "pysbd": PySBDSentenceSplitter,
        "regex": RegexSentenceSplitter,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> SentenceSplitter:
        """Create a sentence splitter by name."""
        if name not in cls._splitters:
            raise ChunkerConfigurationError(f"Unknown sentence splitter: {name}")

        try:
            return cls._splitters[name](**kwargs)
        except Exception as exc:
            raise ChunkerConfigurationError(
                f"Failed to create sentence splitter '{name}': {exc}"
            ) from exc

    @classmethod
    def list_splitters(cls) -> list[str]:
        """List available sentence splitter names."""
        return list(cls._splitters.keys())


# Create factory instance
sentence_splitter_factory = SentenceSplitterFactory()
