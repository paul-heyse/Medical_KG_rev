"""Sentence splitter adapters used by chunkers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .exceptions import ChunkerConfigurationError


class SentenceSplitter(Protocol):
    """Protocol for sentence splitter adapters."""

    def split(self, text: str) -> list[str]:
        ...


@dataclass(slots=True)
class NLTKSentenceSplitter:
    """Adapter around the NLTK Punkt tokenizer."""

    language: str = "english"

    def __post_init__(self) -> None:
        try:
            import nltk
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ChunkerConfigurationError(
                "nltk must be installed to use the NLTKSentenceSplitter"
            ) from exc
        try:
            nltk.data.find(f"tokenizers/punkt/{self.language}.pickle")
        except LookupError:  # pragma: no cover - data download path
            nltk.download("punkt")
        self._tokenizer = nltk.data.load(f"tokenizers/punkt/{self.language}.pickle")

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        return [segment.strip() for segment in self._tokenizer.tokenize(text) if segment.strip()]


@dataclass(slots=True)
class SpacySentenceSplitter:
    """Adapter that wraps spaCy pipelines for sentence boundary detection."""

    model_name: str = "en_core_web_sm"

    def __post_init__(self) -> None:
        try:
            import spacy
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ChunkerConfigurationError(
                "spaCy must be installed to use the SpacySentenceSplitter"
            ) from exc
        try:
            self._nlp = spacy.load(self.model_name)
        except OSError as exc:  # pragma: no cover - model missing
            raise ChunkerConfigurationError(
                f"spaCy model '{self.model_name}' is not installed"
            ) from exc

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        doc = self._nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


@dataclass(slots=True)
class PySBDSentenceSplitter:
    """Adapter around the PySBD rule-based splitter for English."""

    language: str = "en"

    def __post_init__(self) -> None:
        try:
            import pysbd
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ChunkerConfigurationError(
                "pysbd must be installed to use the PySBDSentenceSplitter"
            ) from exc
        self._segmenter = pysbd.Segmenter(language=self.language, clean=True)

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        segments = self._segmenter.segment(text)
        return [segment.strip() for segment in segments if segment.strip()]


def sentence_splitter_factory(name: str) -> SentenceSplitter:
    """Return a sentence splitter adapter based on configuration."""
    normalized = name.lower()
    if normalized in {"nltk", "punkt"}:
        return NLTKSentenceSplitter()
    if normalized == "spacy":
        return SpacySentenceSplitter()
    if normalized in {"pysbd", "py-sbd", "sbd"}:
        return PySBDSentenceSplitter()
    raise ChunkerConfigurationError(f"Unknown sentence splitter '{name}'")

