"""Sentence splitter adapters."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List


def get_sentence_splitter(name: str) -> Callable[[str], List[str]]:
    name = name.lower()
    if name == "scispacy":
        return _scispacy_split
    if name == "syntok":
        return _syntok_split
    return _simple_split


@lru_cache(maxsize=1)
def _load_scispacy_model():  # pragma: no cover - heavy dependency path
    try:
        import scispacy  # noqa: F401
        import spacy
    except ImportError as exc:  # pragma: no cover - executed when dependency missing
        raise RuntimeError(
            "scispaCy is not installed. Install scispacy and en_core_sci_sm."
        ) from exc
    try:
        return spacy.load("en_core_sci_sm")
    except OSError as exc:  # pragma: no cover - executed if model missing
        raise RuntimeError(
            "The en_core_sci_sm model is required. Run 'python -m spacy download en_core_sci_sm'."
        ) from exc


def _scispacy_split(text: str) -> List[str]:
    nlp = _load_scispacy_model()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _syntok_split(text: str) -> List[str]:
    try:
        from syntok.segmenter import Segmentation  # type: ignore
    except ImportError as exc:  # pragma: no cover - executed when dependency missing
        raise RuntimeError("syntok is not installed. Install syntok>=1.4.4") from exc
    segmenter = Segmentation()
    sentences: List[str] = []
    for paragraph in segmenter.analyze(text):
        for sentence in paragraph:
            sentences.append("".join(token.spacing + token.value for token in sentence).strip())
    return [sent for sent in sentences if sent]


def _simple_split(text: str) -> List[str]:
    return [sent.strip() for sent in text.split(". ") if sent.strip()]
