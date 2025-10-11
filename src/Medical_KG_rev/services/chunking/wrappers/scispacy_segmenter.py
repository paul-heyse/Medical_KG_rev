"""SciSpaCy-based sentence segmentation (placeholder implementation)."""

from __future__ import annotations

from typing import List, Tuple

Segment = Tuple[int, int, str]

try:  # pragma: no cover - optional dependency
    import scispacy  # type: ignore  # noqa: F401
    import spacy
except ImportError as exc:  # pragma: no cover - fallback heuristic
    raise ImportError("scispacy and spaCy are required for SciSpaCy sentence segmentation") from exc


class SciSpacySentenceSegmenter:
    """Expose a ``segment`` API compatible with the historical implementation."""

    def __init__(self, model: str = "en_core_sci_sm") -> None:
        try:
            self._nlp = spacy.load(model)
        except Exception as exc:  # pragma: no cover - optional dependency failure
            raise RuntimeError(f"Failed to load SciSpaCy model '{model}'") from exc

    def segment(self, text: str) -> List[Segment]:
        if not text.strip():
            return []
        doc = self._nlp(text)
        return [(sent.start_char, sent.end_char, sent.text) for sent in doc.sents]


__all__ = ["SciSpacySentenceSegmenter", "Segment"]
