"""Sentence segmentation backed by scispaCy with offset preservation."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable, List, Tuple

Segment = Tuple[int, int, str]


def _default_loader() -> Callable[[str], Iterable[object]]:  # pragma: no cover - heavy dependency
    try:
        import scispacy  # noqa: F401
        import spacy
    except ImportError as exc:  # pragma: no cover - executed when dependency missing
        raise RuntimeError(
            "scispaCy is not installed. Install scispacy and en_core_sci_sm."
        ) from exc

    model_name = "en_core_sci_sm"
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - executed when model missing
        raise RuntimeError(
            "The en_core_sci_sm model is required. Run 'python -m spacy download en_core_sci_sm'."
        ) from exc

    return nlp


class SciSpaCySentenceSegmenter:
    """Biomedical-aware sentence segmenter using scispaCy."""

    def __init__(self, loader: Callable[[], Callable[[str], Iterable[object]]] | None = None) -> None:
        self._loader = loader or _cached_loader

    def segment(self, text: str) -> List[Segment]:
        nlp = self._loader()
        doc = nlp(text)
        segments: List[Segment] = []
        for sent in doc.sents:  # type: ignore[attr-defined]
            start = getattr(sent, "start_char", 0)
            end = getattr(sent, "end_char", start)
            start, end = _trim_offsets(text, start, end)
            if start >= end:
                continue
            segments.append((start, end, text[start:end]))
        return segments


@lru_cache(maxsize=1)
def _cached_loader() -> Callable[[str], Iterable[object]]:
    return _default_loader()


def _trim_offsets(text: str, start: int, end: int) -> Tuple[int, int]:
    end = min(len(text), max(start, end))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


__all__ = ["SciSpaCySentenceSegmenter"]
