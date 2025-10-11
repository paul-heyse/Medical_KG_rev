"""Syntok sentence segmentation wrapper."""

from __future__ import annotations

from typing import List, Tuple

Segment = Tuple[int, int, str]

try:  # pragma: no cover - optional dependency
    from syntok.segmenter import Tokenizer, seg
except ImportError as exc:  # pragma: no cover - fallback
    raise ImportError("syntok is required for syntok sentence segmentation") from exc


class SyntokSentenceSegmenter:
    """Expose a ``segment`` API backed by syntok when available."""

    def __init__(self) -> None:
        self._tokenizer = Tokenizer()

    def segment(self, text: str) -> List[Segment]:
        if not text.strip():
            return []
        segments: List[Segment] = []
        cursor = 0
        for paragraphs in seg(self._tokenizer(text)):
            for sentence in paragraphs:
                start = sentence[0].offset
                end = sentence[-1].offset + len(sentence[-1].value)
                segments.append((start, end, text[start:end]))
                cursor = end
        if not segments:
            raise RuntimeError("syntok failed to produce sentence boundaries")
        return segments


__all__ = ["SyntokSentenceSegmenter", "Segment"]
