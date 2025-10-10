"""Sentence segmentation wrapper powered by syntok."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache

Segment = tuple[int, int, str]


class SyntokSentenceSegmenter:
    """Lightweight syntok-based sentence segmenter with offset tracking."""

    def __init__(
        self,
        analyzer_factory: Callable[[], Callable[[str], Iterable[Iterable[object]]]] | None = None,
    ) -> None:
        self._analyzer_factory = analyzer_factory or _cached_analyzer

    def segment(self, text: str) -> list[Segment]:
        analyze = self._analyzer_factory()
        segments: list[Segment] = []
        cursor = 0
        for paragraph in analyze(text):
            for sentence in paragraph:
                rendered = "".join(
                    getattr(token, "spacing", "") + getattr(token, "value", "")
                    for token in sentence
                )
                rendered = rendered.strip()
                if not rendered:
                    continue
                start = text.find(rendered, cursor)
                if start == -1:
                    start = cursor
                end = start + len(rendered)
                segments.append((start, end, text[start:end]))
                cursor = end
        return segments


@lru_cache(maxsize=1)
def _cached_analyzer() -> (
    Callable[[str], Iterable[Iterable[object]]]
):  # pragma: no cover - heavy dependency
    try:
        from syntok import segmenter
    except ImportError as exc:  # pragma: no cover - executed when dependency missing
        raise RuntimeError("syntok is not installed. Install syntok>=1.4.4") from exc

    segmentation = segmenter.Segmentation()
    return segmentation.analyze


__all__ = ["SyntokSentenceSegmenter"]
