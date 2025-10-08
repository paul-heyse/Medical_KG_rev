"""Sentence splitter adapters."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Tuple

Segment = Tuple[int, int, str]


def get_sentence_splitter(name: str) -> Callable[[str], List[Segment]]:
    name = name.lower()
    if name == "scispacy":
        return _scispacy_split
    if name == "syntok":
        return _syntok_split
    return _simple_split


@lru_cache(maxsize=1)
def _scispacy_segmenter():  # pragma: no cover - heavy dependency path
    from .wrappers.scispacy_segmenter import SciSpaCySentenceSegmenter

    return SciSpaCySentenceSegmenter()


def _scispacy_split(text: str) -> List[Segment]:
    segmenter = _scispacy_segmenter()
    return segmenter.segment(text)


@lru_cache(maxsize=1)
def _syntok_segmenter():  # pragma: no cover - heavy dependency path
    from .wrappers.syntok_segmenter import SyntokSentenceSegmenter

    return SyntokSentenceSegmenter()


def _syntok_split(text: str) -> List[Segment]:
    segmenter = _syntok_segmenter()
    return segmenter.segment(text)


def _simple_split(text: str) -> List[Segment]:
    sentences: List[Segment] = []
    cursor = 0
    for part in [segment.strip() for segment in text.split(". ") if segment.strip()]:
        idx = text.find(part, cursor)
        if idx == -1:
            idx = cursor
        start = idx
        end = start + len(part)
        sentences.append((start, end, text[start:end]))
        cursor = end
    return sentences
