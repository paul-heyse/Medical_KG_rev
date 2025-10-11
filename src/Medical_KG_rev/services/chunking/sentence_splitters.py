"""Sentence splitter adapters."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import List, Tuple

from .wrappers.huggingface_segmenter import HuggingFaceSentenceSegmenter
from .wrappers.syntok_segmenter import SyntokSentenceSegmenter

Segment = Tuple[int, int, str]


@lru_cache(maxsize=1)
def _syntok_segmenter() -> SyntokSentenceSegmenter:
    return SyntokSentenceSegmenter()


def syntok_split(text: str) -> List[Segment]:
    segmenter = _syntok_segmenter()
    return segmenter.segment(text)


def simple_split(text: str) -> List[Segment]:
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


DEFAULT_SPLITTER: Callable[[str], List[Segment]] = syntok_split


__all__ = ["Segment", "syntok_split", "simple_split", "DEFAULT_SPLITTER"]
