"""Sentence segmentation backed by Hugging Face tokenizers."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import transformers

Segment = tuple[int, int, str]

# Common biomedical abbreviations that should not terminate a sentence even
# though they end with a period.
_ABBREVIATION_SUFFIXES: tuple[str, ...] = (
    "Fig.",
    "Figs.",
    "Dr.",
    "Prof.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "No.",
    "vs.",
    "et al.",
    "Eq.",
)


def _default_loader() -> Callable[[str], list[Segment]]:  # pragma: no cover - heavy dependency path
    """Load a Hugging Face tokenizer and return a callable segmenter."""
    model_name = os.getenv("MEDICAL_KG_SENTENCE_MODEL")
    if not model_name:
        warnings.warn(
            "MEDICAL_KG_SENTENCE_MODEL is not set; falling back to heuristic "
            "sentence splitting. Configure a Hugging Face model for "
            "production deployments.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _HeuristicSentenceSplitter()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - executed when dependency missing
        raise RuntimeError(
            "transformers is required for Hugging Face sentence segmentation"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:  # pragma: no cover - executed when download fails
        raise RuntimeError(
            f"Failed to load Hugging Face tokenizer '{model_name}': {exc}"
        ) from exc

    return _TokenizerSentenceSplitter(tokenizer)


@dataclass
class _TokenizerSentenceSplitter:
    """Adapter that uses a fast Hugging Face tokenizer to locate sentences."""

    tokenizer: transformers.PreTrainedTokenizerBase

    def __call__(self, text: str) -> list[Segment]:
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend is None:  # pragma: no cover - only happens with slow tokenizers
            raise RuntimeError(
                "The selected tokenizer does not expose a fast backend."
            )

        tokens = backend.pre_tokenizer.pre_tokenize_str(text)
        if not tokens:
            return []

        spans: list[tuple[int, int]] = []
        current_start: int | None = None
        for token, (start, end) in tokens:
            if current_start is None:
                current_start = start

            if _should_close_sentence(token, text, start, end):
                spans.append((current_start, end))
                current_start = None

        if current_start is not None:
            spans.append((current_start, len(text)))

        segments: list[Segment] = []
        for start, end in spans:
            trimmed_start, trimmed_end = _trim_offsets(text, start, end)
            if trimmed_start >= trimmed_end:
                continue
            segments.append((trimmed_start, trimmed_end, text[trimmed_start:trimmed_end]))

        return _merge_abbreviation_segments(text, segments)


class _HeuristicSentenceSplitter:
    """Lightweight fallback that mirrors the legacy heuristic splitter."""

    def __call__(self, text: str) -> list[Segment]:
        sentences: list[Segment] = []
        cursor = 0
        for part in [segment.strip() for segment in text.split(". ") if segment.strip()]:
            idx = text.find(part, cursor)
            if idx == -1:
                idx = cursor
            start = idx
            end = start + len(part)
            sentences.append((start, end, text[start:end]))
            cursor = end
        if not sentences and text.strip():
            stripped = text.strip()
            idx = text.find(stripped)
            sentences.append((idx if idx >= 0 else 0, idx + len(stripped), stripped))
        return sentences


def _should_close_sentence(token: str, text: str, start: int, end: int) -> bool:
    stripped = token.strip()
    if not stripped:
        return False

    last_char = stripped[-1]
    if last_char in {".", "?", "!"}:
        if _is_abbreviation(text[:end]):
            return False
        return True

    # Handle explicit sentence separators used by some tokenizers.
    if stripped in {"</s>", "<s>"}:
        return True

    # Break on double newlines which often separate paragraphs.
    if text[end:end + 2] == "\n\n":
        return True

    return False


def _trim_offsets(text: str, start: int, end: int) -> tuple[int, int]:
    end = min(len(text), max(start, end))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _merge_abbreviation_segments(text: str, segments: Sequence[Segment]) -> list[Segment]:
    if not segments:
        return list(segments)

    merged: list[Segment] = []
    index = 0
    while index < len(segments):
        start, end, sentence = segments[index]
        if _is_abbreviation(sentence) and index + 1 < len(segments):
            next_start, next_end, _ = segments[index + 1]
            combined = text[start:next_end]
            merged.append((start, next_end, combined))
            index += 2
            continue
        merged.append((start, end, sentence))
        index += 1
    return merged


def _is_abbreviation(sentence: str) -> bool:
    stripped = sentence.strip()
    return any(stripped.endswith(suffix) for suffix in _ABBREVIATION_SUFFIXES)


@lru_cache(maxsize=1)
def default_segmenter() -> Callable[[str], list[Segment]]:
    return _default_loader()


class HuggingFaceSentenceSegmenter:
    """Expose a simple ``segment`` API backed by Hugging Face tokenizers."""

    def __init__(self, loader: Callable[[], Callable[[str], list[Segment]]] | None = None) -> None:
        self._loader = loader or default_segmenter

    def segment(self, text: str) -> list[Segment]:
        segmenter = self._loader()
        segments = segmenter(text)
        return _merge_abbreviation_segments(text, segments)


__all__ = ["HuggingFaceSentenceSegmenter"]
