"""Sentence segmentation backed by Hugging Face models with offset preservation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache

Segment = tuple[int, int, str]


def _default_loader() -> Callable[[str], Iterable[object]]:  # pragma: no cover - heavy dependency
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - executed when dependency missing
        raise RuntimeError(
            "transformers is not installed. Install transformers for Hugging Face models."
        ) from exc

    # Use a biomedical text classification model for sentence segmentation
    # This is a skeletal implementation - you may want to use a different model
    # or approach depending on your specific biomedical text classification needs
    try:
        # Using a general-purpose model for now - replace with biomedical-specific model
        classifier = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",  # Placeholder - replace with biomedical model
            return_all_scores=False
        )
        return classifier
    except Exception as exc:  # pragma: no cover - executed when model loading fails
        raise RuntimeError(
            "Failed to load Hugging Face model. Check model availability and dependencies."
        ) from exc


class HuggingFaceSentenceSegmenter:
    """Biomedical-aware sentence segmenter using Hugging Face models."""

    def __init__(self, loader: Callable[[], Callable[[str], Iterable[object]]] | None = None) -> None:
        self._loader = loader or _cached_loader

    def segment(self, text: str) -> list[Segment]:
        """
        Segment text into sentences using Hugging Face models.

        This is a skeletal implementation that provides the same interface
        as the scispacy segmenter but uses Hugging Face models instead.

        Args:
            text: Input text to segment

        Returns:
            List of segments as (start, end, text) tuples
        """
        # For now, implement a simple fallback segmentation
        # TODO: Replace with actual Hugging Face model-based segmentation
        segments: list[Segment] = []

        # Simple sentence splitting as fallback
        sentences = text.split('. ')
        cursor = 0

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Find the sentence in the original text
            idx = text.find(sentence, cursor)
            if idx == -1:
                idx = cursor

            start = idx
            end = start + len(sentence)

            # Add period if not the last sentence and doesn't end with punctuation
            if i < len(sentences) - 1 and not sentence.rstrip().endswith(('.', '!', '?')):
                end += 1

            start, end = _trim_offsets(text, start, end)
            if start >= end:
                continue

            segments.append((start, end, text[start:end]))
            cursor = end

        return segments


@lru_cache(maxsize=1)
def _cached_loader() -> Callable[[str], Iterable[object]]:
    return _default_loader()


def _trim_offsets(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim whitespace from segment boundaries."""
    end = min(len(text), max(start, end))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


__all__ = ["HuggingFaceSentenceSegmenter"]
