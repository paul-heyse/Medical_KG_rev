"""Sentence segmentation backed by Hugging Face tokenizers."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import List, Tuple

Segment = Tuple[int, int, str]

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer  # type: ignore
except ImportError as exc:  # pragma: no cover - fallback heuristic
    raise ImportError("transformers is required for Hugging Face sentence segmentation") from exc


@lru_cache(maxsize=1)
def _default_loader() -> Callable[[str], List[Segment]]:
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    except Exception as exc:  # pragma: no cover - optional dependency failure
        raise RuntimeError(
            "Failed to load Hugging Face tokenizer for sentence segmentation"
        ) from exc

    def splitter(text: str) -> List[Segment]:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise RuntimeError("Tokenizer backend does not expose pre_tokenizer API")
        spans: List[Segment] = []
        current_start: int | None = None
        for token, (start, end) in backend.pre_tokenizer.pre_tokenize_str(text):
            if current_start is None:
                current_start = start
            if token.strip().endswith(('.', '?', '!')):
                spans.append((current_start, end, text[current_start:end]))
                current_start = None
        if current_start is not None:
            spans.append((current_start, len(text), text[current_start:]))
        return spans

    return splitter


class HuggingFaceSentenceSegmenter:
    """Expose a simple ``segment`` API backed by Hugging Face tokenizers."""

    def __init__(self, loader: Callable[[], Callable[[str], List[Segment]]] | None = None) -> None:
        self._loader = loader or _default_loader

    def segment(self, text: str) -> List[Segment]:
        return self._loader()(text)


__all__ = ["HuggingFaceSentenceSegmenter", "Segment"]
