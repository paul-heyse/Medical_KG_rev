"""Tokenizer helpers for chunking wrappers."""

from __future__ import annotations

from typing import Callable

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]


def load_tokenizer(model_name: str) -> Callable[[str], list[str]]:
    """Return a callable that tokenizes text."""
    if AutoTokenizer is None:
        return lambda text: text.split()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer.tokenize


__all__ = ["load_tokenizer"]
