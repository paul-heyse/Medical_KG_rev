"""Token counting helpers that wrap Transformers and tiktoken."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_TIKTOKEN_MODEL = "gpt-4o-mini"


def _load_hf_tokenizer(model_name: str):  # pragma: no cover - heavy dependency
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def _load_tiktoken(model_name: str):  # pragma: no cover - heavy dependency
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=None)
def get_hf_tokenizer(model_name: str = DEFAULT_HF_MODEL):
    return _load_hf_tokenizer(model_name)


@lru_cache(maxsize=None)
def get_tiktoken_encoder(model_name: str = DEFAULT_TIKTOKEN_MODEL):
    return _load_tiktoken(model_name)


def count_tokens_hf(text: str, model_name: str = DEFAULT_HF_MODEL) -> int:
    tokenizer = get_hf_tokenizer(model_name)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def count_tokens_tiktoken(text: str, model_name: str = DEFAULT_TIKTOKEN_MODEL) -> int:
    encoder = get_tiktoken_encoder(model_name)
    return len(encoder.encode(text))


def ensure_within_budget(
    text: str, *, budget: int, counter: Callable[[str], int] = count_tokens_hf
) -> bool:
    return counter(text) <= budget


__all__ = [
    "DEFAULT_HF_MODEL",
    "DEFAULT_TIKTOKEN_MODEL",
    "count_tokens_hf",
    "count_tokens_tiktoken",
    "ensure_within_budget",
    "get_hf_tokenizer",
    "get_tiktoken_encoder",
]
