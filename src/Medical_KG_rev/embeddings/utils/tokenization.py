"""Tokenization utilities aligned with embedding model vocabularies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


class TokenizerUnavailableError(RuntimeError):
    """Raised when a requested tokenizer cannot be loaded."""


class TokenLimitExceededError(RuntimeError):
    """Raised when a text exceeds the configured token budget."""


@dataclass(slots=True)
class _TokenizerWrapper:
    """Lazy loader for HuggingFace tokenizers with caching."""

    model_id: str
    _tokenizer: object | None = None

    def _load(self) -> object:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer  # type: ignore import-not-found
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise TokenizerUnavailableError(
                    "transformers is required for tokenizer-backed token counting"
                ) from exc
            logger.info("embedding.tokenizer.load", model=self.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    def count(self, text: str) -> int:
        tokenizer = self._load()
        encode = tokenizer.encode
        tokens = encode(text, add_special_tokens=False)
        return len(tokens)


@dataclass(slots=True)
class TokenizerCache:
    """Caches tokenizers per model id and validates token budgets."""

    _tokenizers: dict[str, _TokenizerWrapper] = field(default_factory=dict)

    def ensure_within_limit(
        self,
        *,
        model_id: str,
        texts: Sequence[str],
        max_tokens: int | None,
        correlation_id: str | None = None,
    ) -> None:
        if max_tokens is None:
            return
        wrapper = self._tokenizers.setdefault(model_id, _TokenizerWrapper(model_id))
        for index, text in enumerate(texts):
            token_count = wrapper.count(text)
            if token_count > max_tokens:
                logger.error(
                    "embedding.tokenizer.limit_exceeded",
                    model=model_id,
                    tokens=token_count,
                    max_tokens=max_tokens,
                    correlation_id=correlation_id,
                    chunk_index=index,
                )
                raise TokenLimitExceededError(
                    f"Text has {token_count} tokens, max {max_tokens} for model {model_id}"
                )
        logger.debug(
            "embedding.tokenizer.verified",
            model=model_id,
            max_tokens=max_tokens,
            chunks=len(texts),
        )


__all__ = [
    "TokenLimitExceededError",
    "TokenizerCache",
    "TokenizerUnavailableError",
]
