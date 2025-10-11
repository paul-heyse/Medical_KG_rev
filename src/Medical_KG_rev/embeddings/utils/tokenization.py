"""Tokenization utilities for embeddings."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


class TokenizerUnavailableError(Exception):
    """Raised when tokenizer is not available."""

    pass


class TokenCounter(ABC):
    """Abstract base class for token counters."""

    @abstractmethod
    def count(self, text: str) -> int:
        """Count tokens in text."""
        pass


class SimpleTokenCounter(TokenCounter):
    """Simple word-based token counter."""

    def count(self, text: str) -> int:
        """Count tokens as words."""
        if not text:
            return 0
        return len(text.split())


class CharacterTokenCounter(TokenCounter):
    """Character-based token counter."""

    def count(self, text: str) -> int:
        """Count tokens as characters."""
        if not text:
            return 0
        return len(text)


class TransformersTokenCounter(TokenCounter):
    """Token counter using transformers tokenizer."""

    def __init__(self, model_id: str = "bert-base-uncased") -> None:
        """Initialize the transformers token counter."""
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise TokenizerUnavailableError(
                "transformers is required for tokenizer-backed token counting"
            ) from exc

        self.model_id = model_id
        self._tokenizer = None
        logger.info("embedding.tokenizer.load", model=self.model_id)

    def _get_tokenizer(self):
        """Get or create tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    def count(self, text: str) -> int:
        """Count tokens using transformers tokenizer."""
        if not text:
            return 0

        try:
            tokenizer = self._get_tokenizer()
            tokens = tokenizer.tokenize(text)
            return len(tokens)
        except Exception as exc:
            logger.warning("tokenizer.count_failed", error=str(exc))
            # Fallback to simple word counting
            return len(text.split())


@dataclass
class TokenizationConfig:
    """Configuration for tokenization."""

    method: str = "simple"
    model_id: str = "bert-base-uncased"
    max_length: int = 512
    truncation: bool = True
    padding: bool = False


class TokenizationService:
    """Service for text tokenization."""

    def __init__(self, config: TokenizationConfig | None = None) -> None:
        """Initialize the tokenization service."""
        self.config = config or TokenizationConfig()
        self._tokenizer = None

    def _get_tokenizer(self):
        """Get or create tokenizer."""
        if self._tokenizer is None:
            if self.config.method == "transformers":
                try:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
                except ImportError:
                    logger.warning("transformers not available, falling back to simple tokenization")
                    self._tokenizer = SimpleTokenCounter()
            else:
                self._tokenizer = SimpleTokenCounter()
        return self._tokenizer

    def tokenize(self, text: str) -> Sequence[str]:
        """Tokenize text."""
        if not text:
            return []

        tokenizer = self._get_tokenizer()

        if isinstance(tokenizer, TransformersTokenCounter):
            try:
                tokens = tokenizer._get_tokenizer().tokenize(text)
                return tokens
            except Exception as exc:
                logger.warning("tokenization.failed", error=str(exc))
                return text.split()
        else:
            return text.split()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0

        tokenizer = self._get_tokenizer()

        if isinstance(tokenizer, TokenCounter):
            return tokenizer.count(text)
        else:
            return len(text.split())

    def truncate(self, text: str, max_length: int | None = None) -> str:
        """Truncate text to maximum length."""
        if not text:
            return text

        max_len = max_length or self.config.max_length
        tokens = self.tokenize(text)

        if len(tokens) <= max_len:
            return text

        truncated_tokens = tokens[:max_len]

        if isinstance(self._tokenizer, TransformersTokenCounter):
            try:
                tokenizer = self._tokenizer._get_tokenizer()
                return tokenizer.convert_tokens_to_string(truncated_tokens)
            except Exception:
                return " ".join(truncated_tokens)
        else:
            return " ".join(truncated_tokens)


def create_token_counter(method: str = "simple", model_id: str = "bert-base-uncased") -> TokenCounter:
    """Create a token counter by method."""
    if method == "transformers":
        return TransformersTokenCounter(model_id)
    elif method == "character":
        return CharacterTokenCounter()
    else:
        return SimpleTokenCounter()


def default_token_counter() -> TokenCounter:
    """Get the default token counter."""
    return SimpleTokenCounter()
