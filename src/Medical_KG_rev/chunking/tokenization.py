"""Token counting helpers backed by tiktoken."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

try:  # pragma: no cover - optional dependency
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore[assignment]


class TokenCounter:
    """Wrapper around tiktoken with a deterministic fallback implementation."""

    def __init__(
        self,
        encoding: str = "cl100k_base",
        *,
        fallback_chars_per_token: float = 4.0,
    ) -> None:
        self._encoder = None
        self._fallback_chars_per_token = max(fallback_chars_per_token, 1.0)
        if tiktoken is not None:  # pragma: no branch - simple optional import guard
            try:
                self._encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except KeyError:
                self._encoder = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is None:
            return max(1, int(len(text) / self._fallback_chars_per_token))
        return len(self._encoder.encode(text))

    def count_many(self, texts: Iterable[str]) -> int:
        return sum(self.count(text) for text in texts)

    def trim_to_tokens(self, text: str, max_tokens: int) -> str:
        if self._encoder is None:
            if not text or max_tokens <= 0:
                return ""
            max_chars = int(max_tokens * self._fallback_chars_per_token)
            if max_chars <= 0:
                return ""
            return text if len(text) <= max_chars else text[:max_chars]
        tokens = self._encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoder.decode(tokens[:max_tokens])


@lru_cache(maxsize=4)
def default_token_counter() -> TokenCounter:
    return TokenCounter()
