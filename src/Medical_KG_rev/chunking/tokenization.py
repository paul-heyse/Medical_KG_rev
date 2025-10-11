"""Token counting helpers backed by tiktoken."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

try:
    import tiktoken
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ImportError("tiktoken is required for token counting") from exc


class TokenCounter:
    """Wrapper around tiktoken for token counting."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            self._encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except KeyError:
            self._encoder = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def count_many(self, texts: Iterable[str]) -> int:
        return sum(self.count(text) for text in texts)

    def trim_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self._encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoder.decode(tokens[:max_tokens])


@lru_cache(maxsize=4)
def default_token_counter() -> TokenCounter:
    return TokenCounter()
