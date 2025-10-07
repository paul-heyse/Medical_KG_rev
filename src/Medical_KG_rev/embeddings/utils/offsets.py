"""Utilities for generating token offset maps used by framework adapters."""

from __future__ import annotations

from typing import Iterable, Mapping


def compute_offsets(text: str) -> list[Mapping[str, int | str]]:
    offsets: list[Mapping[str, int | str]] = []
    cursor = 0
    lowered = text.lower()
    for token in text.split():
        token_lower = token.lower()
        start = lowered.find(token_lower, cursor)
        if start == -1:
            start = cursor
        end = start + len(token)
        offsets.append({"token": token, "start": start, "end": end})
        cursor = end
    return offsets


def batch_offsets(texts: Iterable[str]) -> list[list[Mapping[str, int | str]]]:
    return [compute_offsets(text) for text in texts]
