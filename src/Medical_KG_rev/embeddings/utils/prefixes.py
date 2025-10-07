"""Helpers for applying query/document prefixes such as E5 models."""

from __future__ import annotations

from collections.abc import Sequence


def apply_prefixes(
    texts: Sequence[str],
    *,
    prefix: str | None = None,
) -> list[str]:
    if not prefix:
        return list(texts)
    trimmed = prefix.strip()
    if trimmed and not trimmed.endswith(" "):
        trimmed += " "
    return [f"{trimmed}{text}" for text in texts]
