"""Identifier utilities for deterministic document ids."""

from __future__ import annotations

import hashlib
import secrets



def hash_content(content: str) -> str:
    """Return a stable 12 character hash for the provided content."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return digest[:12]


def build_document_id(
    source: str, source_id: str, version: str = "v1", content: str | None = None
) -> str:
    """Construct a globally unique identifier following the design convention."""
    if content:
        suffix = hash_content(content)
    else:
        suffix = secrets.token_hex(6)
    return f"{source}:{source_id}#{version}:{suffix}"


def normalize_identifier(value: str) -> str:
    """Normalize identifiers to lowercase without whitespace."""
    return value.strip().lower()
