"""Metadata extraction helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def flatten_metadata(records: Iterable[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metadata dictionaries."""
    flattened: dict[str, Any] = {}
    for index, record in enumerate(records):
        for key, value in record.items():
            flattened_key = f"{prefix}{index}.{key}"
            flattened[flattened_key] = value
    return flattened
