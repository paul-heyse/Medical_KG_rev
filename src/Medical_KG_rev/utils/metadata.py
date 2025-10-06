"""Metadata extraction helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable


def flatten_metadata(records: Iterable[Dict[str, Any]], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested metadata dictionaries."""

    flattened: Dict[str, Any] = {}
    for index, record in enumerate(records):
        for key, value in record.items():
            flattened_key = f"{prefix}{index}.{key}"
            flattened[flattened_key] = value
    return flattened
