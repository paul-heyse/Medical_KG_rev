"""Metadata extraction helpers.

Key Responsibilities:
    - Provide utilities for flattening nested metadata structures into
      telemetry-friendly key/value pairs

Collaborators:
    - Upstream: Observability instrumentation and logging helpers
    - Downstream: Metrics and logging pipelines consume flattened metadata

Side Effects:
    - None; operations are purely functional

Thread Safety:
    - Thread-safe; helpers operate on caller-provided data only
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# ==============================================================================
# PUBLIC HELPERS
# ==============================================================================


def flatten_metadata(records: Iterable[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metadata dictionaries.

    Args:
        records: Iterable of metadata dictionaries where each entry represents
            structured information for a particular component.
        prefix: Optional prefix applied to flattened keys (e.g. ``"chunk."``).

    Returns:
        Dictionary mapping flattened metadata keys to their corresponding
        values.
    """
    flattened: dict[str, Any] = {}
    for index, record in enumerate(records):
        for key, value in record.items():
            flattened_key = f"{prefix}{index}.{key}"
            flattened[flattened_key] = value
    return flattened
