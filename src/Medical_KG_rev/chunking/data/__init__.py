"""Data helpers for chunking module."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

# ==============================================================================
# RESOURCE HELPERS
# ==============================================================================


def load_json_resource(package: str, resource_name: str) -> Any:
    """Load a JSON resource bundled with the package.

    Args:
        package: Dotted path to the package containing the resource.
        resource_name: File name of the JSON resource.

    Returns:
        Parsed JSON payload or an empty dictionary when unavailable.
    """
    try:
        file_ref = resources.files(package).joinpath(resource_name)
    except AttributeError:  # Python <3.9 or missing importlib resources
        return {}
    if not file_ref.is_file():
        return {}
    with file_ref.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["load_json_resource"]
