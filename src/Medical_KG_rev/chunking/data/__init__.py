"""Data helpers for chunking module."""

from __future__ import annotations

from importlib import resources
from typing import Any


def load_json_resource(name: str) -> Any:
    with resources.files(__package__).joinpath(name).open("r", encoding="utf-8") as handle:
        import json

        return json.load(handle)

