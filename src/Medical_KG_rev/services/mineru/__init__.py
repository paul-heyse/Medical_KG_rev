"""Legacy MinerU implementation (archived)."""

from __future__ import annotations

import importlib
import warnings
from types import ModuleType
from typing import Any

warnings.warn(
    "The MinerU service package is archived and will be removed in a future release. "
    "Switch to Docling VLM services.",
    DeprecationWarning,
    stacklevel=2,
)

_archive_module: ModuleType | None = None


def _load_archive() -> ModuleType:
    global _archive_module
    if _archive_module is None:
        _archive_module = importlib.import_module("Medical_KG_rev.services.mineru.archive")
    return _archive_module


def __getattr__(name: str) -> Any:  # pragma: no cover - compatibility shim
    module = _load_archive()
    return getattr(module, name)


__all__ = [
    name
    for name in dir(_load_archive())
    if not name.startswith("_")
]
