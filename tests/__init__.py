"""Test package configuration for Medical_KG_rev."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the src/ directory is available on sys.path so imports work in tests without
# relying on editable installs. This mirrors the layout used in development and CI.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_PATH = _PROJECT_ROOT / "src"
if _SRC_PATH.is_dir():
    src_str = str(_SRC_PATH)
    if src_str not in sys.path:
        # Prepend so local modules take priority over installed packages.
        sys.path.insert(0, src_str)

# Some test helpers expect a default settings module hint. Align with application
# defaults when the environment variable is missing to reduce boilerplate in tests.
os.environ.setdefault("MK_SETTINGS_MODULE", "Medical_KG_rev.config.settings")

__all__ = [
    "_PROJECT_ROOT",
    "_SRC_PATH",
]
