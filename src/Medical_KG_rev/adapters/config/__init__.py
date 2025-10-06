"""Packaged adapter configuration templates."""

from collections.abc import Iterator
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent


def list_configs() -> Iterator[str]:
    """Yield available configuration resource names."""

    for resource in _BASE_DIR.iterdir():
        if resource.suffix == ".yaml":
            yield resource.name


def config_path(name: str) -> Path:
    """Return the path to a bundled configuration file."""

    return _BASE_DIR / name
