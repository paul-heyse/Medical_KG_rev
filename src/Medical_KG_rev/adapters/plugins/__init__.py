"""Adapter plugin conveniences."""

from __future__ import annotations

from typing import Any

hookimpl = lambda func: func  # type: ignore
hookspec = lambda func: func  # type: ignore


__all__ = ["hookimpl", "hookspec"]
