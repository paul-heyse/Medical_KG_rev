"""Custom exceptions for the adapter plugin framework."""

from __future__ import annotations


class AdapterPluginError(RuntimeError):
    """Raised when adapter plugin operations fail."""


__all__ = ["AdapterPluginError"]
