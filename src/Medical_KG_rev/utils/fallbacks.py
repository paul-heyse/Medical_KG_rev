"""Centralize helpers for disabling runtime fallbacks.

Key Responsibilities:
    - Provide a canonical exception for prohibited fallback code paths.
    - Offer a helper that assembles consistent error messages.

Collaborators:
    - Upstream: Modules that previously relied on silent degradation.

Side Effects:
    - Raises `FallbackNotAllowedError` to halt execution.

Thread Safety:
    - Thread-safe: Pure helper functions without shared state.
"""

from __future__ import annotations

from typing import NoReturn


class FallbackNotAllowedError(RuntimeError):
    """Raised when code attempts to use a degraded fallback path."""


def _describe(reason: str | BaseException | None) -> str:
    if reason is None:
        return ""
    if isinstance(reason, BaseException):
        return f": {reason}"
    if reason:
        return f": {reason}"
    return ""


def fallback_unavailable(component: str, reason: str | BaseException | None = None) -> NoReturn:
    """Raise an error indicating a fallback path is prohibited.

    Args:
        component: Logical component that attempted to fallback.
        reason: Optional context or original exception.
    """
    message = (
        f"{component} fallback is disabled outside production readiness"
        f"; address the underlying issue{_describe(reason)}"
    )
    raise FallbackNotAllowedError(message)


__all__ = ["FallbackNotAllowedError", "fallback_unavailable"]
