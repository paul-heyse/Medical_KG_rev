"""Authentication and authorization helpers for the gateway.

This package exposes reusable primitives for request authentication, scope
management, and rate limiting. It gracefully degrades when FastAPI is not
available so that non-HTTP contexts (e.g., batch jobs) can import shared data
structures without pulling optional dependencies.
"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================
import importlib.util
from collections.abc import Callable
from typing import Any

from .context import SecurityContext
from .scopes import Scopes

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["Scopes", "SecurityContext"]


# ============================================================================
# OPTIONAL FASTAPI INTEGRATION
# ============================================================================

if importlib.util.find_spec("fastapi") is not None:
    from .dependencies import get_security_context, secure_endpoint  # type: ignore

    __all__.extend(["get_security_context", "secure_endpoint"])
else:  # pragma: no cover - optional dependency fallback

    def get_security_context(*args: Any, **kwargs: Any) -> SecurityContext:
        """Placeholder that signals FastAPI is required for authentication hooks.

        Raises:
            RuntimeError: Always raised because FastAPI is unavailable.

        """
        raise RuntimeError("FastAPI is required for get_security_context")

    def secure_endpoint(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Placeholder that signals FastAPI is required to secure endpoints.

        Raises:
            RuntimeError: Always raised because FastAPI is unavailable.

        """
        raise RuntimeError("FastAPI is required to secure endpoints")

    __all__.extend(["get_security_context", "secure_endpoint"])
