"""Authentication and authorization helpers for the gateway."""

from __future__ import annotations

import importlib.util
from typing import Any, Callable

from .context import SecurityContext
from .scopes import Scopes

__all__ = ["Scopes", "SecurityContext"]

if importlib.util.find_spec("fastapi") is not None:
    from .dependencies import get_security_context, secure_endpoint  # type: ignore

    __all__.extend(["get_security_context", "secure_endpoint"])
else:  # pragma: no cover - optional dependency fallback

    def get_security_context(*args: Any, **kwargs: Any) -> SecurityContext:
        raise RuntimeError("FastAPI is required for get_security_context")

    def secure_endpoint(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        raise RuntimeError("FastAPI is required to secure endpoints")

    __all__.extend(["get_security_context", "secure_endpoint"])
