"""Error utilities implementing RFC 7807 problem details."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import importlib.util

_PYDANTIC_AVAILABLE = importlib.util.find_spec("pydantic") is not None

if _PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, Field  # type: ignore

    class ProblemDetail(BaseModel):
        """Representation of RFC 7807 problem details object."""

        type: str = Field(default="about:blank")
        title: str
        status: int
        detail: str | None = None
        instance: str | None = None
        extra: dict[str, Any] = Field(default_factory=dict)

        def to_response(self) -> dict[str, Any]:
            data = self.model_dump()
            payload = {key: value for key, value in data.items() if value is not None}
            if payload.get("extra") == {}:
                payload.pop("extra", None)
            return payload

else:  # pragma: no cover - optional dependency fallback

    @dataclass(slots=True)
    class ProblemDetail:
        """Lightweight problem details implementation without pydantic."""

        title: str
        status: int
        type: str = "about:blank"
        detail: str | None = None
        instance: str | None = None
        extra: dict[str, Any] = field(default_factory=dict)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

        def to_response(self) -> dict[str, Any]:
            data = self.model_dump()
            payload = {key: value for key, value in data.items() if value is not None}
            if payload.get("extra") == {}:
                payload.pop("extra", None)
            return payload


class FoundationError(RuntimeError):
    """Base exception for foundation utilities."""

    def __init__(self, message: str, *, status: int = 500, detail: str | None = None) -> None:
        super().__init__(message)
        self.problem = ProblemDetail(title=message, status=status, detail=detail)
