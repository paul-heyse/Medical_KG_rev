"""Error utilities implementing RFC 7807 problem details."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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


class FoundationError(RuntimeError):
    """Base exception for foundation utilities."""

    def __init__(self, message: str, *, status: int = 500, detail: str | None = None) -> None:
        super().__init__(message)
        self.problem = ProblemDetail(title=message, status=status, detail=detail)
