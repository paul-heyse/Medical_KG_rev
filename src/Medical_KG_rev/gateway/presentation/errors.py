"""Error payload helpers for presentation formatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class ErrorDetail:
    """Structured error metadata for JSON:API responses."""

    status: int
    code: str
    title: str
    detail: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        payload = {
            "status": str(self.status),
            "code": self.code,
            "title": self.title,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload


__all__ = ["ErrorDetail"]
