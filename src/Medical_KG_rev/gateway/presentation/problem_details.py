"""Helpers for constructing Docling-specific problem detail payloads."""

from __future__ import annotations

from typing import Any

from ..models import DoclingErrorCode, ProblemDetail


def build_docling_problem(
    *,
    title: str,
    status: int,
    detail: str,
    error_code: DoclingErrorCode,
    type_override: str | None = None,
    extensions: dict[str, Any] | None = None,
) -> ProblemDetail:
    """Create a ProblemDetail instance tailored for Docling VLM errors."""

    payload = {
        "title": title,
        "status": status,
        "type": type_override or f"https://medical-kg/errors/{error_code.value}",
        "detail": detail,
        "extensions": dict(extensions or {}),
    }
    payload["extensions"].setdefault("error_code", error_code.value)
    return ProblemDetail.model_validate(payload)


__all__ = ["build_docling_problem"]
