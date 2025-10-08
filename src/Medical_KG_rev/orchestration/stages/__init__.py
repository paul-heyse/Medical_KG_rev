"""Common orchestration stage utilities."""

from __future__ import annotations

from typing import Any

from Medical_KG_rev.utils.errors import ProblemDetail


class StageFailure(RuntimeError):
    """Wrap a stage failure with retry metadata and RFC 7807 details."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 500,
        detail: str | None = None,
        stage: str | None = None,
        error_type: str | None = None,
        retriable: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.retriable = retriable
        self.error_type = error_type or ("transient" if retriable else "permanent")
        self.problem = ProblemDetail(
            title=message,
            status=status,
            detail=detail,
            extra=extra or {},
        )


__all__ = ["StageFailure"]
