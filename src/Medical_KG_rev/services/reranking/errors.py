"""Domain specific exceptions for the reranking system."""

from __future__ import annotations

from typing import Any

from Medical_KG_rev.utils.errors import ProblemDetail


class RerankingError(RuntimeError):
    """Base exception translating to an RFC 7807 problem detail."""

    def __init__(
        self,
        title: str,
        *,
        status: int,
        detail: str | None = None,
        type: str = "https://docs.medical-kg/reranking/errors",
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(title)
        self.title = title
        self.status = status
        self.detail = detail
        self.type = type
        self.extra = extra or {}

    def to_problem(self) -> ProblemDetail:
        return ProblemDetail(
            title=self.title,
            status=self.status,
            detail=self.detail,
            type=self.type,
            extra=dict(self.extra),
        )


class InvalidPairFormatError(RerankingError):
    def __init__(self, detail: str) -> None:
        super().__init__(
            title="Invalid query/document pair", status=400, detail=detail
        )


class UnknownRerankerError(RerankingError):
    def __init__(self, reranker_id: str, available: list[str]) -> None:
        super().__init__(
            title="Reranker not found",
            status=422,
            detail=f"Reranker '{reranker_id}' is not registered",
            extra={"available": available},
        )


class GPUUnavailableError(RerankingError):
    def __init__(self, reranker_id: str) -> None:
        super().__init__(
            title="GPU unavailable",
            status=503,
            detail=f"Reranker '{reranker_id}' requires GPU acceleration",
            type="https://docs.medical-kg/reranking/errors/gpu-unavailable",
        )


class CircuitBreakerOpenError(RerankingError):
    def __init__(self, reranker_id: str) -> None:
        super().__init__(
            title="Reranker temporarily unavailable",
            status=503,
            detail=f"Circuit breaker open for reranker '{reranker_id}'",
            type="https://docs.medical-kg/reranking/errors/circuit-open",
        )
