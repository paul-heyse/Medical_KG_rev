"""Custom exceptions for the vector store subsystem."""

from __future__ import annotations

from typing import Any

from Medical_KG_rev.utils.errors import FoundationError


class VectorStoreError(FoundationError):
    """Base class for vector store errors with RFC 7807 payloads."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        detail: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status=status, detail=detail)
        if extra:
            self.problem.extra = extra


class NamespaceNotFoundError(VectorStoreError):
    def __init__(self, namespace: str, *, tenant_id: str) -> None:
        super().__init__(
            "Vector namespace not found",
            status=404,
            detail=f"Namespace '{namespace}' is not registered for tenant '{tenant_id}'.",
            extra={"namespace": namespace, "tenant_id": tenant_id},
        )


class DimensionMismatchError(VectorStoreError):
    def __init__(self, expected: int, actual: int, *, namespace: str) -> None:
        super().__init__(
            "Vector dimension mismatch",
            status=422,
            detail=f"Expected dimension {expected}, received {actual}.",
            extra={"expected": expected, "actual": actual, "namespace": namespace},
        )


class ResourceExhaustedError(VectorStoreError):
    def __init__(self, namespace: str, *, detail: str | None = None) -> None:
        super().__init__(
            "Vector store capacity exceeded",
            status=507,
            detail=detail or "The vector backend cannot accept more data.",
            extra={"namespace": namespace},
        )


class BackendUnavailableError(VectorStoreError):
    def __init__(
        self, message: str = "Vector backend unavailable", *, retry_after: float | None = None
    ) -> None:
        super().__init__(
            message,
            status=503,
            detail="Vector store backend is temporarily unavailable.",
            extra={"retry_after": retry_after} if retry_after else None,
        )


class ScopeError(VectorStoreError):
    def __init__(self, *, required_scope: str) -> None:
        super().__init__(
            "Missing required scope",
            status=403,
            detail=f"The caller lacks the '{required_scope}' scope.",
            extra={"required_scope": required_scope},
        )


class InvalidNamespaceConfigError(VectorStoreError):
    def __init__(self, namespace: str, *, detail: str) -> None:
        super().__init__(
            "Invalid namespace configuration",
            status=422,
            detail=detail,
            extra={"namespace": namespace},
        )
