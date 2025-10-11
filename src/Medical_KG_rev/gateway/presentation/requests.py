"""Request normalisation helpers for the presentation layer."""

from __future__ import annotations

from typing import TypeVar, cast

from fastapi import Request
from pydantic import BaseModel

from ...auth import SecurityContext


TModel = TypeVar("TModel", bound=BaseModel)


def apply_tenant_context(
    request_model: TModel,
    security: SecurityContext,
    http_request: Request | None = None,
) -> TModel:
    """Ensure request payloads inherit the authenticated tenant context."""
    tenant_id = getattr(request_model, "tenant_id", None)
    if tenant_id and tenant_id != security.tenant_id:
        raise PermissionError("Tenant mismatch")
    updated = cast(TModel, request_model.model_copy(update={"tenant_id": security.tenant_id}))
    if http_request is not None:
        http_request.state.requested_tenant_id = getattr(updated, "tenant_id", security.tenant_id)
    return updated


__all__ = ["apply_tenant_context"]
