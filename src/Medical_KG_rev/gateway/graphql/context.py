"""GraphQL context helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from fastapi import Request
from strawberry.fastapi.context import BaseContext

from ..services import GatewayService
from .loaders import GraphQLLoaders


@dataclass(slots=True)
class GraphQLContext(BaseContext):
    service: GatewayService
    loaders: GraphQLLoaders
    tenant_id: str
    scopes: set[str]

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - strawberry compatibility
        if key == "tenant_id":
            return self.tenant_id
        if key == "scopes":
            return self.scopes
        raise KeyError(key)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


async def build_context(service: GatewayService, request: Request) -> GraphQLContext:
    security_context = getattr(request.state, "security_context", None)
    validated = getattr(request.state, "validated_tenant_id", None)
    tenant_id = validated or getattr(security_context, "tenant_id", "")
    raw_scopes: Iterable[str] = getattr(security_context, "scopes", set()) or set()
    scopes = {scope for scope in raw_scopes}
    return GraphQLContext(
        service=service,
        loaders=GraphQLLoaders(service),
        tenant_id=str(tenant_id),
        scopes=scopes,
    )
