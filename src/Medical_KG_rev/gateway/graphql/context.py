"""GraphQL context helpers."""

from __future__ import annotations

from dataclasses import dataclass

from strawberry.fastapi.context import BaseContext

from ..services import GatewayService
from .loaders import GraphQLLoaders


@dataclass(slots=True)
class GraphQLContext(BaseContext):
    service: GatewayService
    loaders: GraphQLLoaders


async def build_context(service: GatewayService) -> GraphQLContext:
    return GraphQLContext(service=service, loaders=GraphQLLoaders(service))
