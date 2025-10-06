"""GraphQL context helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..services import GatewayService
from .loaders import GraphQLLoaders


@dataclass(slots=True)
class GraphQLContext:
    service: GatewayService
    loaders: GraphQLLoaders


async def build_context(service: GatewayService) -> GraphQLContext:
    return GraphQLContext(service=service, loaders=GraphQLLoaders(service))
