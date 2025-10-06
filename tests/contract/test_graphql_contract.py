from __future__ import annotations

import pytest

pytest.importorskip("strawberry")

from Medical_KG_rev.gateway.graphql.context import GraphQLContext
from Medical_KG_rev.gateway.graphql.loaders import GraphQLLoaders
from Medical_KG_rev.gateway.graphql.schema import schema
from Medical_KG_rev.gateway.services import get_gateway_service


def test_schema_exports_sdl(tmp_path) -> None:
    sdl_path = tmp_path / "schema.graphql"
    sdl_path.write_text(schema.as_str())
    assert "type Query" in sdl_path.read_text()


@pytest.mark.anyio("asyncio")
async def test_query_document() -> None:
    service = get_gateway_service()
    query = """
    query Example($id: ID!) {
      document(id: $id) { id title }
    }
    """
    context = GraphQLContext(service=service, loaders=GraphQLLoaders(service))
    result = await schema.execute(query, variable_values={"id": "doc-1"}, context_value=context)
    assert result.errors is None
    assert result.data["document"]["id"] == "doc-1"
