from fastapi.testclient import TestClient

from Medical_KG_rev.gateway.app import create_app


def test_graphql_adapters_query(api_key: str) -> None:
    client = TestClient(create_app())
    query = """
    query {
      adapters {
        name
        domain
      }
    }
    """
    response = client.post(
        "/graphql",
        json={"query": query},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "data" in payload
    assert payload["data"]["adapters"]


def test_graphql_adapter_health(api_key: str) -> None:
    client = TestClient(create_app())
    query = """
    query($name: String!) {
      adapter(name: $name) {
        name
        summary
      }
      adapterHealth(name: $name) {
        healthy
      }
    }
    """
    adapters = client.post(
        "/graphql",
        json={"query": "{ adapters { name } }"},
        headers={"X-API-Key": api_key},
    ).json()
    name = adapters["data"]["adapters"][0]["name"]
    response = client.post(
        "/graphql",
        json={"query": query, "variables": {"name": name}},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["adapter"]["name"] == name
    assert body["data"]["adapterHealth"]["healthy"] is True
