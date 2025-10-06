from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from Medical_KG_rev.kg import GRAPH_SCHEMA, CypherTemplates, Neo4jClient, ShaclValidator


@dataclass
class _RunCall:
    query: str
    parameters: Dict[str, Any]


class _FakeTransaction:
    def __init__(self, calls: List[_RunCall]):
        self._calls = calls

    def run(self, query: str, parameters: Dict[str, Any]):
        self._calls.append(_RunCall(query=query, parameters=parameters))
        return self

    def data(self):
        return [{"query": self._calls[-1].query}]


class _FakeSession:
    def __init__(self, calls: List[_RunCall]):
        self._calls = calls

    def execute_write(self, func):
        tx = _FakeTransaction(self._calls)
        return func(tx)

    def close(self):  # pragma: no cover - included for interface completeness
        return None


class _FakeDriver:
    def __init__(self):
        self.calls: List[_RunCall] = []

    def session(self):
        return _FakeSession(self.calls)


def test_merge_node_enforces_validation():
    client = Neo4jClient(
        driver=_FakeDriver(),
        templates=CypherTemplates(GRAPH_SCHEMA),
        validator=ShaclValidator.from_schema(GRAPH_SCHEMA),
    )

    client.merge_node(
        "Document",
        {
            "document_id": "doc-1",
            "title": "A",
            "ingested_at": "2024-01-01T00:00:00Z",
        },
    )

    assert client.driver.calls[0].query.startswith("MERGE (n:Document")
    assert "n.title = $props.title" in client.driver.calls[0].query


def test_merge_node_missing_property_raises():
    client = Neo4jClient(
        driver=_FakeDriver(),
        templates=CypherTemplates(GRAPH_SCHEMA),
        validator=ShaclValidator.from_schema(GRAPH_SCHEMA),
    )

    try:
        client.merge_node("Entity", {"entity_id": "e-1"})
    except Exception as exc:  # noqa: BLE001
        assert "Missing required properties" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Validation should fail")


def test_link_generates_merge_statement():
    client = Neo4jClient(driver=_FakeDriver())

    client.link("Document", "Entity", "MENTIONS", "doc-1", "e-1", {"sentence_index": 1})

    params = client.driver.calls[0].parameters
    assert params["start"] == "doc-1"
    assert params["end"] == "e-1"
    assert "MERGE (a)-[r:MENTIONS]->(b)" in client.driver.calls[0].query
