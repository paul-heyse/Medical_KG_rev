"""Thin wrapper around the Neo4j Python driver."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping

from .cypher_templates import CypherTemplates
from .schema import GRAPH_SCHEMA
from .shacl import ShaclValidator


@dataclass(slots=True)
class Neo4jClient:
    """Provides convenience helpers for common graph operations."""

    driver: Any
    templates: CypherTemplates = field(default_factory=lambda: CypherTemplates(GRAPH_SCHEMA))
    validator: ShaclValidator = field(default_factory=lambda: ShaclValidator.from_schema(GRAPH_SCHEMA))

    @contextmanager
    def _session(self) -> Iterator[Any]:
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def write(self, query: str, parameters: Mapping[str, object] | None = None) -> Iterable[Mapping[str, Any]]:
        with self._session() as session:
            return session.execute_write(lambda tx: tx.run(query, parameters or {}).data())

    def merge_node(self, label: str, properties: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        self.validator.validate_node(label, properties)
        query, parameters = self.templates.merge_node(label, properties)
        return self.write(query, parameters)

    def link(
        self,
        start_label: str,
        end_label: str,
        rel_type: str,
        start_key: Any,
        end_key: Any,
        properties: Mapping[str, Any] | None = None,
    ) -> Iterable[Mapping[str, Any]]:
        query, parameters = self.templates.link_nodes(
            start_label,
            end_label,
            rel_type,
            start_key,
            end_key,
            properties,
        )
        return self.write(query, parameters)

    def with_transaction(self, func: Callable[[Any], Any]) -> Any:
        with self._session() as session:
            return session.execute_write(func)
