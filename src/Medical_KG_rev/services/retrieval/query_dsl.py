"""Simple query DSL parser for retrieval filters and facets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass



class QueryValidationError(ValueError):
    """Raised when a query fails validation."""


@dataclass(slots=True)
class QueryDSL:
    allowed_filters: Mapping[str, set[str]]

    def parse(self, payload: Mapping[str, object]) -> dict[str, object]:
        filters = payload.get("filters", {})
        facets = payload.get("facets", [])
        if not isinstance(filters, Mapping):
            raise QueryValidationError("filters must be a mapping")
        if not isinstance(facets, list):
            raise QueryValidationError("facets must be a list")
        validated_filters = self._validate_filters(filters)
        validated_facets = self._validate_facets(facets)
        return {"filters": validated_filters, "facets": validated_facets}

    def _validate_filters(self, filters: Mapping[str, object]) -> dict[str, object]:
        validated: dict[str, object] = {}
        for key, value in filters.items():
            if key not in self.allowed_filters:
                raise QueryValidationError(f"Unknown filter: {key}")
            allowed_values = self.allowed_filters[key]
            if allowed_values and str(value) not in allowed_values:
                raise QueryValidationError(f"Invalid value for {key}: {value}")
            validated[key] = value
        return validated

    def _validate_facets(self, facets: list[object]) -> list[str]:
        valid: list[str] = []
        for facet in facets:
            if not isinstance(facet, str):
                raise QueryValidationError("Facet entries must be strings")
            if facet not in self.allowed_filters:
                raise QueryValidationError(f"Unknown facet: {facet}")
            valid.append(facet)
        return valid
