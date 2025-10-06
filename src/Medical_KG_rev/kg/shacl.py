"""Lightweight SHACL-like validation for graph entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class ValidationError(ValueError):
    """Raised when data violates the configured shapes."""


@dataclass(slots=True)
class ShaclValidator:
    required_properties: Mapping[str, set[str]]

    @classmethod
    def from_schema(cls, schema: Mapping[str, object]) -> "ShaclValidator":
        requirements: dict[str, set[str]] = {}
        for label, node_schema in schema.items():
            required = set()
            required.add(node_schema.key)  # type: ignore[attr-defined]
            for prop, requirement in getattr(node_schema, "properties", {}).items():
                if requirement == "required":
                    required.add(prop)
            requirements[label] = required
        return cls(requirements)

    def validate_node(self, label: str, properties: Mapping[str, object]) -> None:
        try:
            required = self.required_properties[label]
        except KeyError as exc:
            raise ValidationError(f"Unknown label: {label}") from exc
        missing = [name for name in required if name not in properties or properties[name] in (None, "")]
        if missing:
            raise ValidationError(f"Missing required properties for {label}: {', '.join(sorted(missing))}")
