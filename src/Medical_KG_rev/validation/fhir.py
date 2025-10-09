"""FHIR resource validation using JSON Schemas.

This module provides FHIR resource validation using JSON Schema validation
with terminology checks for coding systems and code values.

The module supports:
- JSON Schema validation for FHIR resources
- Terminology validation for coding systems
- Multiple resource types (Evidence, ResearchStudy, MedicationStatement)
- Custom schema support

Thread Safety:
    Thread-safe: Validator instances are stateless.

Performance:
    Schema compilation happens once during initialization.
    Validation performance depends on resource complexity.

Example:
    >>> validator = FHIRValidator()
    >>> resource = {"resourceType": "Evidence", "status": "active"}
    >>> validator.validate(resource)
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

from jsonschema import Draft202012Validator

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class _CompiledSchema:
    """Internal data model for compiled schema information.

    Attributes:
        validator: Compiled JSON Schema validator.
        resource_type: FHIR resource type name.
    """

    validator: Draft202012Validator
    resource_type: str


# FHIR JSON Schema definitions for resource validation
_CODING_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["system", "code"],
    "properties": {
        "system": {"type": "string"},
        "code": {"type": "string"},
        "display": {"type": "string"},
    },
}

_CODEABLE_CONCEPT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "coding": {
            "type": "array",
            "minItems": 1,
            "items": _CODING_SCHEMA,
        },
        "text": {"type": "string"},
    },
}

_CHARACTERISTIC_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["code"],
    "properties": {
        "code": _CODEABLE_CONCEPT_SCHEMA,
        "valueCodeableConcept": _CODEABLE_CONCEPT_SCHEMA,
    },
}

FHIR_SCHEMAS: dict[str, dict[str, object]] = {
    "Evidence": {
        "$id": "https://example.org/fhir/Evidence",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["resourceType", "status", "description", "outcome"],
        "properties": {
            "resourceType": {"const": "Evidence"},
            "status": {"type": "string"},
            "description": {"type": "string"},
            "outcome": {
                "type": "object",
                "required": ["reference"],
                "properties": {"reference": {"type": "string"}},
            },
            "characteristic": {
                "type": "array",
                "items": {"$ref": "#/definitions/Characteristic"},
            },
        },
        "definitions": {
            "Characteristic": _CHARACTERISTIC_SCHEMA,
            "CodeableConcept": _CODEABLE_CONCEPT_SCHEMA,
            "Coding": _CODING_SCHEMA,
        },
    },
    "ResearchStudy": {
        "$id": "https://example.org/fhir/ResearchStudy",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["resourceType", "status", "title", "identifier"],
        "properties": {
            "resourceType": {"const": "ResearchStudy"},
            "status": {"type": "string"},
            "title": {"type": "string"},
            "identifier": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["system", "value"],
                    "properties": {
                        "system": {"type": "string"},
                        "value": {"type": "string"},
                    },
                },
            },
            "phase": {"$ref": "#/definitions/CodeableConcept"},
            "category": {
                "type": "array",
                "items": {"$ref": "#/definitions/CodeableConcept"},
            },
        },
        "definitions": {
            "CodeableConcept": _CODEABLE_CONCEPT_SCHEMA,
            "Coding": _CODING_SCHEMA,
        },
    },
    "MedicationStatement": {
        "$id": "https://example.org/fhir/MedicationStatement",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["resourceType", "status", "medication", "subject"],
        "properties": {
            "resourceType": {"const": "MedicationStatement"},
            "status": {"type": "string"},
            "medication": {
                "oneOf": [
                    {"$ref": "#/definitions/CodeableConcept"},
                    {
                        "type": "object",
                        "required": ["reference"],
                        "properties": {"reference": {"type": "string"}},
                    },
                ]
            },
            "subject": {
                "type": "object",
                "required": ["reference"],
                "properties": {"reference": {"type": "string"}},
            },
            "dosage": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "doseAndRate": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "doseQuantity": {
                                        "type": "object",
                                        "required": ["value", "unit"],
                                        "properties": {
                                            "value": {"type": "number"},
                                            "unit": {"type": "string"},
                                        },
                                    }
                                },
                            },
                        },
                    },
                },
            },
        },
        "definitions": {
            "CodeableConcept": _CODEABLE_CONCEPT_SCHEMA,
            "Coding": _CODING_SCHEMA,
        },
    },
}


# ==============================================================================
# VALIDATOR IMPLEMENTATION
# ==============================================================================

class FHIRValidationError(ValueError):
    """Raised when a FHIR resource fails schema or terminology validation."""

    def __init__(self, errors: Sequence[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = list(errors)


class FHIRValidator:
    """Validate FHIR resources against curated schemas.

    Validates FHIR resources using JSON Schema validation and terminology
    checks for coding systems and code values.
    """

    def __init__(self, *, schemas: Mapping[str, Mapping[str, object]] | None = None) -> None:
        """Initialize validator with schemas.

        Args:
            schemas: Optional custom schemas to use instead of defaults.
        """
        source = schemas or FHIR_SCHEMAS
        self._validators: MutableMapping[str, _CompiledSchema] = {}
        for resource_type, schema in source.items():
            validator = Draft202012Validator(schema)
            self._validators[resource_type] = _CompiledSchema(
                validator=validator, resource_type=resource_type
            )

    def validate(self, resource: Mapping[str, object]) -> None:
        """Validate a FHIR resource against its schema.

        Args:
            resource: FHIR resource to validate.

        Raises:
            FHIRValidationError: If validation fails.
        """
        resource_type = resource.get("resourceType")
        if not resource_type:
            raise FHIRValidationError(["Missing resourceType"])
        compiled = self._validators.get(str(resource_type))
        if not compiled:
            raise FHIRValidationError([f"Unsupported resource type: {resource_type}"])
        errors = self._validate_schema(compiled.validator, resource)
        errors.extend(self._validate_terminology(resource))
        if errors:
            raise FHIRValidationError(errors)

    def validate_evidence(self, resource: Mapping[str, object]) -> None:
        """Validate an Evidence resource.

        Args:
            resource: Evidence resource to validate.

        Raises:
            FHIRValidationError: If validation fails.
        """
        self.validate(resource)

    def validate_research_study(self, resource: Mapping[str, object]) -> None:
        """Validate a ResearchStudy resource.

        Args:
            resource: ResearchStudy resource to validate.

        Raises:
            FHIRValidationError: If validation fails.
        """
        self.validate(resource)

    def validate_medication_statement(self, resource: Mapping[str, object]) -> None:
        """Validate a MedicationStatement resource.

        Args:
            resource: MedicationStatement resource to validate.

        Raises:
            FHIRValidationError: If validation fails.
        """
        self.validate(resource)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_schema(
        self, validator: Draft202012Validator, resource: Mapping[str, object]
    ) -> list[str]:
        errors: list[str] = []
        for error in validator.iter_errors(resource):
            path = ".".join(str(part) for part in error.path)
            message = f"{path or 'root'}: {error.message}"
            errors.append(message)
        return errors

    def _validate_terminology(self, resource: Mapping[str, object]) -> list[str]:
        errors: list[str] = []
        for coding in self._iter_coding(resource):
            system = coding.get("system")
            code = coding.get("code")
            if not system or not isinstance(system, str):
                errors.append("Coding.system must be a non-empty string")
            if not code or not isinstance(code, str):
                errors.append("Coding.code must be a non-empty string")
        return errors

    def _iter_coding(self, resource: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
        stack: list[object] = [resource]
        while stack:
            current = stack.pop()
            if isinstance(current, Mapping):
                if "coding" in current and isinstance(current["coding"], Sequence):
                    for entry in current["coding"]:
                        if isinstance(entry, Mapping):
                            yield entry
                stack.extend(current.values())
            elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
                stack.extend(current)


# ==============================================================================
# ERROR HANDLING
# ==============================================================================


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = ["FHIR_SCHEMAS", "FHIRValidationError", "FHIRValidator"]
