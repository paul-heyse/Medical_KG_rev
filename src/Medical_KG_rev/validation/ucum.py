"""UCUM unit validation utilities using pint."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

from pint import UnitRegistry
from pint.errors import PintError


class UnitValidationError(ValueError):
    """Raised when a measurement cannot be validated against UCUM rules."""


@dataclass(frozen=True)
class UnitValidationResult:
    """Result of validating and normalising a measurement."""

    original_value: float
    original_unit: str
    normalized_value: float
    normalized_unit: str
    context: str


_DefaultContexts: dict[str, dict[str, object]] = {
    "dose": {
        "canonical": "mg",
        "allowed": {"mg", "g", "mcg", "mg/kg"},
        "range": (0.0, 5000.0),
    },
    "lab": {
        "canonical": "mg/dL",
        "allowed": {"mg/dL", "mmol/L", "ug/mL"},
        "range": (0.0, 1000.0),
    },
    "vitals": {
        "canonical": "mmHg",
        "allowed": {"mmHg", "kPa"},
        "range": (0.0, 400.0),
    },
}


class UCUMValidator:
    """Validate medical measurements against UCUM units and ranges."""

    def __init__(
        self,
        *,
        registry: UnitRegistry | None = None,
        contexts: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self._ureg = registry or UnitRegistry()
        self._contexts: dict[str, dict[str, object]] = {
            name: {
                "canonical": (
                    self._canonicalise_unit(str(ctx["canonical"])) if "canonical" in ctx else None
                ),
                "allowed": {
                    self._canonicalise_unit(str(unit)) for unit in ctx.get("allowed", set())
                },
                "range": ctx.get("range", (None, None)),
            }
            for name, ctx in (contexts or _DefaultContexts).items()
        }
        self._normalization_cache: MutableMapping[tuple[str, str], tuple[float, str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate_measurement(self, measurement: str, *, context: str) -> UnitValidationResult:
        """Validate a textual measurement such as "20 mg/dL"."""

        measurement = measurement.strip()
        if not measurement:
            raise UnitValidationError("Measurement is empty")
        parts = measurement.split()
        if len(parts) == 1:
            raise UnitValidationError("Measurement must include both value and unit")
        value_str = parts[0]
        unit_str = " ".join(parts[1:])
        value = self._coerce_value(value_str)
        return self.validate_value(value, unit_str, context=context)

    def validate_value(
        self,
        value: float,
        unit: str,
        *,
        context: str,
    ) -> UnitValidationResult:
        """Validate and normalise a numeric value with unit for a context."""

        if unit is None or str(unit).strip() == "":
            raise UnitValidationError("Unit is required for medical measurements")
        ctx = self._get_context(context)
        normalized_value, normalized_unit = self._normalize(value, unit, ctx)
        lower, upper = ctx.get("range", (None, None))
        if lower is not None and normalized_value < float(lower):
            raise UnitValidationError(
                f"Value {normalized_value} {normalized_unit} is below minimum of {lower}"
            )
        if upper is not None and normalized_value > float(upper):
            raise UnitValidationError(
                f"Value {normalized_value} {normalized_unit} exceeds maximum of {upper}"
            )
        return UnitValidationResult(
            original_value=value,
            original_unit=self._format_unit(unit),
            normalized_value=normalized_value,
            normalized_unit=normalized_unit,
            context=context,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_context(self, name: str) -> dict[str, object]:
        try:
            return self._contexts[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise UnitValidationError(f"Unknown validation context: {name}") from exc

    def _format_unit(self, unit: str) -> str:
        return unit.strip()

    def _coerce_value(self, value: str) -> float:
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise UnitValidationError(f"Invalid numeric value '{value}'") from exc

    def _normalize(
        self,
        value: float,
        unit: str,
        ctx: Mapping[str, object],
    ) -> tuple[float, str]:
        formatted_unit = self._canonicalise_unit(unit)
        canonical_unit = str(ctx["canonical"])  # type: ignore[index]
        cache_key = (formatted_unit, canonical_unit)
        allowed_units = ctx["allowed"]  # type: ignore[index]
        if formatted_unit not in allowed_units:
            raise UnitValidationError(f"Unit '{formatted_unit}' is not permitted in context")
        cached = self._normalization_cache.get(cache_key)
        if cached is None:
            try:
                quantity = self._ureg.Quantity(1, formatted_unit)
                normalised = quantity.to(canonical_unit)
            except PintError as exc:
                raise UnitValidationError(f"Failed to normalise unit '{formatted_unit}'") from exc
            cached = (float(normalised.magnitude), canonical_unit)
            self._normalization_cache[cache_key] = cached
        factor, normalised_unit = cached
        normalized_value = value * factor
        return float(normalized_value), normalised_unit

    def _canonicalise_unit(self, unit: str) -> str:
        candidate = unit.strip()
        lowered = candidate.lower()
        lowered = lowered.replace("per", "/")
        lowered = lowered.replace(" ", "")
        replacements = {
            "milligrams": "mg",
            "milligram": "mg",
            "micrograms": "mcg",
            "microgram": "mcg",
            "grams": "g",
            "gram": "g",
            "kilograms": "kg",
            "kilogram": "kg",
            "deciliter": "dL",
            "litre": "L",
            "liter": "L",
            "milliliter": "mL",
            "millilitre": "mL",
        }
        for source, target in replacements.items():
            lowered = lowered.replace(source, target)
        # Normalise case for litre-based units
        lowered = (
            lowered.replace("dl", "dL")
            .replace("ml", "mL")
            .replace("/l", "/L")
            .replace("mmhg", "mmHg")
            .replace("kpa", "kPa")
        )
        return lowered


__all__ = ["UCUMValidator", "UnitValidationError", "UnitValidationResult"]
