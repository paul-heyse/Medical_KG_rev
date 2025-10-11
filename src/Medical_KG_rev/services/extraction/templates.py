"""Structured extraction templates for domain specific results.

Key Responsibilities:
    - Define Pydantic models for structured entity extraction
    - Provide validation for extracted spans and entities
    - Support domain-specific extraction templates (medical, legal, etc.)
    - Enable type-safe extraction result processing

Collaborators:
    - Upstream: Extraction services use these templates for result validation
    - Downstream: Application layers consume validated extraction results

Side Effects:
    - None: Pure data models with validation

Thread Safety:
    - Thread-safe: Pydantic models are immutable

Performance Characteristics:
    - O(1) model instantiation and validation
    - O(n) text verification where n is span length
    - Efficient JSON serialization with Pydantic

Example:
    >>> span = Span(text="patient", start=10, end=17)
    >>> span.verify("The patient was admitted")
    >>> # Validates that the span matches the source text
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, Field, ValidationError, model_validator



class TemplateValidationError(ValueError):
    """Raised when extracted data violates template constraints.

    This exception is raised when validation fails during template-based
    extraction result processing.
    """


class Span(BaseModel):
    """Text span with position information for entity extraction.

    Represents a contiguous segment of text extracted from a document,
    with validation to ensure the span boundaries are consistent.

    Attributes:
        text: The actual text content of the span.
        start: Starting character position in the source text.
        end: Ending character position in the source text (exclusive).
    """
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(cls, values: Span) -> Span:  # type: ignore[override]
        """Validate that span end is greater than start.

        Args:
            values: The span instance to validate.

        Returns:
            The validated span instance.

        Raises:
            ValueError: If span end is not greater than start.
        """
        if values.end <= values.start:
            raise ValueError("Span end must be greater than start")
        return values

    def verify(self, source_text: str) -> None:
        """Verify that the span matches the source text at the specified position.

        Args:
            source_text: The original text from which the span was extracted.

        Raises:
            TemplateValidationError: If the span doesn't match the source text.
        """
        snippet = source_text[self.start : self.end]
        if snippet == self.text:
            return
        tolerance = 8
        window_start = max(self.start - tolerance, 0)
        window_end = min(len(source_text), self.end + tolerance)
        window = source_text[window_start:window_end]
        index = window.find(self.text)
        if index == -1:
            raise TemplateValidationError(
                f"Span text mismatch: expected '{self.text}' near index {self.start}"
            )
        actual_start = window_start + index
        actual_end = actual_start + len(self.text)
        if actual_start < self.start - tolerance or actual_end > self.end + tolerance:
            raise TemplateValidationError(
                f"Span indices [{self.start}, {self.end}) do not align with text '{self.text}'"
            )


class Population(BaseModel):
    description: str
    age_range: str | None = None
    gender: str | None = None
    condition: str | None = None
    sample_size: int | None = Field(default=None, ge=0)
    span: Span


class Intervention(BaseModel):
    name: str
    type: str | None = None
    route: str | None = None
    dose: str | None = None
    span: Span


class Comparison(BaseModel):
    description: str
    span: Span


class Outcome(BaseModel):
    name: str
    measurement: str | None = None
    timepoint: str | None = None
    effect_size: float | None = None
    span: Span


class PICOExtraction(BaseModel):
    population: Population
    interventions: Sequence[Intervention]
    comparison: Comparison | None = None
    outcomes: Sequence[Outcome]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EffectMeasure(BaseModel):
    outcome: str
    effect_size: float
    unit: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    span: Span

    @model_validator(mode="after")
    def validate_interval(cls, values: EffectMeasure) -> EffectMeasure:  # type: ignore[override]
        if values.ci_low is not None and values.ci_high is not None:
            if values.ci_low > values.ci_high:
                raise ValueError("Confidence interval lower bound must be <= upper bound")
        return values


class EffectsExtraction(BaseModel):
    measures: Sequence[EffectMeasure]
    model: str | None = None


class AdverseEvent(BaseModel):
    event_type: str
    severity: str = Field(pattern="^(mild|moderate|severe|life-threatening)$")
    frequency: str
    causality: str = Field(pattern="^(definite|probable|possible|unlikely|unrelated)$")
    span: Span


class AdverseEventsExtraction(BaseModel):
    events: Sequence[AdverseEvent]


class DoseRegimen(BaseModel):
    drug: str
    dose_value: float
    dose_unit: str
    route: str
    frequency: str
    duration: str | None = None
    span: Span


class DoseExtraction(BaseModel):
    regimens: Sequence[DoseRegimen]


class CriteriaItem(BaseModel):
    text: str
    span: Span
    metadata: dict[str, str] = Field(default_factory=dict)


class EligibilityExtraction(BaseModel):
    inclusion: Sequence[CriteriaItem]
    exclusion: Sequence[CriteriaItem]


_TEMPLATE_REGISTRY: dict[str, type[BaseModel]] = {
    "pico": PICOExtraction,
    "effects": EffectsExtraction,
    "ae": AdverseEventsExtraction,
    "dose": DoseExtraction,
    "eligibility": EligibilityExtraction,
}


def validate_template(
    kind: str, payload: Mapping[str, object], source_text: str
) -> Mapping[str, object]:
    try:
        model = _TEMPLATE_REGISTRY[kind]
    except KeyError as exc:
        raise TemplateValidationError(f"Unknown extraction kind '{kind}'") from exc
    try:
        parsed = model.model_validate(payload)
    except ValidationError as exc:
        raise TemplateValidationError(str(exc)) from exc
    _validate_spans(parsed, source_text)
    return parsed.model_dump(mode="json")


def _validate_spans(instance: BaseModel, source_text: str) -> None:
    for span in _iter_spans(instance):
        span.verify(source_text)


def _iter_spans(instance: BaseModel | Sequence | Mapping) -> Iterable[Span]:
    if isinstance(instance, Span):
        yield instance
        return
    if isinstance(instance, BaseModel):
        for key, value in instance.__dict__.items():
            if key.startswith("_"):
                continue
            yield from _iter_spans(value)
    elif isinstance(instance, Mapping):
        for value in instance.values():
            yield from _iter_spans(value)
    elif isinstance(instance, Sequence) and not isinstance(instance, (str, bytes)):
        for item in instance:
            yield from _iter_spans(item)


__all__ = [
    "AdverseEvent",
    "AdverseEventsExtraction",
    "Comparison",
    "CriteriaItem",
    "DoseExtraction",
    "DoseRegimen",
    "EffectMeasure",
    "EffectsExtraction",
    "EligibilityExtraction",
    "Intervention",
    "Outcome",
    "PICOExtraction",
    "Population",
    "Span",
    "TemplateValidationError",
    "validate_template",
]
