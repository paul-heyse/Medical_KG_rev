"""Heuristics for detecting clinical query intent and metadata alignment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence


class ClinicalIntent(str, Enum):
    """Canonical set of clinical retrieval intents."""

    ELIGIBILITY = "eligibility"
    ADVERSE_EVENTS = "adverse_events"
    RESULTS = "results"
    METHODS = "methods"
    DOSAGE = "dosage"
    INDICATIONS = "indications"


@dataclass(slots=True)
class ClinicalIntentScore:
    """Scored intent prediction returned by the analyzer."""

    intent: ClinicalIntent
    confidence: float
    keywords: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "keywords": list(self.keywords),
        }


@dataclass(slots=True)
class ClinicalIntentAnalysis:
    """Full analysis payload including matches and override context."""

    intents: tuple[ClinicalIntentScore, ...]
    override: tuple[ClinicalIntent, ...] = ()
    tokens: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "intents": [score.as_dict() for score in self.intents],
            "override": [intent.value for intent in self.override],
            "tokens": list(self.tokens),
        }

    @property
    def primary(self) -> ClinicalIntentScore | None:
        return self.intents[0] if self.intents else None


DEFAULT_INTENT_KEYWORDS: dict[ClinicalIntent, tuple[str, ...]] = {
    ClinicalIntent.ELIGIBILITY: (
        "eligibility",
        "inclusion criteria",
        "exclusion criteria",
        "participant requirements",
    ),
    ClinicalIntent.ADVERSE_EVENTS: (
        "adverse event",
        "side effect",
        "safety",
        "toxicity",
    ),
    ClinicalIntent.RESULTS: (
        "results",
        "efficacy",
        "outcome",
        "endpoint",
    ),
    ClinicalIntent.METHODS: (
        "methods",
        "study design",
        "randomized",
        "protocol",
    ),
    ClinicalIntent.DOSAGE: (
        "dose",
        "dosage",
        "administration",
        "titration",
    ),
    ClinicalIntent.INDICATIONS: (
        "indication",
        "treatment for",
        "approved for",
        "use in",
    ),
}


DEFAULT_SECTION_HINTS: dict[ClinicalIntent, tuple[str, ...]] = {
    ClinicalIntent.ELIGIBILITY: (
        "eligibility",
        "inclusion",
        "exclusion",
        "participant",
    ),
    ClinicalIntent.ADVERSE_EVENTS: (
        "adverse",
        "safety",
        "side effect",
        "ae",
    ),
    ClinicalIntent.RESULTS: (
        "result",
        "outcome",
        "efficacy",
        "findings",
    ),
    ClinicalIntent.METHODS: (
        "method",
        "design",
        "protocol",
        "study conduct",
    ),
    ClinicalIntent.DOSAGE: (
        "dose",
        "dosage",
        "administration",
        "schedule",
    ),
    ClinicalIntent.INDICATIONS: (
        "indication",
        "introduction",
        "background",
        "disease",
    ),
}


DEFAULT_INTENT_BOOSTS: dict[ClinicalIntent, float] = {
    ClinicalIntent.ELIGIBILITY: 3.0,
    ClinicalIntent.ADVERSE_EVENTS: 2.5,
    ClinicalIntent.RESULTS: 2.0,
    ClinicalIntent.METHODS: 1.5,
    ClinicalIntent.DOSAGE: 2.0,
    ClinicalIntent.INDICATIONS: 1.5,
}


def _normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def _tokenise(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.lower().split() if token)


class ClinicalIntentAnalyzer:
    """Rule-based detector that extracts clinical intents from queries."""

    def __init__(
        self,
        *,
        taxonomy: Mapping[ClinicalIntent, Sequence[str]] | None = None,
        threshold: float = 0.6,
    ) -> None:
        self._taxonomy: dict[ClinicalIntent, tuple[str, ...]] = {
            intent: tuple(phrases)
            for intent, phrases in (taxonomy or DEFAULT_INTENT_KEYWORDS).items()
        }
        self.threshold = float(threshold)

    def analyse(
        self,
        query: str,
        *,
        override: Sequence[ClinicalIntent] | None = None,
    ) -> ClinicalIntentAnalysis:
        tokens = _tokenise(query)
        if override:
            overrides = tuple(dict.fromkeys(override))
            scores = tuple(
                ClinicalIntentScore(intent=item, confidence=1.0, keywords=())
                for item in overrides
            )
            return ClinicalIntentAnalysis(intents=scores, override=overrides, tokens=tokens)

        if not query.strip():
            return ClinicalIntentAnalysis(intents=(), tokens=tokens)

        normalised = _normalise_text(query)
        matches: dict[ClinicalIntent, set[str]] = {
            intent: set() for intent in self._taxonomy
        }
        total_matches = 0
        for intent, phrases in self._taxonomy.items():
            for phrase in phrases:
                if phrase in normalised:
                    matches[intent].add(phrase)
                    total_matches += 1
        if total_matches == 0:
            return ClinicalIntentAnalysis(intents=(), tokens=tokens)

        scores: list[ClinicalIntentScore] = []
        for intent, keywords in matches.items():
            if not keywords:
                continue
            confidence = round(len(keywords) / total_matches, 3)
            scores.append(
                ClinicalIntentScore(
                    intent=intent,
                    confidence=confidence,
                    keywords=tuple(sorted(keywords)),
                )
            )
        scores.sort(key=lambda item: item.confidence, reverse=True)
        return ClinicalIntentAnalysis(intents=tuple(scores), tokens=tokens)

    @staticmethod
    def parse_intent(value: str | ClinicalIntent | None) -> ClinicalIntent | None:
        if value is None:
            return None
        if isinstance(value, ClinicalIntent):
            return value
        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        for intent in ClinicalIntent:
            if cleaned in {intent.value, intent.name.lower()}:
                return intent
        return None

    def resolve_overrides(
        self, override: str | ClinicalIntent | Sequence[str | ClinicalIntent] | None
    ) -> tuple[ClinicalIntent, ...]:
        if override is None:
            return ()
        if isinstance(override, (str, ClinicalIntent)):
            intent = self.parse_intent(override)
            return (intent,) if intent else ()
        resolved: list[ClinicalIntent] = []
        for item in override:
            intent = self.parse_intent(item)
            if intent and intent not in resolved:
                resolved.append(intent)
        return tuple(resolved)


def _extract_values(metadata: Mapping[str, object], keys: Iterable[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            values.append(value.lower())
    return values


def infer_document_intents(metadata: Mapping[str, object]) -> set[ClinicalIntent]:
    """Infer potential clinical intents from stored chunk metadata."""

    intents: set[ClinicalIntent] = set()
    values = _extract_values(
        metadata,
        (
            "section_label",
            "section",
            "section_name",
            "section_title",
            "intent_hint",
            "intent",
        ),
    )
    nested = metadata.get("metadata")
    if isinstance(nested, Mapping):
        values.extend(
            _extract_values(
                nested,
                (
                    "section_label",
                    "section",
                    "intent_hint",
                    "intent",
                ),
            )
        )
    if not values:
        return intents
    corpus = " ".join(values)
    for intent, hints in DEFAULT_SECTION_HINTS.items():
        if any(hint in corpus for hint in hints):
            intents.add(intent)
    return intents


__all__ = [
    "ClinicalIntent",
    "ClinicalIntentAnalysis",
    "ClinicalIntentAnalyzer",
    "ClinicalIntentScore",
    "DEFAULT_INTENT_BOOSTS",
    "DEFAULT_INTENT_KEYWORDS",
    "DEFAULT_SECTION_HINTS",
    "infer_document_intents",
]

