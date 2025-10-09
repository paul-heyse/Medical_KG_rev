"""Query intent classification helpers for table-aware routing."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class QueryIntent(str, Enum):
    """Supported query intent labels."""

    TABULAR = "tabular"
    NARRATIVE = "narrative"
    MIXED = "mixed"


@dataclass(slots=True, frozen=True)
class IntentClassification:
    """Result returned by :class:`IntentClassifier`."""

    intent: QueryIntent
    confidence: float
    matched_patterns: tuple[str, ...]
    override: QueryIntent | None = None


_DEFAULT_TABULAR_KEYWORDS: Mapping[str, float] = {
    "adverse events": 0.9,
    "side effects": 0.8,
    "effect sizes": 0.85,
    "effect size": 0.85,
    "outcome measures": 0.9,
    "outcome measure": 0.9,
    "results table": 1.0,
    "results tables": 1.0,
    "demographics table": 0.7,
    "baseline characteristics": 0.75,
    "ae table": 0.9,
}

_DEFAULT_TABULAR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgrade\s+[0-9]{1,2}\b", re.IGNORECASE),
    re.compile(r"\bctcae\b", re.IGNORECASE),
    re.compile(r"\bserious adverse events?\b", re.IGNORECASE),
    re.compile(r"\btable\s+[0-9ivx]+\b", re.IGNORECASE),
)


class IntentClassifier:
    """Rule-based query intent classifier."""

    def __init__(
        self,
        tabular_keywords: Mapping[str, float] | None = None,
        tabular_patterns: Iterable[re.Pattern[str]] | None = None,
    ) -> None:
        self._keywords = dict(_DEFAULT_TABULAR_KEYWORDS)
        if tabular_keywords:
            for key, weight in tabular_keywords.items():
                if weight <= 0:
                    continue
                self._keywords[key.lower()] = min(float(weight), 1.0)
        self._patterns = tuple(tabular_patterns) if tabular_patterns else _DEFAULT_TABULAR_PATTERNS

    # ------------------------------------------------------------------
    def classify(
        self,
        query: str,
        *,
        override: QueryIntent | str | None = None,
    ) -> IntentClassification:
        """Return the detected intent for *query*.

        Manual overrides (either :class:`QueryIntent` or string values) take
        precedence and result in a classification with confidence ``1.0``.
        """
        query = (query or "").strip()
        override_intent = self._normalise_override(override)
        if override_intent is not None:
            return IntentClassification(
                intent=override_intent,
                confidence=1.0,
                matched_patterns=(),
                override=override_intent,
            )

        lowered = query.lower()
        best_match = 0.0
        matched: list[str] = []
        for keyword, weight in self._keywords.items():
            if keyword in lowered:
                matched.append(keyword)
                best_match = max(best_match, weight)
        for pattern in self._patterns:
            if pattern.search(query):
                matched.append(pattern.pattern)
                best_match = max(best_match, 0.75)

        confidence = min(max(best_match, 0.0), 1.0)
        if confidence >= 0.65:
            intent = QueryIntent.TABULAR
        elif 0.25 <= confidence < 0.65:
            intent = QueryIntent.MIXED
        else:
            intent = QueryIntent.NARRATIVE
        return IntentClassification(
            intent=intent,
            confidence=confidence,
            matched_patterns=tuple(sorted(set(matched))),
            override=None,
        )

    # ------------------------------------------------------------------
    def benchmark(self, labelled_queries: Mapping[str, QueryIntent]) -> float:
        """Return accuracy for *labelled_queries*.

        The helper is lightweight and intended for unit-level regression
        coverage rather than runtime monitoring.
        """
        if not labelled_queries:
            return 1.0
        correct = 0
        for query, expected in labelled_queries.items():
            detected = self.classify(query).intent
            if detected == expected:
                correct += 1
        return correct / float(len(labelled_queries))

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_override(value: QueryIntent | str | None) -> QueryIntent | None:
        if value is None:
            return None
        if isinstance(value, QueryIntent):
            return value
        try:
            return QueryIntent(str(value).strip().lower())
        except ValueError:
            logger.debug("intent_classifier.invalid_override", override=value)
            return None


__all__ = [
    "IntentClassification",
    "IntentClassifier",
    "QueryIntent",
]
