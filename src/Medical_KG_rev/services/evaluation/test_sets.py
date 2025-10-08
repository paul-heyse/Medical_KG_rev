"""Utilities for loading and validating evaluation test sets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from random import Random
from typing import Iterable, Mapping, Sequence

import yaml


class QueryType(str, Enum):
    """Enumeration of supported query intents used for stratification."""

    EXACT_TERM = "exact_term"
    PARAPHRASE = "paraphrase"
    COMPLEX_CLINICAL = "complex_clinical"


@dataclass(slots=True, frozen=True)
class QueryJudgment:
    """Single query with graded relevance labels."""

    query_id: str
    query_text: str
    query_type: QueryType
    relevant_docs: tuple[tuple[str, float], ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_relevance_mapping(self) -> dict[str, float]:
        return {doc_id: float(grade) for doc_id, grade in self.relevant_docs}

    def has_relevant_document(self) -> bool:
        return any(grade > 0 for _, grade in self.relevant_docs)


@dataclass(slots=True)
class TestSet:
    """In-memory representation of a retrieval evaluation dataset."""

    name: str
    version: str
    queries: tuple[QueryJudgment, ...]
    source: Path | None = None

    def stratify(self) -> dict[QueryType, tuple[QueryJudgment, ...]]:
        buckets: dict[QueryType, list[QueryJudgment]] = defaultdict(list)
        for record in self.queries:
            buckets[record.query_type].append(record)
        return {key: tuple(value) for key, value in buckets.items()}

    def split(self, *, holdout_ratio: float = 0.2, seed: int = 7) -> tuple[TestSet, TestSet]:
        """Return (evaluation, hold-out) splits preserving stratification."""

        if not 0 < holdout_ratio < 1:
            raise ValueError("holdout_ratio must be between 0 and 1")
        rng = Random(seed)
        evaluation: list[QueryJudgment] = []
        holdout: list[QueryJudgment] = []
        for _, bucket in self.stratify().items():
            items = list(bucket)
            rng.shuffle(items)
            cutoff = max(1, int(len(items) * holdout_ratio)) if len(items) > 1 else 0
            holdout.extend(items[:cutoff])
            evaluation.extend(items[cutoff:])
        return (
            TestSet(name=f"{self.name}-eval", version=self.version, queries=tuple(evaluation)),
            TestSet(name=f"{self.name}-holdout", version=self.version, queries=tuple(holdout)),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "queries": [
                {
                    "query_id": query.query_id,
                    "query_text": query.query_text,
                    "query_type": query.query_type.value,
                    "relevant_docs": [
                        {"doc_id": doc_id, "grade": grade} for doc_id, grade in query.relevant_docs
                    ],
                    "metadata": dict(query.metadata),
                }
                for query in self.queries
            ],
        }

    def ensure_quality(self) -> None:
        """Validate schema constraints defined in the specification."""

        ids = {query.query_id for query in self.queries}
        if len(ids) != len(self.queries):
            raise ValueError("Query identifiers must be unique")
        for query in self.queries:
            if not query.query_text.strip():
                raise ValueError(f"Query '{query.query_id}' has empty text")
            if not query.has_relevant_document():
                raise ValueError(f"Query '{query.query_id}' is missing relevant documents")
            for doc_id, grade in query.relevant_docs:
                if not doc_id:
                    raise ValueError(f"Query '{query.query_id}' contains blank doc_id")
                if grade < 0 or grade > 3:
                    raise ValueError(
                        f"Query '{query.query_id}' has invalid grade {grade}; expected range 0-3"
                    )

    def describe(self) -> dict[str, float]:
        stratified = self.stratify()
        return {key.value: float(len(bucket)) for key, bucket in stratified.items()}


class TestSetManager:
    """Loads and caches evaluation datasets stored on disk."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or "eval/test_sets")
        self._cache: dict[tuple[str, str | None], TestSet] = {}

    def _resolve_path(self, name: str) -> Path:
        candidate = self.root / f"{name}.yaml"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Test set '{name}' not found at {candidate}")

    def load(self, name: str, *, expected_version: str | None = None) -> TestSet:
        cache_key = (name, expected_version)
        if cache_key in self._cache:
            return self._cache[cache_key]
        path = self._resolve_path(name)
        raw = yaml.safe_load(path.read_text()) or {}
        version = str(raw.get("version") or "unknown")
        if expected_version is not None and version != expected_version:
            raise ValueError(
                f"Requested version '{expected_version}' but file {path} declares version '{version}'"
            )
        queries = _parse_queries(raw.get("queries", []))
        test_set = TestSet(name=name, version=version, queries=tuple(queries), source=path)
        test_set.ensure_quality()
        self._cache[cache_key] = test_set
        return test_set

    def refresh(self, name: str, *, new_queries: Sequence[Mapping[str, object]], version: str) -> TestSet:
        """Create a new version of a dataset replacing the cached entry."""

        latest = self.root / f"{name}.yaml"
        archive = self.root / name / f"{version}.yaml"
        archive.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": version, "queries": list(new_queries)}
        serialised = yaml.safe_dump(payload, sort_keys=False)
        archive.write_text(serialised, encoding="utf-8")
        latest.write_text(serialised, encoding="utf-8")
        test_set = TestSet(name=name, version=version, queries=tuple(_parse_queries(new_queries)), source=archive)
        test_set.ensure_quality()
        self._cache[(name, version)] = test_set
        self._cache[(name, None)] = test_set
        return test_set


def _parse_queries(values: Iterable[Mapping[str, object]]) -> list[QueryJudgment]:
    records: list[QueryJudgment] = []
    for payload in values:
        query_id = str(payload.get("query_id"))
        query_text = str(payload.get("query_text"))
        query_type = QueryType(str(payload.get("query_type", QueryType.EXACT_TERM.value)))
        docs_raw = payload.get("relevant_docs", [])
        docs: list[tuple[str, float]] = []
        for entry in docs_raw:
            doc_id = str(entry.get("doc_id"))
            grade = float(entry.get("grade", 0))
            docs.append((doc_id, grade))
        metadata_payload = payload.get("metadata") or {}
        records.append(
            QueryJudgment(
                query_id=query_id,
                query_text=query_text,
                query_type=query_type,
                relevant_docs=tuple(docs),
                metadata=dict(metadata_payload),
            )
        )
    return records


def build_test_set(name: str, queries: Sequence[Mapping[str, object]], *, version: str) -> TestSet:
    test_set = TestSet(name=name, version=version, queries=tuple(_parse_queries(queries)))
    test_set.ensure_quality()
    return test_set


def cohens_kappa(labels_a: Sequence[object], labels_b: Sequence[object]) -> float:
    """Compute Cohen's kappa for two annotator label sequences."""

    if len(labels_a) != len(labels_b):
        raise ValueError("Sequences must be of equal length")
    if not labels_a:
        return 1.0
    categories = sorted(set(labels_a) | set(labels_b))
    totals = Counter(zip(labels_a, labels_b))
    total = float(len(labels_a))
    observed = sum(totals[(category, category)] for category in categories) / total
    marginal_a = Counter(labels_a)
    marginal_b = Counter(labels_b)
    expected = sum((marginal_a[category] / total) * (marginal_b[category] / total) for category in categories)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


__all__ = [
    "QueryJudgment",
    "QueryType",
    "TestSet",
    "TestSetManager",
    "build_test_set",
    "cohens_kappa",
]
