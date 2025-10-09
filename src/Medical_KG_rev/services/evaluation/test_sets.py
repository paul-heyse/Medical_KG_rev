"""Utilities for loading and validating evaluation test sets.

This module provides utilities for loading, validating, and managing
evaluation test sets used for retrieval system evaluation. It includes
data structures for representing query judgments, test set management,
and statistical analysis functions.

The module defines:
- QueryType: Enumeration of supported query intents
- QueryJudgment: Single query with graded relevance labels
- TestSet: In-memory representation of evaluation datasets
- TestSetManager: Loads and caches evaluation datasets
- build_test_set: Factory function for creating test sets
- cohens_kappa: Statistical function for inter-annotator agreement

Architecture:
- Immutable data structures for test set representation
- Caching mechanism for loaded datasets
- Support for both filesystem and packaged resources
- Quality validation and schema enforcement
- Stratified sampling and splitting capabilities

Thread Safety:
- TestSet instances are immutable and thread-safe
- TestSetManager is thread-safe for concurrent access
- Statistical functions are stateless and thread-safe

Performance:
- Efficient caching of loaded datasets
- Lazy loading of test set data
- Optimized statistical computations

Examples:
    # Load a test set
    manager = TestSetManager()
    test_set = manager.load("medical_queries")

    # Create stratified splits
    eval_set, holdout_set = test_set.split(holdout_ratio=0.2)

    # Validate quality
    test_set.ensure_quality()
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from importlib.abc import Traversable
from pathlib import Path
from random import Random

import yaml

# ==============================================================================
# TYPE DEFINITIONS & CONSTANTS
# ==============================================================================
_DATA_PACKAGE = "Medical_KG_rev.services.evaluation.data"
_DEFAULT_DATASET_SUBDIR = "test_sets"


# ==============================================================================
# DATA MODELS
# ==============================================================================
class QueryType(str, Enum):
    """Enumeration of supported query intents used for stratification.

    This enumeration defines the different types of queries used in
    evaluation test sets, enabling stratified sampling and analysis
    of retrieval performance across different query types.

    Values:
        EXACT_TERM: Queries using exact medical terminology
        PARAPHRASE: Queries using paraphrased or alternative terms
        COMPLEX_CLINICAL: Complex clinical scenario queries

    Thread Safety:
        Enum values are immutable and thread-safe.

    Examples:
        query_type = QueryType.EXACT_TERM
        if query_type == QueryType.COMPLEX_CLINICAL:
            # Handle complex clinical query
    """

    EXACT_TERM = "exact_term"
    PARAPHRASE = "paraphrase"
    COMPLEX_CLINICAL = "complex_clinical"


@dataclass(slots=True, frozen=True)
class QueryJudgment:
    """Single query with graded relevance labels.

    This dataclass represents a single query with its associated
    relevance judgments for evaluation purposes. It provides
    methods for accessing relevance information and validation.

    Attributes:
        query_id: Unique identifier for the query
        query_text: The query text
        query_type: Type of query for stratification
        relevant_docs: Tuple of (doc_id, relevance_grade) pairs
        metadata: Additional metadata for the query

    Thread Safety:
        Immutable dataclass, thread-safe.

    Examples:
        judgment = QueryJudgment(
            query_id="q1",
            query_text="diabetes treatment",
            query_type=QueryType.EXACT_TERM,
            relevant_docs=(("doc1", 3.0), ("doc2", 2.0))
        )
        relevance_map = judgment.as_relevance_mapping()
    """

    query_id: str
    query_text: str
    query_type: QueryType
    relevant_docs: tuple[tuple[str, float], ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_relevance_mapping(self) -> dict[str, float]:
        """Convert relevant documents to a mapping.

        Returns:
            Dictionary mapping document IDs to relevance grades

        Raises:
            None: This method never raises exceptions.
        """
        return {doc_id: float(grade) for doc_id, grade in self.relevant_docs}

    def has_relevant_document(self) -> bool:
        """Check if the query has any relevant documents.

        Returns:
            True if any document has relevance grade > 0

        Raises:
            None: This method never raises exceptions.
        """
        return any(grade > 0 for _, grade in self.relevant_docs)


@dataclass(slots=True)
class TestSet:
    """In-memory representation of a retrieval evaluation dataset.

    This dataclass represents a complete evaluation test set with
    queries, relevance judgments, and metadata. It provides methods
    for stratification, splitting, validation, and serialization.

    Attributes:
        name: Name of the test set
        version: Version identifier
        queries: Tuple of query judgments
        source: Optional source file path

    Thread Safety:
        Immutable dataclass, thread-safe.

    Examples:
        test_set = TestSet(
            name="medical_queries",
            version="1.0",
            queries=(query1, query2, query3)
        )
        eval_set, holdout_set = test_set.split(holdout_ratio=0.2)
    """

    name: str
    version: str
    queries: tuple[QueryJudgment, ...]
    source: Path | None = None

    def stratify(self) -> dict[QueryType, tuple[QueryJudgment, ...]]:
        """Stratify queries by type.

        Returns:
            Dictionary mapping query types to query tuples

        Raises:
            None: This method never raises exceptions.
        """
        buckets: dict[QueryType, list[QueryJudgment]] = defaultdict(list)
        for record in self.queries:
            buckets[record.query_type].append(record)
        return {key: tuple(value) for key, value in buckets.items()}

    def split(self, *, holdout_ratio: float = 0.2, seed: int = 7) -> tuple[TestSet, TestSet]:
        """Return (evaluation, hold-out) splits preserving stratification.

        Args:
            holdout_ratio: Fraction of queries to reserve for holdout
            seed: Random seed for reproducible splits

        Returns:
            Tuple of (evaluation_set, holdout_set)

        Raises:
            ValueError: If holdout_ratio is not between 0 and 1
        """
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
        """Convert the test set to a serializable payload.

        Returns:
            Dictionary representation suitable for serialization

        Raises:
            None: This method never raises exceptions.
        """
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
        """Validate schema constraints defined in the specification.

        Raises:
            ValueError: If the test set fails quality validation
        """
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
        """Generate descriptive statistics for the test set.

        Returns:
            Dictionary mapping query types to counts

        Raises:
            None: This method never raises exceptions.
        """
        stratified = self.stratify()
        return {key.value: float(len(bucket)) for key, bucket in stratified.items()}


# ==============================================================================
# TEST SET MANAGEMENT
# ==============================================================================
class TestSetManager:
    """Loads and caches evaluation datasets stored on disk.

    This class provides a manager for loading and caching evaluation
    test sets from both filesystem and packaged resources. It supports
    version validation, quality checking, and efficient caching.

    Attributes:
        root: Optional filesystem root for test sets
        _resource_root: Packaged resource root for test sets
        _cache: Cache of loaded test sets

    Thread Safety:
        Thread-safe for concurrent access to cached test sets.

    Performance:
        Efficient caching prevents repeated file I/O.
        Lazy loading of test set data.

    Examples:
        # Load from packaged resources
        manager = TestSetManager()
        test_set = manager.load("medical_queries")

        # Load from filesystem
        manager = TestSetManager(root="/path/to/test_sets")
        test_set = manager.load("custom_queries")
    """

    def __init__(self, root: str | Path | None = None) -> None:
        """Initialize the test set manager.

        Args:
            root: Optional filesystem root for test sets

        Raises:
            None: Initialization always succeeds.
        """
        self.root = Path(root) if root is not None else None
        self._resource_root: Traversable | None
        if self.root is None:
            self._resource_root = (
                resources.files(_DATA_PACKAGE) / _DEFAULT_DATASET_SUBDIR
            )
        else:
            self._resource_root = None
        self._cache: dict[tuple[str, str | None], TestSet] = {}

    def _resolve_path(self, name: str) -> Path | Traversable:
        """Resolve the path to a test set file.

        Args:
            name: Name of the test set

        Returns:
            Path or Traversable object for the test set file

        Raises:
            FileNotFoundError: If the test set is not found
        """
        filename = f"{name}.yaml"
        if self.root is not None:
            candidate = self.root / filename
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"Test set '{name}' not found at {candidate}")
        if self._resource_root is not None:
            resource = self._resource_root / filename
            if resource.is_file():
                return resource
        raise FileNotFoundError(
            f"Test set '{name}' not found in packaged resources or provided root"
        )

    def _load_yaml(self, location: Path | Traversable) -> dict[str, object]:
        """Load YAML data from a file location.

        Args:
            location: Path or Traversable object

        Returns:
            Parsed YAML data as dictionary

        Raises:
            YAMLError: If YAML parsing fails
            IOError: If file reading fails
        """
        if isinstance(location, Path):
            text = location.read_text(encoding="utf-8")
        else:
            assert self._resource_root is not None
            text = location.read_text(encoding="utf-8")
        return yaml.safe_load(text) or {}

    def load(self, name: str, *, expected_version: str | None = None) -> TestSet:
        """Load a test set from storage.

        Args:
            name: Name of the test set to load
            expected_version: Optional expected version for validation

        Returns:
            Loaded and validated test set

        Raises:
            FileNotFoundError: If the test set is not found
            ValueError: If version validation fails
            ValueError: If quality validation fails
        """
        cache_key = (name, expected_version)
        if cache_key in self._cache:
            return self._cache[cache_key]
        path = self._resolve_path(name)
        raw = self._load_yaml(path)
        version = str(raw.get("version") or "unknown")
        if expected_version is not None and version != expected_version:
            raise ValueError(
                f"Requested version '{expected_version}' but file {path} declares version '{version}'"
            )
        queries = _parse_queries(raw.get("queries", []))
        source_path = path if isinstance(path, Path) else None
        test_set = TestSet(
            name=name,
            version=version,
            queries=tuple(queries),
            source=source_path,
        )
        test_set.ensure_quality()
        self._cache[cache_key] = test_set
        return test_set

    def refresh(self, name: str, *, new_queries: Sequence[Mapping[str, object]], version: str) -> TestSet:
        """Create a new version of a dataset replacing the cached entry.

        Args:
            name: Name of the test set
            new_queries: New query data to replace existing queries
            version: Version identifier for the new dataset

        Returns:
            New test set with updated queries

        Raises:
            RuntimeError: If manager was initialized without filesystem root
            ValueError: If quality validation fails
            IOError: If file writing fails
        """
        if self.root is None:
            raise RuntimeError(
                "Cannot refresh packaged datasets; provide a filesystem root when instantiating"
            )
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


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def _parse_queries(values: Iterable[Mapping[str, object]]) -> list[QueryJudgment]:
    """Parse query data from raw payloads.

    Args:
        values: Iterable of query payload dictionaries

    Returns:
        List of parsed QueryJudgment objects

    Raises:
        ValueError: If query data is invalid
    """
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
    """Build a test set from query data.

    Args:
        name: Name of the test set
        queries: Sequence of query payload dictionaries
        version: Version identifier

    Returns:
        Validated test set

    Raises:
        ValueError: If quality validation fails
    """
    test_set = TestSet(name=name, version=version, queries=tuple(_parse_queries(queries)))
    test_set.ensure_quality()
    return test_set


def cohens_kappa(labels_a: Sequence[object], labels_b: Sequence[object]) -> float:
    """Compute Cohen's kappa for two annotator label sequences.

    Args:
        labels_a: First annotator's labels
        labels_b: Second annotator's labels

    Returns:
        Cohen's kappa coefficient (0-1)

    Raises:
        ValueError: If sequences have different lengths
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Sequences must be of equal length")
    if not labels_a:
        return 1.0
    categories = sorted(set(labels_a) | set(labels_b))
    totals = Counter(zip(labels_a, labels_b, strict=True))
    total = float(len(labels_a))
    observed = sum(totals[(category, category)] for category in categories) / total
    marginal_a = Counter(labels_a)
    marginal_b = Counter(labels_b)
    expected = sum((marginal_a[category] / total) * (marginal_b[category] / total) for category in categories)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "QueryJudgment",
    "QueryType",
    "TestSet",
    "TestSetManager",
    "build_test_set",
    "cohens_kappa",
]
