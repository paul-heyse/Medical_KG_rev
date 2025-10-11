"""In-memory OpenSearch-like client for tests and local workflows.

Key Responsibilities:
    - Provide a lightweight stand-in for OpenSearch indexing and search APIs.
    - Support index template registration and document storage.
    - Execute simple TF-IDF style scoring for unit tests.

Collaborators:
    - Upstream: Retrieval components requiring OpenSearch behavior in tests.
    - Downstream: None; data is stored in process memory only.

Side Effects:
    - Maintains an in-memory index of documents and templates.

Thread Safety:
    - Not thread-safe: Designed for single-threaded test scenarios.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import math



@dataclass(slots=True)
class DocumentIndexTemplate:
    """Index template metadata tracked by the mock client.

    Attributes:
        name: Template name.
        settings: Index settings definition.
        mappings: Field mappings associated with the template.
    """

    name: str
    settings: Mapping[str, object]
    mappings: Mapping[str, object]


@dataclass(slots=True)
class _IndexedDocument:
    """Internal representation of an indexed document."""

    doc_id: str
    body: Mapping[str, object]


class OpenSearchClient:
    """Minimal in-memory implementation of OpenSearch APIs."""

    def __init__(self) -> None:
        """Create an empty in-memory index store."""
        self._indices: MutableMapping[str, dict[str, _IndexedDocument]] = defaultdict(dict)
        self._templates: dict[str, DocumentIndexTemplate] = {}

    def put_index_template(self, template: DocumentIndexTemplate) -> None:
        """Register or replace an index template."""
        self._templates[template.name] = template

    def index(self, index: str, doc_id: str, body: Mapping[str, object]) -> None:
        """Index a single document into the in-memory store."""
        stored = dict(body)
        metadata = stored.get("metadata")
        if isinstance(metadata, Mapping):
            profile = metadata.get("chunking_profile")
            if profile and "chunking_profile" not in stored:
                stored["chunking_profile"] = profile
        self._indices[index][doc_id] = _IndexedDocument(doc_id=doc_id, body=stored)

    def bulk_index(
        self, index: str, documents: Sequence[Mapping[str, object]], id_field: str
    ) -> None:
        """Bulk insert a collection of documents."""
        for doc in documents:
            doc_id = str(doc[id_field])
            self.index(index, doc_id, doc)

    def get(self, index: str, doc_id: str) -> Mapping[str, object] | None:
        """Fetch a document body if it exists in the index."""
        stored = self._indices.get(index, {}).get(doc_id)
        if stored is None:
            return None
        return stored.body

    def has_document(self, index: str, doc_id: str) -> bool:
        """Check whether a document is present in the simulated index."""
        return doc_id in self._indices.get(index, {})

    def search(
        self,
        index: str,
        query: str,
        strategy: str = "bm25",
        filters: Mapping[str, object] | None = None,
        highlight: bool = True,
        size: int = 10,
    ) -> list[Mapping[str, object]]:
        """Execute a search against the in-memory index.

        Args:
            index: Name of the index to search.
            query: Plain-text query tokens.
            strategy: Scoring strategy (e.g., ``bm25`` or ``splade``).
            filters: Optional field filters applied before scoring.
            highlight: Whether to attach naive highlight snippets.
            size: Maximum number of results to return.

        Returns:
            A list of search hit dictionaries compatible with OpenSearch responses.
        """
        documents = list(self._indices.get(index, {}).values())
        filtered = self._apply_filters(documents, filters or {})
        scored = self._score(filtered, query, strategy)
        scored.sort(key=lambda item: item["_score"], reverse=True)
        results = scored[:size]
        if highlight:
            for result in results:
                result["highlight"] = self._highlight(result["_source"]["text"], query)
        return results

    def _apply_filters(
        self, documents: Iterable[_IndexedDocument], filters: Mapping[str, object]
    ) -> list[_IndexedDocument]:
        """Apply equality filters to the in-memory documents."""
        if not filters:
            return list(documents)
        filtered = []
        for doc in documents:
            body = doc.body
            if all(body.get(key) == value for key, value in filters.items()):
                filtered.append(doc)
        return filtered

    def _score(
        self,
        documents: Iterable[_IndexedDocument],
        query: str,
        strategy: str,
    ) -> list[dict[str, object]]:
        """Score documents using the requested strategy."""
        scores: list[dict[str, object]] = []
        query_terms = [term for term in query.lower().split() if term]
        for doc in documents:
            text = str(doc.body.get("text", ""))
            tokens = [token for token in text.lower().split() if token]
            tf = sum(tokens.count(term) for term in query_terms)
            if strategy == "bm25":
                score = self._bm25(tf, len(tokens))
            elif strategy == "splade":
                score = self._splade(tf, query_terms, tokens)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            scores.append({"_id": doc.doc_id, "_score": score, "_source": doc.body})
        return scores

    def _bm25(self, term_frequency: int, length: int) -> float:
        """Compute a simple BM25-like score for a document."""
        if length == 0:
            return 0.0
        k1 = 1.5
        b = 0.75
        avgdl = max(length, 1)
        return (term_frequency * (k1 + 1)) / (term_frequency + k1 * (1 - b + b * (length / avgdl)))

    def _splade(
        self, term_frequency: int, query_terms: Sequence[str], tokens: Sequence[str]
    ) -> float:
        """Approximate a SPLADE score using query-term overlap heuristics."""
        unique_terms = len(set(query_terms) & set(tokens))
        return math.log1p(term_frequency + unique_terms)

    def _highlight(self, text: str, query: str) -> list[Mapping[str, object]]:
        """Generate naive highlight spans for matching terms."""
        spans: list[Mapping[str, object]] = []
        lower = text.lower()
        for term in {t for t in query.lower().split() if t}:
            start = lower.find(term)
            if start == -1:
                continue
            end = start + len(term)
            spans.append({"term": term, "start": start, "end": end, "text": text[start:end]})
        return spans
