"""Sparse retrieval utilities for OpenSearch-backed adapters."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import log
from typing import Any


def _tokenise(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


@dataclass(slots=True)
class SparseDocument:
    doc_id: str
    text: str
    fields: Mapping[str, str] = field(default_factory=dict)
    rank_features: Mapping[str, float] = field(default_factory=dict)


class BM25Retriever:
    """Simplified BM25 retriever supporting filterable fields."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._documents: dict[str, SparseDocument] = {}
        self._index: dict[str, Counter[str]] = {}
        self._avg_len = 0.0

    def index_documents(self, documents: Iterable[SparseDocument]) -> None:
        docs = list(documents)
        if not docs:
            return
        total_length = 0
        for document in docs:
            tokens = Counter(_tokenise(document.text))
            self._documents[document.doc_id] = document
            self._index[document.doc_id] = tokens
            total_length += sum(tokens.values())
        self._avg_len = total_length / max(len(self._documents), 1)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[str, float, SparseDocument]]:
        if not self._documents:
            return []
        query_terms = Counter(_tokenise(query))
        results: list[tuple[str, float, SparseDocument]] = []
        filters = filters or {}
        for doc_id, tokens in self._index.items():
            document = self._documents[doc_id]
            if not self._passes_filters(document, filters):
                continue
            score = self._score(tokens, query_terms)
            if score <= 0:
                continue
            results.append((doc_id, score, document))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def _passes_filters(self, document: SparseDocument, filters: Mapping[str, Any]) -> bool:
        for key, expected in filters.items():
            if document.fields.get(key) != expected:
                return False
        return True

    def _score(self, tokens: Counter[str], query_terms: Counter[str]) -> float:
        score = 0.0
        doc_len = sum(tokens.values())
        for term, qf in query_terms.items():
            tf = tokens.get(term, 0)
            if tf == 0:
                continue
            df = sum(1 for counter in self._index.values() if term in counter)
            idf = log((len(self._index) - df + 0.5) / (df + 0.5) + 1)
            norm_tf = (
                tf
                * (self._k1 + 1)
                / (tf + self._k1 * (1 - self._b + self._b * doc_len / (self._avg_len or 1)))
            )
            score += idf * norm_tf * qf
        return score


class BM25FRetriever(BM25Retriever):
    """Multi-field BM25 implementation applying field boosts."""

    def __init__(self, *, boosts: Mapping[str, float] | None = None) -> None:
        super().__init__()
        self._boosts = dict(boosts or {})

    def index_documents(self, documents: Iterable[SparseDocument]) -> None:  # type: ignore[override]
        weighted_docs: list[SparseDocument] = []
        for document in documents:
            text = document.text
            for field, value in document.fields.items():
                boost = self._boosts.get(field, 1.0)
                text = f"{text} {' '.join([value] * int(boost))}" if value else text
            weighted_docs.append(SparseDocument(document.doc_id, text, document.fields))
        super().index_documents(weighted_docs)


class SPLADEDocWriter:
    """Writes SPLADE rank features for ingestion into OpenSearch."""

    def __init__(self) -> None:
        self._features: dict[str, Mapping[str, float]] = {}

    def write(self, doc_id: str, terms: Mapping[str, float]) -> None:
        self._features[doc_id] = dict(terms)

    def features_for(self, doc_id: str) -> Mapping[str, float]:
        return self._features.get(doc_id, {})


class SPLADEQueryEncoder:
    """Encodes queries into sparse term-weight dictionaries."""

    def encode(self, query: str) -> Mapping[str, float]:
        counts = Counter(_tokenise(query))
        total = sum(counts.values()) or 1
        return {term: freq / total for term, freq in counts.items()}


class NeuralSparseRetriever:
    """Simple neural sparse retriever using SPLADE features."""

    def __init__(self, writer: SPLADEDocWriter, encoder: SPLADEQueryEncoder | None = None) -> None:
        self.writer = writer
        self.encoder = encoder or SPLADEQueryEncoder()

    def search(
        self,
        *,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        encoded = self.encoder.encode(query)
        scores: list[tuple[str, float]] = []
        for doc_id, features in self.writer._features.items():
            score = sum(encoded.get(term, 0.0) * weight for term, weight in features.items())
            if score <= 0:
                continue
            scores.append((doc_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


class SparseQueryBuilder:
    """Builds OpenSearch-compatible DSL for hybrid sparse queries."""

    def __init__(self) -> None:
        self.must: list[Mapping[str, Any]] = []
        self.should: list[Mapping[str, Any]] = []
        self.filters: list[Mapping[str, Any]] = []

    def add_term(self, field: str, value: str) -> "SparseQueryBuilder":
        self.must.append({"term": {field: value}})
        return self

    def add_rank_feature(self, field: str, weight: float) -> "SparseQueryBuilder":
        self.should.append({"rank_feature": {"field": field, "boost": weight}})
        return self

    def add_filter(self, field: str, value: Any) -> "SparseQueryBuilder":
        self.filters.append({"term": {field: value}})
        return self

    def build(self) -> Mapping[str, Any]:
        query: dict[str, Any] = {"bool": {}}
        if self.must:
            query["bool"]["must"] = list(self.must)
        if self.should:
            query["bool"]["should"] = list(self.should)
            query["bool"]["minimum_should_match"] = 1
        if self.filters:
            query["bool"]["filter"] = list(self.filters)
        return query


__all__ = [
    "BM25FRetriever",
    "BM25Retriever",
    "NeuralSparseRetriever",
    "SPLADEDocWriter",
    "SPLADEQueryEncoder",
    "SparseDocument",
    "SparseQueryBuilder",
]
