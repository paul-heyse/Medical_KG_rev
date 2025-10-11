"""Feature extraction pipeline utilities for learning-to-rank rerankers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Protocol

from .models import QueryDocumentPair



class FeatureExtractor(Protocol):
    """Protocol describing a single feature extractor."""

    name: str

    def extract(self, pair: QueryDocumentPair) -> float:
        """Return a numeric feature value for the supplied pair."""


def _safe_numeric(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


@dataclass(slots=True)
class MetadataFeature:
    """Feature extractor pulling values from metadata by key."""

    name: str
    key: str
    transform: Callable[[float], float] | None = None
    default: float = 0.0

    def extract(self, pair: QueryDocumentPair) -> float:
        value = _safe_numeric(pair.metadata.get(self.key), self.default)
        return self.transform(value) if self.transform else value


@dataclass(slots=True)
class DocumentLengthFeature:
    """Feature representing the token length of the candidate document."""

    name: str = "document_length"
    max_length: int = 4096

    def extract(self, pair: QueryDocumentPair) -> float:
        tokens = pair.text.split()
        length = min(len(tokens), self.max_length)
        return length / float(self.max_length)


@dataclass(slots=True)
class QueryDocumentOverlapFeature:
    """Lexical overlap ratio between the query and the document."""

    name: str = "query_document_overlap"

    def extract(self, pair: QueryDocumentPair) -> float:
        query_terms = {term for term in pair.query.lower().split() if term}
        document_terms = {term for term in pair.text.lower().split() if term}
        if not query_terms or not document_terms:
            return 0.0
        intersection = len(query_terms & document_terms)
        return intersection / float(len(query_terms))


@dataclass(slots=True)
class FeaturePipeline:
    """Composable feature extraction pipeline."""

    extractors: Sequence[FeatureExtractor]
    post_processors: Sequence[Callable[[MutableMapping[str, float]], None]] = field(
        default_factory=list
    )

    def extract(self, pair: QueryDocumentPair) -> Mapping[str, float]:
        features: MutableMapping[str, float] = {}
        for extractor in self.extractors:
            features[extractor.name] = float(extractor.extract(pair))
        for processor in self.post_processors:
            processor(features)
        return features

    def batch(self, pairs: Sequence[QueryDocumentPair]) -> list[Mapping[str, float]]:
        return [self.extract(pair) for pair in pairs]

    def feature_names(self) -> list[str]:
        return [extractor.name for extractor in self.extractors]

    @classmethod
    def default(cls) -> FeaturePipeline:
        return cls(
            extractors=[
                MetadataFeature(name="bm25_score", key="bm25_score"),
                MetadataFeature(name="splade_score", key="splade_score"),
                MetadataFeature(name="dense_score", key="dense_score"),
                MetadataFeature(
                    name="recency",
                    key="recency_days",
                    transform=lambda value: 1.0 / (1.0 + max(value, 0.0)),
                ),
                DocumentLengthFeature(),
                QueryDocumentOverlapFeature(),
            ],
            post_processors=[_derive_interaction_feature],
        )


def _derive_interaction_feature(features: MutableMapping[str, float]) -> None:
    """Derive interaction terms once base features have been extracted."""
    lexical = features.get("bm25_score", 0.0)
    dense = features.get("dense_score", 0.0)
    overlap = features.get("query_document_overlap", 0.0)
    features["lexical_semantic_interaction"] = (
        mean([lexical, dense, overlap])
        if any(value for value in (lexical, dense, overlap))
        else 0.0
    )


@dataclass(slots=True)
class FeatureVector:
    """Utility wrapper bundling features with the originating document id."""

    doc_id: str
    values: Mapping[str, float]

    def as_ordered(self, feature_order: Sequence[str]) -> list[float]:
        return [float(self.values.get(name, 0.0)) for name in feature_order]
