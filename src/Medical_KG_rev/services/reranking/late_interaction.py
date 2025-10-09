"""Late interaction reranking based on simplified ColBERT style scoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from time import monotonic, perf_counter
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import structlog

from .base import BaseReranker
from .errors import RerankingError
from .models import QueryDocumentPair, RerankResult, RerankingResponse
from .utils import FeatureView, clamp

logger = structlog.get_logger(__name__)


def _cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    numerator = 0.0
    sum_a = 0.0
    sum_b = 0.0
    for value_a, value_b in zip(vec_a, vec_b, strict=False):
        numerator += value_a * value_b
        sum_a += value_a * value_a
        sum_b += value_b * value_b
    if sum_a == 0 or sum_b == 0:
        return 0.0
    return numerator / (sqrt(sum_a) * sqrt(sum_b))


def _normalise_vectors(raw: Sequence[Any]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for vector in raw:
        if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes, bytearray)):
            numeric = [float(value) for value in vector if isinstance(value, (int, float))]
            if numeric:
                vectors.append(numeric)
    return vectors


@dataclass(slots=True)
class _CacheEntry:
    expires_at: float
    vectors: Sequence[Sequence[float]]


class ColBERTReranker(BaseReranker):
    """Implements MaxSim using metadata-supplied token vectors."""

    def __init__(self, batch_size: int = 16, cache_ttl: int = 300) -> None:
        super().__init__(
            identifier="colbertv2-maxsim",
            model_version="v1.0",
            batch_size=batch_size,
            requires_gpu=False,
        )
        self.cache_ttl = cache_ttl
        self._vector_cache: MutableMapping[str, _CacheEntry] = {}

    # ------------------------------------------------------------------
    def score_pairs(
        self,
        pairs: Sequence[QueryDocumentPair],
        *,
        top_k: int | None = None,
        normalize: bool = True,
        batch_size: int | None = None,
    ) -> RerankingResponse:
        """Perform a cached batch MaxSim computation."""

        started = perf_counter()
        limit = len(pairs) if top_k is None else min(len(pairs), top_k)
        prepared = self._prepare_vectors(pairs[:limit])
        evaluated: list[tuple[str, float, int, Mapping[str, object]]] = []
        for rank, (pair, query_vectors, doc_vectors) in enumerate(prepared, start=1):
            score = self._maxsim(query_vectors, doc_vectors)
            evaluated.append((pair.doc_id, score, rank, dict(pair.metadata)))

        raw_scores = [score for _, score, _, _ in evaluated]
        normalised = list(raw_scores)
        if normalize and normalised:
            minimum = min(normalised)
            maximum = max(normalised)
            if maximum != minimum:
                span = maximum - minimum
                normalised = [(score - minimum) / span for score in normalised]
            else:
                normalised = [0.5 for _ in normalised]
        duration = perf_counter() - started
        results = []
        for idx, (doc_id, raw_score, rank, metadata) in enumerate(evaluated):
            score_value = normalised[idx] if normalize and idx < len(normalised) else raw_score
            results.append(
                RerankResult(
                    doc_id=doc_id,
                    score=float(clamp(score_value)),
                    rank=rank,
                    metadata=dict(metadata) | {"maxsim_raw": raw_score},
                )
            )
        metrics = {
            "model": self.identifier,
            "version": self.model_version,
            "evaluated": len(results),
            "batch_size": batch_size or self.batch_size,
            "duration_ms": round(duration * 1000, 3),
            "cache_entries": len(self._vector_cache),
        }
        logger.debug(
            "colbert.batch_scored",
            reranker=self.identifier,
            evaluated=len(results),
            duration_ms=metrics["duration_ms"],
            cache_entries=metrics["cache_entries"],
        )
        return RerankingResponse(results=results, metrics=metrics)

    # ------------------------------------------------------------------
    def _prepare_vectors(
        self, pairs: Sequence[QueryDocumentPair]
    ) -> list[tuple[QueryDocumentPair, Sequence[Sequence[float]], Sequence[Sequence[float]]]]:
        prepared: list[
            tuple[QueryDocumentPair, Sequence[Sequence[float]], Sequence[Sequence[float]]]
        ] = []
        for pair in pairs:
            view = FeatureView(pair.metadata)
            query_vectors = self._query_vectors_from_metadata(view)
            doc_vectors = self._doc_vectors_from_metadata(pair.doc_id, view)
            prepared.append((pair, query_vectors, doc_vectors))
        return prepared

    # ------------------------------------------------------------------
    def _query_vectors_from_metadata(self, view: FeatureView) -> Sequence[Sequence[float]]:
        raw = view.get_sequence("query_vectors")
        if not raw:
            return []
        return _normalise_vectors(raw)

    # ------------------------------------------------------------------
    def _doc_vectors_from_metadata(
        self, doc_id: str, view: FeatureView
    ) -> Sequence[Sequence[float]]:
        cached = self._vector_cache.get(doc_id)
        now = monotonic()
        if cached and cached.expires_at > now:
            return cached.vectors
        vectors = _normalise_vectors(view.get_sequence("doc_vectors"))
        self._vector_cache[doc_id] = _CacheEntry(
            expires_at=now + float(self.cache_ttl),
            vectors=vectors,
        )
        return vectors

    # ------------------------------------------------------------------
    def _maxsim(
        self,
        query_vectors: Sequence[Sequence[float]],
        doc_vectors: Sequence[Sequence[float]],
    ) -> float:
        if not query_vectors or not doc_vectors:
            return 0.0
        max_sim = 0.0
        for query_vector in query_vectors:
            similarities = [
                _cosine_similarity(query_vector, doc_vector) for doc_vector in doc_vectors
            ]
            if similarities:
                max_sim += max(similarities)
        return clamp(max_sim / len(query_vectors))


class ColbertIndexReranker(ColBERTReranker):
    """Fetch token vectors from an external ColBERT-style index."""

    def __init__(
        self,
        index: object,
        *,
        batch_size: int = 16,
        cache_ttl: int = 300,
    ) -> None:
        super().__init__(batch_size=batch_size, cache_ttl=cache_ttl)
        self.identifier = "colbertv2-external-index"
        self._index = index

    def _prepare_vectors(
        self, pairs: Sequence[QueryDocumentPair]
    ) -> list[tuple[QueryDocumentPair, Sequence[Sequence[float]], Sequence[Sequence[float]]]]:
        if not hasattr(self._index, "encode_queries") or not hasattr(
            self._index, "get_document_vectors"
        ):
            raise RerankingError(
                title="ColBERT integration error",
                status=500,
                detail="External index is missing required ColBERT interfaces",
            )
        queries = [pair.query for pair in pairs]
        encoded = self._index.encode_queries(queries)
        prepared: list[
            tuple[QueryDocumentPair, Sequence[Sequence[float]], Sequence[Sequence[float]]]
        ] = []
        for pair, query_vectors in zip(pairs, encoded, strict=False):
            doc_vectors = self._cached_fetch(pair.doc_id, self._index.get_document_vectors)
            prepared.append(
                (pair, _normalise_vectors(query_vectors), _normalise_vectors(doc_vectors))
            )
        return prepared

    def _cached_fetch(
        self,
        doc_id: str,
        loader: Callable[[str], Sequence[Sequence[float]]],
    ) -> Sequence[Sequence[float]]:
        cached = self._vector_cache.get(doc_id)
        now = monotonic()
        if cached and cached.expires_at > now:
            return cached.vectors
        vectors = loader(doc_id)
        self._vector_cache[doc_id] = _CacheEntry(
            expires_at=now + float(self.cache_ttl),
            vectors=_normalise_vectors(vectors),
        )
        return self._vector_cache[doc_id].vectors


class QdrantColBERTReranker(ColBERTReranker):
    """Retrieve ColBERT token vectors from a Qdrant multivector collection."""

    def __init__(
        self,
        client: object,
        collection: str,
        *,
        batch_size: int = 16,
        cache_ttl: int = 300,
    ) -> None:
        super().__init__(batch_size=batch_size, cache_ttl=cache_ttl)
        self.identifier = "colbertv2-qdrant"
        self._client = client
        self._collection = collection

    def _prepare_vectors(
        self, pairs: Sequence[QueryDocumentPair]
    ) -> list[tuple[QueryDocumentPair, Sequence[Sequence[float]], Sequence[Sequence[float]]]]:
        if not hasattr(self._client, "retrieve"):
            raise RerankingError(
                title="Qdrant integration error",
                status=500,
                detail="Qdrant client does not expose a 'retrieve' method",
            )
        prepared: list[
            tuple[QueryDocumentPair, Sequence[Sequence[float]], Sequence[Sequence[float]]]
        ] = []
        ids = [pair.doc_id for pair in pairs]
        records = self._client.retrieve(  # type: ignore[call-arg]
            collection_name=self._collection,
            ids=ids,
            with_vectors=True,
        )
        record_map = {str(record.id): record for record in records}
        for pair in pairs:
            record = record_map.get(pair.doc_id)
            doc_vectors: Sequence[Sequence[float]] = []
            if record is not None:
                vectors = getattr(record, "vectors", None)
                if isinstance(vectors, Mapping):
                    doc_vectors = _normalise_vectors(list(vectors.values()))
            view = FeatureView(pair.metadata)
            query_vectors = self._query_vectors_from_metadata(view)
            if not query_vectors:
                logger.debug("qdrant.colbert.missing_query_vectors", doc=pair.doc_id)
            prepared.append((pair, query_vectors, doc_vectors))
        return prepared
