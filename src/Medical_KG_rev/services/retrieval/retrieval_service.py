"""Multi-strategy retrieval service combining sparse and dense search."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .faiss_index import FAISSIndex
from .opensearch_client import OpenSearchClient
from .reranker import CrossEncoderReranker


@dataclass(slots=True)
class RetrievalResult:
    id: str
    text: str
    retrieval_score: float
    rerank_score: float | None
    highlights: Sequence[Mapping[str, object]]
    metadata: Mapping[str, object]


class RetrievalService:
    def __init__(
        self,
        opensearch: OpenSearchClient,
        faiss: FAISSIndex,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.opensearch = opensearch
        self.faiss = faiss
        self.reranker = reranker or CrossEncoderReranker()

    def search(
        self,
        index: str,
        query: str,
        filters: Mapping[str, object] | None = None,
        k: int = 10,
        rerank: bool = False,
    ) -> list[RetrievalResult]:
        bm25_results = self.opensearch.search(
            index, query, strategy="bm25", filters=filters, size=k
        )
        splade_results = self.opensearch.search(
            index, query, strategy="splade", filters=filters, size=k
        )
        dense_results = self._dense_search(query, k)
        fused = self._fuse_results([bm25_results, splade_results, dense_results])
        if rerank:
            fused = self._apply_rerank(query, fused)
        fused.sort(key=lambda item: item.rerank_score or item.retrieval_score, reverse=True)
        return fused

    def _dense_search(self, query: str, k: int) -> list[Mapping[str, object]]:
        if not self.faiss.ids:
            return []
        pseudo_query = [float(hash(token) % 100) for token in query.split()]
        if len(pseudo_query) < self.faiss.dimension:
            pseudo_query.extend([0.0] * (self.faiss.dimension - len(pseudo_query)))
        elif len(pseudo_query) > self.faiss.dimension:
            pseudo_query = pseudo_query[: self.faiss.dimension]
        hits = self.faiss.search(pseudo_query, k=k)
        results: list[Mapping[str, object]] = []
        for chunk_id, score, metadata in hits:
            results.append(
                {
                    "_id": chunk_id,
                    "_score": score,
                    "_source": {"text": metadata.get("text", ""), **metadata},
                    "highlight": [],
                }
            )
        return results

    def _fuse_results(
        self, result_sets: Sequence[Sequence[Mapping[str, object]]]
    ) -> list[RetrievalResult]:
        aggregated: dict[str, dict[str, object]] = {}
        for results in result_sets:
            for rank, result in enumerate(results, start=1):
                chunk_id = result["_id"]
                data = aggregated.setdefault(
                    chunk_id,
                    {
                        "text": result["_source"].get("text", ""),
                        "metadata": result["_source"],
                        "highlights": list(result.get("highlight", [])),
                        "rrf": 0.0,
                    },
                )
                data["rrf"] += 1.0 / (50 + rank)
        fused: list[RetrievalResult] = []
        for chunk_id, payload in aggregated.items():
            fused.append(
                RetrievalResult(
                    id=chunk_id,
                    text=str(payload["text"]),
                    retrieval_score=float(payload["rrf"]),
                    rerank_score=None,
                    highlights=list(payload["highlights"]),
                    metadata=dict(payload["metadata"]),
                )
            )
        fused.sort(key=lambda item: item.retrieval_score, reverse=True)
        return fused

    def _apply_rerank(
        self, query: str, results: Iterable[RetrievalResult]
    ) -> list[RetrievalResult]:
        materialised = list(results)
        candidates = [
            {"id": result.id, "text": result.text, **result.metadata} for result in materialised
        ]
        scored, _metrics = self.reranker.rerank(query, candidates)
        score_map = {item.get("id"): item.get("rerank_score", 0.0) for item in scored}
        reranked: list[RetrievalResult] = []
        for result in materialised:
            reranked.append(
                RetrievalResult(
                    id=result.id,
                    text=result.text,
                    retrieval_score=result.retrieval_score,
                    rerank_score=score_map.get(result.id),
                    highlights=result.highlights,
                    metadata=result.metadata,
                )
            )
        return reranked
