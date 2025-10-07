"""Evaluation harness for modular chunkers."""

from __future__ import annotations

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from Medical_KG_rev.chunking.models import ChunkerConfig
from Medical_KG_rev.chunking.registry import default_registry
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.chunking.ports import BaseChunker
from Medical_KG_rev.models.ir import Block, BlockType, Document, Section
from Medical_KG_rev.chunking.exceptions import ChunkerConfigurationError


@dataclass(slots=True)
class GoldQuery:
    text: str
    keywords: Sequence[str]


@dataclass(slots=True)
class GoldSample:
    doc_id: str
    source: str
    title: str
    text: str
    boundaries: Sequence[int]
    queries: Sequence[GoldQuery]


@dataclass(slots=True)
class EvaluationSummary:
    chunker: str
    boundary_f1: float
    recall_at_20: float
    ndcg_at_10: float
    ndcg_at_20: float
    latency_ms: float
    chunk_count: float


class ChunkingEvaluationRunner:
    """Loads gold samples and evaluates configured chunkers."""

    def __init__(
        self,
        chunkers: Sequence[str],
        *,
        gold_path: Path | None = None,
        tenant_id: str = "eval",
    ) -> None:
        self.chunker_names = list(chunkers)
        self.gold_path = gold_path or Path(__file__).resolve().parent / "gold"
        self.tenant_id = tenant_id
        self.registry = default_registry()

    def run(self) -> dict[str, EvaluationSummary]:
        samples = self._load_samples()
        results: dict[str, EvaluationSummary] = {}
        for name in self.chunker_names:
            chunker = self._instantiate(name)
            summaries = [self._evaluate_sample(chunker, sample) for sample in samples]
            boundary_scores = [summary["boundary_f1"] for summary in summaries]
            recall_scores = [summary["recall_at_20"] for summary in summaries]
            ndcg10_scores = [summary["ndcg_at_10"] for summary in summaries]
            ndcg20_scores = [summary["ndcg_at_20"] for summary in summaries]
            latencies = [summary["latency"] for summary in summaries]
            chunk_counts = [summary["chunks"] for summary in summaries]
            results[name] = EvaluationSummary(
                chunker=name,
                boundary_f1=self._average(boundary_scores),
                recall_at_20=self._average(recall_scores),
                ndcg_at_10=self._average(ndcg10_scores),
                ndcg_at_20=self._average(ndcg20_scores),
                latency_ms=self._average(latencies) * 1000,
                chunk_count=self._average(chunk_counts),
            )
        return results

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _evaluate_sample(
        self, chunker: BaseChunker, sample: GoldSample
    ) -> Mapping[str, float]:
        document = self._build_document(sample)
        started = perf_counter()
        chunks = chunker.chunk(document, tenant_id=self.tenant_id, granularity="section")
        latency = perf_counter() - started
        predicted_boundaries = [chunk.start_char for chunk in chunks]
        boundary_f1 = self._boundary_f1(sample.boundaries, predicted_boundaries)
        recall, ndcg10, ndcg20 = self._retrieval_metrics(sample.queries, chunks)
        return {
            "boundary_f1": boundary_f1,
            "recall_at_20": recall,
            "ndcg_at_10": ndcg10,
            "ndcg_at_20": ndcg20,
            "latency": latency,
            "chunks": float(len(chunks)),
        }

    def _boundary_f1(self, gold: Sequence[int], predicted: Sequence[int]) -> float:
        if not gold:
            return 1.0 if not predicted else 0.0
        gold_set = set(gold)
        pred_set = set(predicted)
        true_positive = len(gold_set & pred_set)
        precision = true_positive / max(len(pred_set), 1)
        recall = true_positive / max(len(gold_set), 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _retrieval_metrics(
        self, queries: Sequence[GoldQuery], chunks: Sequence[Chunk]
    ) -> tuple[float, float, float]:
        if not queries or not chunks:
            return 0.0, 0.0, 0.0
        recall_scores: list[float] = []
        ndcg10_scores: list[float] = []
        ndcg20_scores: list[float] = []
        for query in queries:
            ranking = self._rank_chunks(query, chunks)
            ideal = [1.0] * min(len(ranking), len(query.keywords))
            gains = [score for _, score in ranking]
            recall_scores.append(self._recall_at_k(gains, len(query.keywords), k=20))
            ndcg10_scores.append(self._ndcg(gains, ideal, k=10))
            ndcg20_scores.append(self._ndcg(gains, ideal, k=20))
        return (
            self._average(recall_scores),
            self._average(ndcg10_scores),
            self._average(ndcg20_scores),
        )

    def _rank_chunks(
        self, query: GoldQuery, chunks: Sequence[Chunk]
    ) -> Sequence[tuple[str, float]]:
        tokens = [token.lower() for token in query.keywords]
        ranking: list[tuple[str, float]] = []
        for chunk in chunks:
            text = chunk.body.lower()
            score = sum(1.0 for token in tokens if token in text)
            ranking.append((chunk.chunk_id, score))
        ranking.sort(key=lambda item: item[1], reverse=True)
        return ranking

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _instantiate(self, name: str) -> BaseChunker:
        config = ChunkerConfig(name=name)
        try:
            return self.registry.create(config, allow_experimental=True)
        except ChunkerConfigurationError:
            if name == "semantic_splitter":
                from Medical_KG_rev.chunking.chunkers.semantic import SemanticSplitterChunker

                class _FallbackEncoder:
                    def encode(self, sentences, convert_to_numpy=True):
                        matrix = np.ones((len(sentences), 8))
                        return matrix

                return SemanticSplitterChunker(encoder=_FallbackEncoder())
            raise

    def _load_samples(self) -> list[GoldSample]:
        samples: list[GoldSample] = []
        for path in sorted(self.gold_path.glob("*.json")):
            payload = json.loads(path.read_text())
            for item in payload:
                queries = [
                    GoldQuery(text=query["text"], keywords=query.get("keywords", []))
                    for query in item.get("queries", [])
                ]
                samples.append(
                    GoldSample(
                        doc_id=item["doc_id"],
                        source=item.get("source", "pmc"),
                        title=item.get("title", item["doc_id"]),
                        text=item["text"],
                        boundaries=item.get("boundaries", []),
                        queries=queries,
                    )
                )
        return samples

    def _build_document(self, sample: GoldSample) -> Document:
        paragraphs = [segment.strip() for segment in sample.text.split("\n") if segment.strip()]
        blocks = [
            Block(
                id=f"{sample.doc_id}:block:{index}",
                type=BlockType.PARAGRAPH,
                text=paragraph,
            )
            for index, paragraph in enumerate(paragraphs)
        ]
        section = Section(
            id=f"{sample.doc_id}:section:0",
            title=sample.title,
            blocks=blocks,
        )
        return Document(
            id=sample.doc_id,
            source=sample.source,
            title=sample.title,
            sections=[section],
        )

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def _average(self, values: Iterable[float]) -> float:
        values = [value for value in values if not isinstance(value, bool)]
        return mean(values) if values else 0.0

    def _recall_at_k(self, gains: Sequence[float], relevant: int, *, k: int) -> float:
        if relevant == 0:
            return 0.0
        hits = sum(1.0 for gain in gains[:k] if gain > 0)
        return hits / relevant

    def _ndcg(
        self, gains: Sequence[float], ideal: Sequence[float], *, k: int
    ) -> float:
        def dcg(scores: Sequence[float]) -> float:
            total = 0.0
            for index, score in enumerate(scores[:k], start=1):
                total += (2**score - 1) / self._log2(index + 1)
            return total

        ideal_dcg = dcg(ideal)
        if ideal_dcg == 0:
            return 0.0
        return dcg(gains) / ideal_dcg

    def _log2(self, value: float) -> float:
        import math

        return math.log(value, 2)


def main() -> None:  # pragma: no cover - CLI helper
    chunkers = [
        "semantic_splitter",
        "section_aware",
        "llm_chaptering",
    ]
    runner = ChunkingEvaluationRunner(chunkers)
    summaries = runner.run()
    for name, summary in summaries.items():
        print(f"{name}: F1={summary.boundary_f1:.3f}, Recall@20={summary.recall_at_20:.3f}, "
              f"nDCG@10={summary.ndcg_at_10:.3f}, Latency={summary.latency_ms:.2f}ms")


if __name__ == "__main__":  # pragma: no cover
    main()
