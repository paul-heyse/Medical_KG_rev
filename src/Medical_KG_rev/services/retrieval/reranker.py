"""Cross-encoder reranking with graceful fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, MutableMapping, Sequence, Tuple


@dataclass(slots=True)
class CrossEncoderReranker:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    batch_size: int = 16
    _model: object | None = field(default=None, init=False, repr=False)
    _load_error: str | None = field(default=None, init=False, repr=False)

    def rerank(
        self,
        query: str,
        candidates: Iterable[Mapping[str, object]],
        *,
        text_field: str = "text",
        top_k: int = 10,
    ) -> Tuple[List[Mapping[str, object]], MutableMapping[str, object]]:
        items = [dict(candidate) for candidate in candidates]
        if not items:
            return [], {"model": self.model_name, "evaluated": 0, "applied": False}
        top_k = min(top_k, len(items))
        evaluated = items[:top_k]
        scores = self._predict(query, evaluated, text_field)
        for item, score in zip(evaluated, scores):
            item["rerank_score"] = float(score)
        evaluated.sort(key=lambda entry: entry.get("rerank_score", 0.0), reverse=True)
        remainder = items[top_k:]
        ranked = evaluated + remainder
        metrics: MutableMapping[str, object] = {
            "model": self.model_name if self._model else f"fallback:{self._load_error or 'lexical'}",
            "evaluated": len(evaluated),
            "applied": True,
        }
        return ranked, metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _predict(
        self, query: str, candidates: Sequence[Mapping[str, object]], text_field: str
    ) -> List[float]:
        model = self._ensure_model()
        if model is None:
            return self._lexical_overlap(query, candidates, text_field)
        pairs = [(query, str(candidate.get(text_field, ""))) for candidate in candidates]
        predictions = model.predict(pairs, batch_size=self.batch_size)
        return [float(score) for score in predictions]

    def _lexical_overlap(
        self, query: str, candidates: Sequence[Mapping[str, object]], text_field: str
    ) -> List[float]:
        query_terms = set(query.lower().split())
        scores: List[float] = []
        for candidate in candidates:
            text = str(candidate.get(text_field, ""))
            terms = set(text.lower().split())
            overlap = len(query_terms & terms)
            scores.append(overlap / max(len(query_terms), 1))
        return scores

    def _ensure_model(self):
        if self._model is not None or self._load_error:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.model_name)
        except Exception as exc:  # pragma: no cover - model download not required in tests
            self._load_error = str(exc)
            self._model = None
        return self._model
