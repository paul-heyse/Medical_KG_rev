"""Two stage retrieval pipeline integrating fusion and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import structlog

from Medical_KG_rev.auth.context import SecurityContext

from ..models import PipelineSettings, ScoredDocument
from ..rerank_engine import RerankingEngine
from ..fusion.service import FusionService

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class TwoStagePipeline:
    """Coordinates retrieval → fusion → reranking."""

    fusion: FusionService
    reranking: RerankingEngine
    settings: PipelineSettings

    def execute(
        self,
        context: SecurityContext,
        query: str,
        candidate_lists: Mapping[str, Sequence[ScoredDocument]],
        *,
        reranker_id: str | None,
        top_k: int,
        rerank: bool,
    ) -> tuple[list[ScoredDocument], Mapping[str, object]]:
        logger.debug(
            "pipeline.two_stage.start",
            tenant=context.tenant_id,
            rerank=rerank,
            reranker=reranker_id,
        )
        fused = self.fusion.fuse(candidate_lists)
        documents = list(fused.documents)
        for document in documents:
            document.metadata.setdefault("retrieval_score", document.score)
        metrics: dict[str, object] = {"fusion": fused.metrics}
        if not rerank or not documents:
            return documents[:top_k], metrics

        rerank_candidates = documents[: self.settings.rerank_candidates]
        response = self.reranking.rerank(
            context=context,
            query=query,
            documents=rerank_candidates,
            reranker_id=reranker_id,
            top_k=self.settings.return_top_k,
        )
        score_map = {item.doc_id: item.score for item in response.results}
        for document in rerank_candidates:
            if document.doc_id in score_map:
                document.score = score_map[document.doc_id]
        rerank_candidates.sort(key=lambda doc: doc.score, reverse=True)
        metrics["reranking"] = response.metrics
        return rerank_candidates[:top_k], metrics
