"""Skeleton for Differentiable Search Index style research experiments."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class DSISearcher:
    config: EmbedderConfig
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        self.name = self.config.name
        self.kind = self.config.kind
        logger.warning(
            "embedding.experimental.dsi",
            message="DSI searcher is experimental and not production ready",
        )

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        logger.info(
            "embedding.experimental.dsi.encode_documents",
            namespace=request.namespace,
            tenant_id=request.tenant_id,
        )
        return []

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        logger.info(
            "embedding.experimental.dsi.encode_queries",
            namespace=request.namespace,
            tenant_id=request.tenant_id,
        )
        return []


def register_dsi(registry: EmbedderRegistry) -> None:
    registry.register("dsi", lambda config: DSISearcher(config=config))
