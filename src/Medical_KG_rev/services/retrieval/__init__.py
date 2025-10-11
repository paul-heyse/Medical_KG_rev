"""Lazy exports for retrieval services to minimise optional dependencies."""

from __future__ import annotations

from typing import Any

from importlib import import_module


__all__ = [
    "Chunk",
    "ChunkCommand",
    "ChunkingOptions",
    "ChunkingService",
    "CrossEncoderReranker",
    "DocumentIndexTemplate",
    "FAISSIndex",
    "HybridComponentSettings",
    "HybridSearchCoordinator",
    "HybridSearchResult",
    "OpenSearchClient",
    "QueryDSL",
    "QueryValidationError",
    "RetrievalResult",
    "RetrievalService",
]

_RETRIEVAL_MAP = {
    "Chunk": ("Medical_KG_rev.services.retrieval.chunking", "Chunk"),
    "ChunkCommand": (
        "Medical_KG_rev.services.retrieval.chunking_command",
        "ChunkCommand",
    ),
    "ChunkingOptions": ("Medical_KG_rev.services.retrieval.chunking", "ChunkingOptions"),
    "ChunkingService": ("Medical_KG_rev.services.retrieval.chunking", "ChunkingService"),
    "CrossEncoderReranker": (
        "Medical_KG_rev.services.retrieval.reranker",
        "CrossEncoderReranker",
    ),
    "DocumentIndexTemplate": (
        "Medical_KG_rev.services.retrieval.opensearch_client",
        "DocumentIndexTemplate",
    ),
    "HybridComponentSettings": (
        "Medical_KG_rev.services.retrieval.hybrid",
        "HybridComponentSettings",
    ),
    "HybridSearchCoordinator": (
        "Medical_KG_rev.services.retrieval.hybrid",
        "HybridSearchCoordinator",
    ),
    "HybridSearchResult": (
        "Medical_KG_rev.services.retrieval.hybrid",
        "HybridSearchResult",
    ),
    "FAISSIndex": ("Medical_KG_rev.services.retrieval.faiss_index", "FAISSIndex"),
    "OpenSearchClient": (
        "Medical_KG_rev.services.retrieval.opensearch_client",
        "OpenSearchClient",
    ),
    "QueryDSL": ("Medical_KG_rev.services.retrieval.query_dsl", "QueryDSL"),
    "QueryValidationError": (
        "Medical_KG_rev.services.retrieval.query_dsl",
        "QueryValidationError",
    ),
    "RetrievalResult": (
        "Medical_KG_rev.services.retrieval.retrieval_service",
        "RetrievalResult",
    ),
    "RetrievalService": (
        "Medical_KG_rev.services.retrieval.retrieval_service",
        "RetrievalService",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_path, attribute = _RETRIEVAL_MAP[name]
    except KeyError as exc:  # pragma: no cover
        raise AttributeError(name) from exc
    module = import_module(module_path)
    return getattr(module, attribute)


def __dir__() -> list[str]:  # pragma: no cover - tooling aid
    return sorted(__all__)
