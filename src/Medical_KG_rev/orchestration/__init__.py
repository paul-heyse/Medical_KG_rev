"""Ingestion and retrieval orchestration primitives."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ATTRIBUTE_MAP: dict[str, tuple[str, str]] = {
    "KafkaClient": ("Medical_KG_rev.orchestration.kafka", "KafkaClient"),
    "KafkaMessage": ("Medical_KG_rev.orchestration.kafka", "KafkaMessage"),
    "JobLedger": ("Medical_KG_rev.orchestration.ledger", "JobLedger"),
    "JobLedgerEntry": ("Medical_KG_rev.orchestration.ledger", "JobLedgerEntry"),
    "JobTransition": ("Medical_KG_rev.orchestration.ledger", "JobTransition"),
    "DagsterOrchestrator": ("Medical_KG_rev.orchestration.dagster", "DagsterOrchestrator"),
    "DagsterRunResult": ("Medical_KG_rev.orchestration.dagster", "DagsterRunResult"),
    "PipelineConfigLoader": ("Medical_KG_rev.orchestration.dagster", "PipelineConfigLoader"),
    "PipelineTopologyConfig": ("Medical_KG_rev.orchestration.dagster", "PipelineTopologyConfig"),
    "ResiliencePolicy": ("Medical_KG_rev.orchestration.dagster", "ResiliencePolicy"),
    "ResiliencePolicyConfig": ("Medical_KG_rev.orchestration.dagster", "ResiliencePolicyConfig"),
    "ResiliencePolicyLoader": ("Medical_KG_rev.orchestration.dagster", "ResiliencePolicyLoader"),
    "StageFactory": ("Medical_KG_rev.orchestration.dagster", "StageFactory"),
    "StageResolutionError": ("Medical_KG_rev.orchestration.dagster", "StageResolutionError"),
    "submit_to_dagster": ("Medical_KG_rev.orchestration.dagster", "submit_to_dagster"),
    "HaystackChunker": ("Medical_KG_rev.orchestration.haystack", "HaystackChunker"),
    "HaystackEmbedder": ("Medical_KG_rev.orchestration.haystack", "HaystackEmbedder"),
    "HaystackIndexWriter": ("Medical_KG_rev.orchestration.haystack", "HaystackIndexWriter"),
    "HaystackRetriever": ("Medical_KG_rev.orchestration.haystack", "HaystackRetriever"),
    "HaystackSparseExpander": ("Medical_KG_rev.orchestration.haystack", "HaystackSparseExpander"),
    "StageFailure": ("Medical_KG_rev.orchestration.stages", "StageFailure"),
    "ChunkStage": ("Medical_KG_rev.orchestration.stages.contracts", "ChunkStage"),
    "EmbedStage": ("Medical_KG_rev.orchestration.stages.contracts", "EmbedStage"),
    "EmbeddingBatch": ("Medical_KG_rev.orchestration.stages.contracts", "EmbeddingBatch"),
    "EmbeddingVector": ("Medical_KG_rev.orchestration.stages.contracts", "EmbeddingVector"),
    "ExtractStage": ("Medical_KG_rev.orchestration.stages.contracts", "ExtractStage"),
    "GraphWriteReceipt": ("Medical_KG_rev.orchestration.stages.contracts", "GraphWriteReceipt"),
    "IngestStage": ("Medical_KG_rev.orchestration.stages.contracts", "IngestStage"),
    "IndexReceipt": ("Medical_KG_rev.orchestration.stages.contracts", "IndexReceipt"),
    "IndexStage": ("Medical_KG_rev.orchestration.stages.contracts", "IndexStage"),
    "KGStage": ("Medical_KG_rev.orchestration.stages.contracts", "KGStage"),
    "ParseStage": ("Medical_KG_rev.orchestration.stages.contracts", "ParseStage"),
    "StageContext": ("Medical_KG_rev.orchestration.stages.contracts", "StageContext"),
}

__all__ = sorted(_ATTRIBUTE_MAP)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _ATTRIBUTE_MAP[name]
    except KeyError as exc:  # pragma: no cover - standard attribute error path
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - convenience helper
    return sorted(globals().keys() | _ATTRIBUTE_MAP.keys())
