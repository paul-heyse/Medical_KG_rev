"""Ingestion and retrieval orchestration primitives."""

from .kafka import KafkaClient, KafkaMessage
from .ledger import JobLedger, JobLedgerEntry, JobTransition
from .dagster import (
    DagsterOrchestrator,
    DagsterRunResult,
    PipelineConfigLoader,
    PipelineTopologyConfig,
    ResiliencePolicy,
    ResiliencePolicyConfig,
    ResiliencePolicyLoader,
    StageFactory,
    StageResolutionError,
    submit_to_dagster,
)
from .state_manager import LedgerStateManager
from .haystack import (
    HaystackChunker,
    HaystackEmbedder,
    HaystackIndexWriter,
    HaystackRetriever,
    HaystackSparseExpander,
)
from .stages import StageFailure
from .stages.contracts import (
    ChunkStage,
    EmbedStage,
    EmbeddingBatch,
    EmbeddingVector,
    ExtractStage,
    GraphWriteReceipt,
    IngestStage,
    IndexReceipt,
    IndexStage,
    KGStage,
    ParseStage,
    StageContext,
)
__all__ = [
    "ChunkStage",
    "DagsterOrchestrator",
    "DagsterRunResult",
    "EmbedStage",
    "EmbeddingBatch",
    "EmbeddingVector",
    "GraphWriteReceipt",
    "HaystackChunker",
    "HaystackEmbedder",
    "HaystackIndexWriter",
    "HaystackRetriever",
    "HaystackSparseExpander",
    "IngestStage",
    "IndexReceipt",
    "IndexStage",
    "JobLedger",
    "JobLedgerEntry",
    "JobTransition",
    "LedgerStateManager",
    "KGStage",
    "KafkaClient",
    "KafkaMessage",
    "ParseStage",
    "PipelineConfigLoader",
    "PipelineTopologyConfig",
    "ResiliencePolicy",
    "ResiliencePolicyConfig",
    "ResiliencePolicyLoader",
    "StageContext",
    "StageFactory",
    "StageFailure",
    "StageResolutionError",
    "submit_to_dagster",
]
