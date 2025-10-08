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
from .haystack import (
    HaystackChunker,
    HaystackEmbedder,
    HaystackIndexWriter,
    HaystackRetriever,
    HaystackSparseExpander,
)
from .stages import StageFailure, StageRegistry
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
from .profiles import PipelineProfile, ProfileDetector, ProfileManager
from .stages import StageRegistry
from .worker import (
    ChunkingWorker,
    EmbeddingPipelineWorker,
    IndexingWorker,
    IngestWorker,
    MappingWorker,
    PipelineWorkerBase,
    RetryPolicy,
    WorkerBase,
)

__all__ = [
    "ChunkingWorker",
    "EmbeddingPipelineWorker",
    "IngestWorker",
    "IndexingWorker",
    "JobLedger",
    "JobLedgerEntry",
    "JobTransition",
    "KGStage",
    "KafkaClient",
    "KafkaMessage",
    "MappingWorker",
    "ParallelExecutor",
    "PipelineConfig",
    "PipelineContext",
    "PipelineDefinition",
    "PipelineExecutor",
    "PipelineProfile",
    "PipelineConfigManager",
    "PipelineWorkerBase",
    "ProfileDefinition",
    "ProfileDetector",
    "ProfileManager",
    "StageRegistry",
    "OrchestrationError",
    "Orchestrator",
    "RetryPolicy",
    "WorkerBase",
]
