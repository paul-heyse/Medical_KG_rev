"""Stage-based embedding worker backed by Dagster components.

This module provides the core embedding service implementation that bridges
between the gateway layer and the orchestration system. It handles embedding
requests, manages stage resolution, and provides both synchronous and
asynchronous (gRPC) interfaces for embedding operations.

The service is built around the Dagster orchestration framework, using
stage factories to resolve and execute embedding stages. It supports
both direct embedding requests and gRPC-based embedding operations.

Key Components:
    - EmbeddingWorker: Main service class for embedding operations
    - EmbeddingRequest/Response: Data models for embedding operations
    - EmbeddingVector: Container for embedding vectors with metadata
    - EmbeddingGrpcService: Async gRPC servicer for remote embedding

Responsibilities:
    - Process embedding requests from gateway services
    - Resolve and execute embedding stages via Dagster
    - Handle chunk processing and vector normalization
    - Provide both sync and async interfaces
    - Manage stage factory resolution and caching

Collaborators:
    - Gateway services (embedding coordinator)
    - Dagster orchestration system
    - Embedding model registry
    - Namespace registry
    - Chunking models

Side Effects:
    - Logs embedding operations and errors
    - Updates stage factory cache
    - Generates correlation IDs for tracking

Thread Safety:
    - Thread-safe: All operations are stateless
    - Stage factory resolution is cached per instance

Performance Characteristics:
    - Embedding operations are CPU/memory intensive
    - Stage resolution is cached to avoid repeated lookups
    - Vector normalization adds computational overhead

Example:
    >>> worker = EmbeddingWorker()
    >>> request = EmbeddingRequest(
    ...     tenant_id="tenant1", texts=["Hello world"], normalize=True
    ... )
    >>> response = worker.run(request)
    >>> print(f"Generated {len(response.vectors)} vectors")

"""

from __future__ import annotations

# ================================================================================
# IMPORTS
# ================================================================================
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog
from Medical_KG_rev.adapters import get_plugin_manager
from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.orchestration.dagster.configuration import StageDefinition
from Medical_KG_rev.orchestration.dagster.runtime import (
    StageFactory,
    StageResolutionError,
    build_stage_factory,
)
from Medical_KG_rev.orchestration.dagster.stages import (
    create_default_pipeline_resource,
    create_stage_plugin_manager,
)
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.stages.contracts import (
    EmbeddingBatch,
    EmbedStage,
    StageContext,
)

from .registry import EmbeddingModelRegistry

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logger = structlog.get_logger(__name__)


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


def _default_stage_factory() -> StageFactory:
    """Instantiate the default stage factory using registered adapters.

    Returns:
        Configured StageFactory instance with default pipeline resource
        and job ledger.

    Note:
        This function creates a stage factory using the default pipeline
        resource and job ledger. It's used as a fallback when no custom
        stage factory is provided.

    Example:
        >>> factory = _default_stage_factory()
        >>> stage = factory.resolve("pipeline", StageDefinition(name="embed", type="embed"))

    """
    adapter_manager = get_plugin_manager()
    pipeline_resource = create_default_pipeline_resource()
    job_ledger = JobLedger()
    return build_stage_factory(adapter_manager, pipeline_resource, job_ledger)
    plugin_manager = create_stage_plugin_manager(get_plugin_manager())
    return StageFactory(plugin_manager)


# ==============================================================================
# DATA MODELS
# ==============================================================================


@dataclass(slots=True)
class EmbeddingRequest:
    """Payload accepted by :class:`EmbeddingWorker`.

    Represents a request to generate embeddings for a collection of texts.
    Supports various options including normalization, model selection, and
    metadata association.

    Attributes:
        tenant_id: Identifier for the tenant making the request
        texts: Sequence of texts to embed
        chunk_ids: Optional sequence of chunk IDs corresponding to texts
        normalize: Whether to normalize the resulting vectors
        model: Optional model name to use for embedding
        metadata: Optional sequence of metadata mappings for each text
        correlation_id: Optional correlation ID for request tracking
        pipeline_name: Optional pipeline name for execution context
        pipeline_version: Optional pipeline version for execution context

    Invariants:
        - tenant_id is never empty
        - texts is never empty when processing
        - chunk_ids length matches texts length when provided
        - metadata length matches texts length when provided

    Example:
        >>> request = EmbeddingRequest(
        ...     tenant_id="tenant1",
        ...     texts=["Hello world", "Goodbye world"],
        ...     normalize=True,
        ...     model="text-embedding-ada-002"
        ... )

    """

    tenant_id: str
    texts: Sequence[str]
    chunk_ids: Sequence[str] | None = None
    normalize: bool = False
    model: str | None = None
    metadata: Sequence[Mapping[str, Any]] | None = None
    correlation_id: str | None = None
    pipeline_name: str | None = None
    pipeline_version: str | None = None


@dataclass(slots=True)
class EmbeddingVector:
    """Embedding vector returned to gateway and gRPC callers.

    Represents a single embedding vector with associated metadata and
    identification information. Used for both internal processing and
    external API responses.

    Attributes:
        id: Unique identifier for the vector
        model: Name of the model that generated the vector
        kind: Type of vector (e.g., "dense", "sparse")
        values: Tuple of floating-point values representing the vector
        metadata: Dictionary of additional metadata

    Invariants:
        - id is never empty
        - model is never empty
        - values tuple is immutable
        - metadata is mutable but thread-safe for reads

    Example:
        >>> vector = EmbeddingVector(
        ...     id="chunk_123",
        ...     model="text-embedding-ada-002",
        ...     values=(0.1, 0.2, 0.3),
        ...     metadata={"source": "document.pdf"}
        ... )
        >>> print(f"Dimension: {vector.dimension}")

    """

    id: str
    model: str
    kind: str = "dense"
    values: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vector.

        Returns:
            Number of dimensions in the vector.

        Example:
            >>> vector = EmbeddingVector(id="test", model="test", values=(1.0, 2.0, 3.0))
            >>> print(vector.dimension)  # 3

        """
        return len(self.values)


@dataclass(slots=True)
class EmbeddingResponse:
    """Container produced by :class:`EmbeddingWorker`.

    Represents the response from an embedding operation, containing
    the generated vectors and any associated metadata.

    Attributes:
        vectors: List of generated embedding vectors

    Invariants:
        - vectors list is never None
        - All vectors in the list are valid EmbeddingVector instances

    Example:
        >>> response = EmbeddingResponse()
        >>> response.vectors.append(EmbeddingVector(
        ...     id="chunk_1", model="test", values=(1.0, 2.0)
        ... ))
        >>> print(f"Generated {len(response.vectors)} vectors")

    """

    vectors: list[EmbeddingVector] = field(default_factory=list)


# ==============================================================================
# SERVICE IMPLEMENTATION
# ==============================================================================


class EmbeddingWorker:
    """Thin wrapper that executes the embed stage via Dagster components.

    Main service class for processing embedding requests. Handles stage
    resolution, chunk processing, and vector generation through the
    Dagster orchestration system.

    Attributes:
        _stage_factory: Factory for resolving and creating stages
        _embed_stage: Cached embedding stage instance
        _stage_definition: Definition for the embedding stage
        _pipeline_name: Name of the pipeline for execution context
        _pipeline_version: Version of the pipeline for execution context

    Invariants:
        - _stage_factory is never None
        - _stage_definition is never None
        - _pipeline_name and _pipeline_version are never empty

    Thread Safety:
        - Thread-safe: All operations are stateless
        - Stage resolution is cached per instance

    Lifecycle:
        - Initialized with optional dependencies
        - Stage factory is resolved on first use
        - Embedding stage is cached after first resolution

    Example:
        >>> worker = EmbeddingWorker()
        >>> request = EmbeddingRequest(
        ...     tenant_id="tenant1", texts=["Hello world"]
        ... )
        >>> response = worker.run(request)
        >>> print(f"Generated {len(response.vectors)} vectors")

    """

    def __init__(
        self,
        registry: EmbeddingModelRegistry | None = None,
        *,
        stage_factory: StageFactory | None = None,
        embed_stage: EmbedStage | None = None,
        stage_definition: StageDefinition | None = None,
        pipeline_name: str = "gateway-direct",
        pipeline_version: str = "v1",
    ) -> None:
        """Initialize embedding worker with optional dependencies.

        Args:
            registry: Optional model registry (retained for compatibility)
            stage_factory: Optional stage factory for stage resolution
            embed_stage: Optional pre-configured embedding stage
            stage_definition: Optional stage definition for resolution
            pipeline_name: Name of the pipeline for execution context
            pipeline_version: Version of the pipeline for execution context

        Note:
            If no stage_factory is provided, a default one is created.
            The embed_stage is cached after first resolution to avoid
            repeated lookups.

        """
        self._stage_factory = stage_factory or _default_stage_factory()
        self._embed_stage = embed_stage
        self._stage_definition = stage_definition or StageDefinition(name="embed", type="embed")
        self._pipeline_name = pipeline_name
        self._pipeline_version = pipeline_version

    def run(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Execute the embed stage for the provided request.

        Args:
            request: Embedding request containing texts and configuration

        Returns:
            Response containing generated embedding vectors

        Raises:
            Exception: If stage execution fails or other errors occur

        Note:
            This method processes the request by cleaning texts, building
            chunks, resolving the embedding stage, and executing it. Empty
            requests return an empty response.

        Example:
            >>> worker = EmbeddingWorker()
            >>> request = EmbeddingRequest(
            ...     tenant_id="tenant1", texts=["Hello world"], normalize=True
            ... )
            >>> response = worker.run(request)
            >>> print(f"Generated {len(response.vectors)} vectors")

        """
        cleaned_texts = [
            text.strip() for text in request.texts if isinstance(text, str) and text.strip()
        ]
        if not cleaned_texts:
            logger.warning(
                "embedding.worker.empty_payload",
                tenant_id=request.tenant_id,
            )
            return EmbeddingResponse()

        chunks = self._build_chunks(
            tenant_id=request.tenant_id,
            texts=cleaned_texts,
            chunk_ids=request.chunk_ids,
            metadata=request.metadata,
        )
        context = StageContext(
            tenant_id=request.tenant_id,
            doc_id=f"embed:{uuid.uuid4().hex[:12]}",
            correlation_id=request.correlation_id or uuid.uuid4().hex,
            metadata={
                "model": request.model,
                "normalize": request.normalize,
            },
            pipeline_name=request.pipeline_name or self._pipeline_name,
            pipeline_version=request.pipeline_version or self._pipeline_version,
        )

        stage = self._resolve_stage()
        try:
            batch = stage.execute(context, chunks)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception(
                "embedding.worker.stage_error",
                tenant_id=request.tenant_id,
                error=str(exc),
            )
            raise
        return self._response_from_batch(batch, normalize=request.normalize)

    def encode_queries(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Alias for :meth:`run` retained for compatibility with rerankers.

        Args:
            request: Embedding request containing texts and configuration

        Returns:
            Response containing generated embedding vectors

        Note:
            This method is maintained for backward compatibility with
            reranking systems that expect an encode_queries method.

        Example:
            >>> worker = EmbeddingWorker()
            >>> request = EmbeddingRequest(
            ...     tenant_id="tenant1", texts=["query text"]
            ... )
            >>> response = worker.encode_queries(request)
            >>> print(f"Encoded {len(response.vectors)} queries")

        """
        return self.run(request)

    def _resolve_stage(self) -> EmbedStage:
        """Resolve and cache the embedding stage.

        Returns:
            Configured embedding stage instance

        Raises:
            StageResolutionError: If stage cannot be resolved
            TypeError: If resolved stage doesn't implement EmbedStage

        Note:
            This method caches the resolved stage to avoid repeated
            resolution. If initial resolution fails, it retries with
            a generic stage definition.

        Example:
            >>> worker = EmbeddingWorker()
            >>> stage = worker._resolve_stage()
            >>> print(f"Stage type: {type(stage).__name__}")

        """
        if self._embed_stage is not None:
            return self._embed_stage
        try:
            stage = self._stage_factory.resolve(
                self._pipeline_name,
                self._stage_definition,
            )
        except StageResolutionError:
            # Retry with a generic definition that omits name, matching configuration defaults.
            stage = self._stage_factory.resolve(
                self._pipeline_name,
                StageDefinition(name="embed", type="embed"),
            )
        if not isinstance(stage, EmbedStage):  # pragma: no cover - defensive
            raise TypeError("Resolved stage does not implement EmbedStage")
        self._embed_stage = stage
        return stage

    def _build_chunks(
        self,
        *,
        tenant_id: str,
        texts: Sequence[str],
        chunk_ids: Sequence[str] | None,
        metadata: Sequence[Mapping[str, Any]] | None,
    ) -> list[Chunk]:
        """Build chunk objects from texts and metadata.

        Args:
            tenant_id: Identifier for the tenant
            texts: Sequence of texts to convert to chunks
            chunk_ids: Optional sequence of chunk IDs
            metadata: Optional sequence of metadata mappings

        Returns:
            List of Chunk objects with proper IDs and metadata

        Note:
            This method generates chunk IDs if not provided and ensures
            metadata is properly aligned with texts. Default values are
            used for missing metadata fields.

        Example:
            >>> worker = EmbeddingWorker()
            >>> chunks = worker._build_chunks(
            ...     tenant_id="tenant1",
            ...     texts=["Hello", "World"],
            ...     chunk_ids=["chunk1", "chunk2"],
            ...     metadata=[{"source": "doc1"}, {"source": "doc2"}]
            ... )
            >>> print(f"Built {len(chunks)} chunks")

        """
        ids = list(chunk_ids or [])
        if len(ids) < len(texts):
            ids.extend(f"{tenant_id}:{index}" for index in range(len(ids), len(texts)))
        else:
            ids = ids[: len(texts)]

        meta_sequence = [dict(item) for item in (metadata or [])][: len(texts)]
        while len(meta_sequence) < len(texts):
            meta_sequence.append({})

        chunks: list[Chunk] = []
        for index, (text, chunk_id, meta) in enumerate(zip(texts, ids, meta_sequence, strict=True)):
            doc_id = meta.get("doc_id") or f"{chunk_id}:doc"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=str(doc_id),
                    tenant_id=tenant_id,
                    body=text,
                    title_path=tuple(meta.get("title_path", ())),
                    section=meta.get("section"),
                    start_char=int(meta.get("start", 0)),
                    end_char=int(meta.get("end", len(text))),
                    granularity=str(meta.get("granularity") or "document"),
                    chunker=str(meta.get("chunker") or "embedding.worker"),
                    chunker_version=str(meta.get("chunker_version") or "1.0.0"),
                    meta={"input_index": index, **meta},
                )
            )
        return chunks

    def _response_from_batch(
        self,
        batch: EmbeddingBatch,
        *,
        normalize: bool,
    ) -> EmbeddingResponse:
        """Convert embedding batch to response format.

        Args:
            batch: Embedding batch from stage execution
            normalize: Whether to normalize vector values

        Returns:
            Response containing converted embedding vectors

        Note:
            This method converts the internal batch format to the external
            response format, optionally normalizing vectors and extracting
            metadata.

        Example:
            >>> worker = EmbeddingWorker()
            >>> # Assuming batch is an EmbeddingBatch instance
            >>> response = worker._response_from_batch(batch, normalize=True)
            >>> print(f"Response contains {len(response.vectors)} vectors")

        """
        vectors: list[EmbeddingVector] = []
        for vector in batch.vectors:
            values = list(vector.values)
            if normalize and values:
                magnitude = math.sqrt(sum(value * value for value in values))
                if magnitude > 0:
                    values = [value / magnitude for value in values]
            metadata = dict(vector.metadata)
            kind = str(metadata.pop("vector_kind", metadata.pop("kind", "dense")))
            vectors.append(
                EmbeddingVector(
                    id=vector.id,
                    model=batch.model,
                    kind=kind,
                    values=tuple(values),
                    metadata=metadata,
                )
            )
        return EmbeddingResponse(vectors=vectors)


# ==============================================================================
# GRPC SERVICE IMPLEMENTATION
# ==============================================================================


@dataclass(slots=True)
class EmbeddingGrpcService:
    """Async gRPC servicer bridging requests into the embedding worker.

    Provides an asynchronous gRPC interface for embedding operations,
    converting gRPC requests to internal format and responses back to
    gRPC format.

    Attributes:
        worker: Embedding worker instance for processing requests

    Invariants:
        - worker is never None
        - All gRPC requests are properly converted to internal format

    Thread Safety:
        - Thread-safe: Uses async/await for concurrency
        - Worker operations are stateless

    Example:
        >>> worker = EmbeddingWorker()
        >>> grpc_service = EmbeddingGrpcService(worker)
        >>> # Used by gRPC server for handling requests

    """

    worker: EmbeddingWorker

    async def EmbedChunks(self, request, context):  # type: ignore[override]
        """Process embedding request via gRPC interface.

        Args:
            request: gRPC request containing embedding parameters
            context: gRPC context for request handling

        Returns:
            gRPC response containing embedding vectors

        Note:
            This method converts the gRPC request to internal format,
            processes it through the embedding worker, and converts
            the response back to gRPC format.

        Example:
            >>> # Called by gRPC server when handling EmbedChunks requests
            >>> response = await grpc_service.EmbedChunks(request, context)
            >>> print(f"Generated {len(response.vectors)} vectors")

        """
        embed_request = EmbeddingRequest(
            tenant_id=request.tenant_id,
            chunk_ids=list(request.chunk_ids) or None,
            texts=list(request.texts),
            normalize=request.normalize,
            model=request.models[0] if request.models else None,
            correlation_id=getattr(request, "correlation_id", None),
        )
        response = self.worker.run(embed_request)

        from Medical_KG_rev.proto.gen import embedding_pb2  # type: ignore import-error

        reply = embedding_pb2.EmbedChunksResponse()
        for vector in response.vectors:
            message = reply.vectors.add()
            message.chunk_id = vector.id
            message.model = vector.model
            message.kind = vector.kind
            message.dimension = vector.dimension
            message.values.extend(vector.values)
        return reply


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "EmbeddingGrpcService",
    "EmbeddingModelRegistry",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingVector",
    "EmbeddingWorker",
]
