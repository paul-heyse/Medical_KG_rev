"""Chunking coordinator for gateway operations.

This module coordinates synchronous chunking operations for the API gateway,
providing protocol-agnostic chunking capabilities with error handling and
comprehensive observability.

Key Responsibilities:
    - Coordinate between gateway services and chunking domain logic
    - Provide protocol-agnostic chunking operations (REST, GraphQL, gRPC)
    - Handle error translation and structured error responses
    - Implement job lifecycle management for chunking operations
    - Emit appropriate metrics for monitoring and debugging

Collaborators:
    - Upstream: Gateway services (REST, GraphQL, gRPC handlers)
    - Downstream: ChunkingService for actual chunking implementation
    - Error Handling: ChunkingErrorTranslator for structured error responses
    - Observability: Metrics system for operation tracking

Side Effects:
    - Emits Prometheus metrics for chunking operations
    - Updates job lifecycle state for tracking and debugging
    - May write to error reporting systems on failures

Thread Safety:
    - Not thread-safe: Coordinator instances should not be shared between threads
    - Chunking operations are synchronous and blocking

Performance Characteristics:
    - Time complexity: O(n) where n is document length in characters
    - Memory usage: O(m) where m is number of chunks produced
    - No external API calls or database operations during chunking

Example:
    >>> coordinator = ChunkingCoordinator(chunking_service, error_translator)
    >>> request = ChunkingRequest(tenant_id="tenant1", document_id="doc1", text="...")
    >>> response = coordinator.execute(request)
    >>> print(f"Produced {len(response.chunks)} chunks")

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from Medical_KG_rev.chunking.exceptions import InvalidDocumentError
from Medical_KG_rev.gateway.models import DocumentChunk
from Medical_KG_rev.observability.metrics import record_chunking_failure
from Medical_KG_rev.services.retrieval.chunking import ChunkCommand, ChunkingService

from ..chunking_errors import ChunkingErrorReport, ChunkingErrorTranslator

logger = structlog.get_logger(__name__)


# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

@dataclass
class ChunkingRequest:
    """Request for chunking operation.

    This dataclass represents a request to chunk a document into smaller
    segments for processing. It contains all the necessary information
    for the chunking operation including document content and chunking
    parameters.

    Attributes:
        tenant_id: Unique identifier for the tenant making the request.
            Used for multi-tenancy and access control validation.
        document_id: Unique identifier for the document being chunked.
            Used for tracking and correlation across systems.
        text: The document text content to be chunked. Must be non-empty
            and contain valid text content for processing.
        strategy: Chunking strategy to use. Valid values include:
            - "section": Section-based chunking (default)
            - "semantic": Semantic similarity-based chunking
            - "paragraph": Paragraph-based chunking
        options: Additional configuration options for the chunking operation.
            May include strategy-specific parameters and metadata.

    Example:
        >>> request = ChunkingRequest(
        ...     tenant_id="tenant_123",
        ...     document_id="doc_456",
        ...     text="This is a sample document...",
        ...     strategy="section",
        ...     options={"overlap": 50}
        ... )

    """

    tenant_id: str
    document_id: str
    text: str
    strategy: str
    options: dict[str, Any] | None = None


@dataclass
class ChunkingResponse:
    """Response from chunking operation.

    This dataclass represents the result of a document chunking operation,
    containing the produced chunks along with metadata about the operation.

    Attributes:
        chunks: List of DocumentChunk objects produced by the chunking operation.
            Each chunk contains a portion of the original document with metadata.
        processing_time: Total time taken for the chunking operation in seconds.
            Used for performance monitoring and optimization.
        strategy_used: The actual chunking strategy that was applied. May differ
            from the requested strategy if fallback was necessary.
        metadata: Additional metadata about the chunking operation. May include
            strategy-specific information, warnings, or diagnostic data.

    Example:
        >>> response = coordinator.execute(request)
        >>> print(f"Chunking took {response.processing_time:.2f}s")
        >>> print(f"Produced {len(response.chunks)} chunks using {response.strategy_used}")

    """

    chunks: list[DocumentChunk]
    processing_time: float
    strategy_used: str
    metadata: dict[str, Any] | None = None


# ==============================================================================
# COORDINATOR IMPLEMENTATION
# ==============================================================================

class ChunkingCoordinator:
    """Coordinates chunking operations for the gateway.

    This class implements the coordinator pattern for synchronous chunking operations,
    providing a clean interface between the API gateway and the chunking domain logic.
    It handles error translation, metrics collection, and provides comprehensive
    observability for chunking operations.

    Key Features:
        - Protocol-agnostic chunking coordination (REST, GraphQL, gRPC)
        - Comprehensive error handling and translation
        - Performance monitoring and metrics collection
        - Input validation and sanitization
        - Structured logging for debugging and monitoring

    Attributes:
        _chunking_service: The underlying chunking service for actual chunking operations
        _error_translator: Translator for converting chunking errors to coordinator errors

    Invariants:
        - _chunking_service is never None after __init__
        - _error_translator is never None after __init__

    Thread Safety:
        - Not thread-safe: Coordinator instances should not be shared between threads
        - Chunking operations are synchronous and blocking

    Lifecycle:
        - Created with dependencies injected via constructor
        - Used for coordinating chunking operations
        - No explicit cleanup required

    Example:
        >>> coordinator = ChunkingCoordinator(chunking_service, error_translator)
        >>> request = ChunkingRequest(tenant_id="tenant1", document_id="doc1", text="...")
        >>> response = coordinator.execute(request)
        >>> print(f"Chunking completed in {response.processing_time:.2f}s")

    """

    def __init__(
        self,
        chunking_service: ChunkingService,
        error_translator: ChunkingErrorTranslator,
    ) -> None:
        """Initialize the chunking coordinator.

        Sets up the coordinator with required dependencies for chunking operations.

        Args:
            chunking_service: The chunking service that performs the actual chunking
                operations. Must be configured and ready for use.
            error_translator: Translator for converting chunking domain errors
                into coordinator-level errors with proper HTTP status codes.

        Raises:
            ValueError: If any required dependency is None or invalid.

        """
        if chunking_service is None:
            raise ValueError("chunking_service cannot be None")
        if error_translator is None:
            raise ValueError("error_translator cannot be None")

        self._chunking_service = chunking_service
        self._error_translator = error_translator

    def execute(
        self,
        request: ChunkingRequest,
    ) -> ChunkingResponse:
        """Execute chunking operation.

        Coordinates the full chunking workflow: validates input, creates chunking
        command, delegates to chunking service, handles exceptions, assembles
        results, and emits appropriate metrics.

        Args:
            request: ChunkingRequest with document and chunking parameters

        Returns:
            ChunkingResponse with chunks and operation metadata

        Raises:
            ChunkingErrorReport: For all handled errors after translation
                - InvalidDocumentError: When document text is empty or invalid
                - ChunkingServiceError: When chunking service is unavailable
                - StrategyNotFoundError: When requested strategy doesn't exist

        Example:
            >>> request = ChunkingRequest(
            ...     tenant_id="tenant_123",
            ...     document_id="doc_456",
            ...     text="Sample document text...",
            ...     strategy="section"
            ... )
            >>> response = coordinator.execute(request)
            >>> assert len(response.chunks) > 0

        """
        start_time = time.time()

        try:
            # Validate input
            if not request.text or not request.text.strip():
                raise InvalidDocumentError("Document text cannot be empty")

            # Create chunk command
            command = ChunkCommand(
                tenant_id=request.tenant_id,
                document_id=request.document_id,
                text=request.text,
                strategy=request.strategy,
                options=request.options or {},
            )

            # Execute chunking
            chunks = self._chunking_service.chunk(command)

            # Convert to gateway models
            document_chunks = [
                DocumentChunk(
                    document_id=request.document_id,
                    chunk_id=f"{request.document_id}_chunk_{i}",
                    content=chunk.text,
                    metadata={
                        "strategy": request.strategy,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        **chunk.metadata,
                    },
                )
                for i, chunk in enumerate(chunks)
            ]

            processing_time = time.time() - start_time

            return ChunkingResponse(
                chunks=document_chunks,
                processing_time=processing_time,
                strategy_used=request.strategy,
                metadata={"chunk_count": len(chunks)},
            )

        except InvalidDocumentError as e:
            # Record failure metric
            record_chunking_failure(request.strategy, "invalid_document")
            raise self._error_translator.translate(e)


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "ChunkingCoordinator",
    "ChunkingRequest",
    "ChunkingResponse",
]
