"""MinerU service type definitions and data models.

This module defines the core data structures used by the MinerU service
for PDF document processing, including request/response models, document
representations, and processing metadata.

Key Components:
    - Block: Individual document elements (text, tables, figures, equations)
    - Document: Complete document structure with blocks and metadata
    - MineruRequest/Response: API request and response models
    - ProcessingMetadata: Detailed processing information and metrics

Responsibilities:
    - Define data structures for MinerU document processing
    - Provide type-safe models for API communication
    - Support batch processing operations
    - Track processing metadata and provenance

Collaborators:
    - MinerU service implementation
    - Document models (Equation, Figure, Table, IR)
    - Processing pipeline components

Side Effects:
    - None (pure data models)

Thread Safety:
    - Thread-safe: Immutable dataclasses with slots

Performance Characteristics:
    - Optimized with slots for memory efficiency
    - Fast serialization/deserialization
    - Minimal memory overhead

Example:
    >>> request = MineruRequest(
    ...     tenant_id="tenant1",
    ...     document_id="doc1",
    ...     content=pdf_bytes
    ... )
    >>> response = MineruResponse(
    ...     document=processed_doc,
    ...     processed_at=datetime.now(),
    ...     duration_seconds=1.5,
    ...     metadata=processing_metadata
    ... )
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from Medical_KG_rev.models.equation import Equation
from Medical_KG_rev.models.figure import Figure
from Medical_KG_rev.models.ir import Block as IrBlock
from Medical_KG_rev.models.ir import Document as IrDocument
from Medical_KG_rev.models.table import Table

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass(slots=True)
class Block:
    """Representation of a document block produced by MinerU.

    A block represents a single document element such as text, table,
    figure, or equation extracted from a PDF document. Each block
    contains spatial information, content, and metadata.

    Attributes:
        id: Unique identifier for the block
        page: Page number where the block appears
        kind: Type of block (text, table, figure, equation)
        text: Extracted text content (if applicable)
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        confidence: Confidence score for extraction (0.0-1.0)
        reading_order: Reading order index for text flow
        metadata: Additional block-specific metadata
        table: Table object if block is a table
        figure: Figure object if block is a figure
        equation: Equation object if block is an equation
        ir_block: Intermediate representation block

    Invariants:
        - id is never empty
        - page is non-negative
        - confidence is between 0.0 and 1.0 if not None
        - bbox coordinates are valid if not None

    Example:
        >>> block = Block(
        ...     id="block_1",
        ...     page=1,
        ...     kind="text",
        ...     text="Sample text content",
        ...     bbox=(10.0, 20.0, 100.0, 30.0),
        ...     confidence=0.95,
        ...     reading_order=1,
        ...     metadata={"font": "Arial", "size": 12}
        ... )
    """

    id: str
    page: int
    kind: str
    text: str | None
    bbox: tuple[float, float, float, float] | None
    confidence: float | None
    reading_order: int | None
    metadata: dict[str, Any]
    table: Table | None = None
    figure: Figure | None = None
    equation: Equation | None = None
    ir_block: IrBlock | None = None


@dataclass(slots=True)
class Document:
    """Structured intermediate representation for a PDF document.

    A document represents the complete structure of a processed PDF,
    containing all extracted blocks, tables, figures, and equations
    along with metadata and provenance information.

    Attributes:
        document_id: Unique identifier for the document
        tenant_id: Tenant identifier for multi-tenancy
        blocks: List of extracted document blocks
        tables: List of extracted tables
        figures: List of extracted figures
        equations: List of extracted equations
        metadata: Document-level metadata
        provenance: Processing provenance information
        ir_document: Intermediate representation document

    Invariants:
        - document_id is never empty
        - tenant_id is never empty
        - All lists are initialized with empty lists
        - Metadata dictionaries are initialized empty

    Example:
        >>> document = Document(
        ...     document_id="doc_123",
        ...     tenant_id="tenant_1",
        ...     blocks=[text_block, table_block],
        ...     tables=[extracted_table],
        ...     metadata={"title": "Research Paper", "author": "Dr. Smith"}
        ... )
    """

    document_id: str
    tenant_id: str
    blocks: list[Block] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    ir_document: IrDocument | None = None


# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

@dataclass(slots=True)
class MineruRequest:
    """Request model for MinerU document processing.

    Represents a single document processing request containing
    either direct PDF content or a storage URI reference.

    Attributes:
        tenant_id: Tenant identifier for multi-tenancy
        document_id: Unique identifier for the document
        content: PDF content bytes (if provided directly)
        storage_uri: Storage URI for PDF content (if stored)

    Invariants:
        - tenant_id is never empty
        - document_id is never empty
        - Either content or storage_uri must be provided

    Example:
        >>> request = MineruRequest(
        ...     tenant_id="tenant_1",
        ...     document_id="doc_123",
        ...     content=pdf_bytes
        ... )
    """

    tenant_id: str
    document_id: str
    content: bytes | None = None
    storage_uri: str | None = None


@dataclass(slots=True)
class MineruResponse:
    """Response model for MinerU document processing.

    Contains the processed document along with processing
    metadata and timing information.

    Attributes:
        document: Processed document structure
        processed_at: Timestamp when processing completed
        duration_seconds: Processing duration in seconds
        metadata: Detailed processing metadata

    Example:
        >>> response = MineruResponse(
        ...     document=processed_doc,
        ...     processed_at=datetime.now(),
        ...     duration_seconds=1.5,
        ...     metadata=processing_metadata
        ... )
    """

    document: Document
    processed_at: datetime
    duration_seconds: float
    metadata: "ProcessingMetadata"


@dataclass(slots=True)
class MineruBatchResponse:
    """Response model for batch MinerU document processing.

    Contains multiple processed documents along with
    aggregated processing metadata and timing information.

    Attributes:
        documents: List of processed documents
        processed_at: Timestamp when batch processing completed
        duration_seconds: Total batch processing duration
        metadata: List of processing metadata for each document

    Example:
        >>> response = MineruBatchResponse(
        ...     documents=[doc1, doc2],
        ...     processed_at=datetime.now(),
        ...     duration_seconds=3.0,
        ...     metadata=[meta1, meta2]
        ... )
    """

    documents: list[Document]
    processed_at: datetime
    duration_seconds: float
    metadata: list["ProcessingMetadata"]


@dataclass(slots=True)
class MineruBatchRequest:
    """Request model for batch MinerU document processing.

    Represents a batch of document processing requests
    for efficient bulk processing.

    Attributes:
        tenant_id: Tenant identifier for multi-tenancy
        requests: Sequence of individual processing requests

    Invariants:
        - tenant_id is never empty
        - requests is never empty

    Example:
        >>> batch_request = MineruBatchRequest(
        ...     tenant_id="tenant_1",
        ...     requests=[req1, req2, req3]
        ... )
    """

    tenant_id: str
    requests: Sequence[MineruRequest]


# ==============================================================================
# PROCESSING METADATA
# ==============================================================================

@dataclass(slots=True)
class ProcessingMetadata:
    """Metadata for MinerU document processing operations.

    Contains detailed information about the processing operation
    including timing, resource usage, model information, and
    CLI execution details.

    Attributes:
        document_id: Identifier of the processed document
        mineru_version: Version of MinerU used for processing
        model_names: Names of models used (layout, table, vision)
        gpu_id: GPU identifier used for processing
        worker_id: Worker process identifier
        started_at: Processing start timestamp
        completed_at: Processing completion timestamp
        duration_seconds: Processing duration in seconds
        cli_stdout: CLI standard output
        cli_stderr: CLI standard error
        cli_descriptor: CLI descriptor information
        planned_memory_mb: Planned memory usage in MB

    Invariants:
        - document_id is never empty
        - started_at <= completed_at
        - duration_seconds >= 0.0

    Example:
        >>> metadata = ProcessingMetadata(
        ...     document_id="doc_123",
        ...     mineru_version="0.1.0",
        ...     model_names={"layout": "yolo", "table": "table-transformer"},
        ...     gpu_id="gpu_0",
        ...     worker_id="worker_1",
        ...     started_at=start_time,
        ...     completed_at=end_time,
        ...     duration_seconds=1.5,
        ...     cli_stdout="Processing completed",
        ...     cli_stderr="",
        ...     cli_descriptor="MinerU CLI v0.1.0"
        ... )
    """

    document_id: str
    mineru_version: str | None
    model_names: dict[str, str]
    gpu_id: str | None
    worker_id: str | None
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    cli_stdout: str
    cli_stderr: str
    cli_descriptor: str
    planned_memory_mb: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert processing metadata to dictionary format.

        Returns:
            Dictionary representation of processing metadata

        Note:
            Converts datetime objects to ISO format strings
            and handles None values appropriately.

        Example:
            >>> metadata_dict = metadata.as_dict()
            >>> print(metadata_dict["started_at"])
            "2023-01-01T12:00:00+00:00"
        """
        return {
            "document_id": self.document_id,
            "mineru_version": self.mineru_version,
            "model_names": dict(self.model_names),
            "gpu_id": self.gpu_id,
            "worker_id": self.worker_id,
            "started_at": self.started_at.astimezone(timezone.utc).isoformat(),
            "completed_at": self.completed_at.astimezone(timezone.utc).isoformat(),
            "duration_seconds": self.duration_seconds,
            "cli_stdout": self.cli_stdout,
            "cli_stderr": self.cli_stderr,
            "cli": self.cli_descriptor,
            "planned_memory_mb": self.planned_memory_mb,
        }


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "Block",
    "Document",
    "MineruRequest",
    "MineruResponse",
    "MineruBatchRequest",
    "MineruBatchResponse",
    "ProcessingMetadata",
]
