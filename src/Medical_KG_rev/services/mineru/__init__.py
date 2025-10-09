"""MinerU GPU microservice implementation.

This module provides the MinerU GPU microservice implementation for
processing PDF documents using GPU-accelerated layout analysis and
text extraction. It includes the main service classes, request/response
models, and custom exceptions.

Key Components:
    - MineruGrpcService: gRPC service implementation
    - MineruProcessor: Core PDF processing engine
    - MineruRequest/Response: Request and response models
    - Custom exceptions for error handling

Responsibilities:
    - Provide gRPC interface for PDF processing
    - Coordinate GPU-accelerated document analysis
    - Handle request/response serialization
    - Manage GPU resource allocation and cleanup
    - Provide error handling and recovery

Collaborators:
    - gRPC framework for service interface
    - GPU drivers for hardware acceleration
    - MinerU CLI for document processing
    - Object storage for asset management

Side Effects:
    - Allocates GPU memory for processing
    - Creates temporary files during processing
    - May raise exceptions for GPU/resource issues

Thread Safety:
    - Thread-safe: Service instances handle concurrent requests
    - GPU operations are serialized per instance

Performance Characteristics:
    - GPU-accelerated processing for large documents
    - Batch processing support for efficiency
    - Memory usage scales with document size
    - Supports concurrent request handling

Example:
    >>> from Medical_KG_rev.services.mineru import MineruGrpcService
    >>> service = MineruGrpcService()
    >>> response = service.ProcessPdf(request)
    >>> assert response.documents is not None

"""

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "MineruGpuUnavailableError",
    "MineruGrpcService",
    "MineruOutOfMemoryError",
    "MineruProcessor",
    "MineruRequest",
    "MineruResponse",
]


# ==============================================================================
# LAZY IMPORT HELPER
# ==============================================================================

def __getattr__(name: str):  # pragma: no cover - simple lazy import helper
    """Lazy import helper for module attributes.

    Args:
        name: Attribute name to import

    Returns:
        Imported attribute

    Raises:
        AttributeError: If attribute is not in __all__

    Example:
        >>> service = __getattr__("MineruGrpcService")
        >>> assert service is not None

    """
    if name in __all__:
        from .service import (
            MineruGpuUnavailableError,
            MineruGrpcService,
            MineruOutOfMemoryError,
            MineruProcessor,
            MineruRequest,
            MineruResponse,
        )

        return {
            "MineruGrpcService": MineruGrpcService,
            "MineruProcessor": MineruProcessor,
            "MineruRequest": MineruRequest,
            "MineruResponse": MineruResponse,
            "MineruOutOfMemoryError": MineruOutOfMemoryError,
            "MineruGpuUnavailableError": MineruGpuUnavailableError,
        }[name]
    raise AttributeError(name)
