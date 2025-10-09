"""Module docstring template for pipeline modules.

This template shows the required structure and content for module-level docstrings
in the Medical_KG_rev pipeline codebase.
"""

# Example module docstring structure:

"""[One-line summary of module purpose].

This module provides [detailed explanation of what the module does, its role
in the larger system, and key design decisions].

Key Responsibilities:
    - [Responsibility 1: Be specific about what the module handles]
    - [Responsibility 2: Include data transformations, external calls, etc.]
    - [Responsibility 3: Mention any caching, rate limiting, etc.]

Collaborators:
    - Upstream: [List modules/services that call into this one]
    - Downstream: [List modules/services this one depends on]

Side Effects:
    - [Database writes, external API calls, file I/O, metric emission]
    - [Global state modifications, cache updates]
    - [None if pure/functional]

Thread Safety:
    - [Thread-safe: All public functions can be called from multiple threads]
    - [Not thread-safe: Must be called from single thread]
    - [Conditionally safe: Describe conditions]

Performance Characteristics:
    - [Time complexity for main operations]
    - [Memory usage patterns]
    - [Rate limits or throttling behavior]

Example:
    >>> from Medical_KG_rev.gateway.coordinators import ChunkingCoordinator
    >>> coordinator = ChunkingCoordinator(...)
    >>> result = coordinator.execute(request)
"""

# Real example for chunking coordinator:

"""Chunking coordinator for synchronous document chunking operations.

This module provides the ChunkingCoordinator class that coordinates synchronous
chunking jobs by managing job lifecycle, delegating to ChunkingService, and
translating chunking exceptions to coordinator errors.

Key Responsibilities:
    - Job lifecycle management (create, track, complete/fail jobs)
    - Request validation and text extraction
    - Error translation from chunking exceptions to HTTP problem details
    - Metrics emission for chunking operations
    - Integration with ChunkingErrorTranslator for consistent error handling

Collaborators:
    - Upstream: Gateway services (REST/GraphQL/gRPC handlers)
    - Downstream: ChunkingService, JobLifecycleManager, ChunkingErrorTranslator

Side Effects:
    - Creates job entries in job lifecycle manager
    - Emits Prometheus metrics for chunking operations
    - Logs errors and performance data
    - Updates job state (pending → completed/failed)

Thread Safety:
    - Not thread-safe: Coordinator instances are not designed for concurrent use
    - Each request should be handled by a single coordinator instance

Performance Characteristics:
    - O(n) time complexity where n is document length
    - Memory usage scales with chunk count and size
    - Typical operation: 100-500ms for documents up to 10K tokens

Example:
    >>> from Medical_KG_rev.gateway.coordinators import ChunkingCoordinator
    >>> coordinator = ChunkingCoordinator(
    ...     lifecycle=JobLifecycleManager(),
    ...     chunker=ChunkingService(),
    ...     config=CoordinatorConfig(name="chunking")
    ... )
    >>> result = coordinator.execute(ChunkingRequest(
    ...     tenant_id="tenant1",
    ...     document_id="doc1",
    ...     text="Sample document text for chunking."
    ... ))
    >>> print(f"Processed {len(result.chunks)} chunks")
"""
