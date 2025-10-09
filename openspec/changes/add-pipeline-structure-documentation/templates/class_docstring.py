"""Class docstring template for pipeline classes.

This template shows the required structure and content for class-level docstrings
in the Medical_KG_rev pipeline codebase.
"""

# Example class docstring structure:

"""[One-line summary of class purpose].

[Detailed explanation of what the class does, why it exists, and how it fits
into the larger architecture. Explain the key abstractions it provides.]

This class implements the [pattern name] pattern for [purpose]. It coordinates
between [upstream components] and [downstream components] to [achieve goal].

Attributes:
    attribute_name: [Type already in code, describe purpose and valid ranges]
    _private_attr: [Describe internal state and invariants]

Invariants:
    - [List any class invariants that must hold throughout object lifetime]
    - [Example: self._cache is never None after __init__]
    - [Example: self._count is always >= 0]

Thread Safety:
    - [Thread-safe if all methods are thread-safe]
    - [Not thread-safe: describe which methods are unsafe]
    - [Conditionally safe: describe locking strategy]

Lifecycle:
    - [Describe object lifecycle: creation, usage, cleanup]
    - [Mention if cleanup is automatic or requires explicit close()]

Example:
    >>> coordinator = ChunkingCoordinator(
    ...     lifecycle=JobLifecycleManager(),
    ...     chunker=ChunkingService(),
    ...     config=CoordinatorConfig(name="chunking")
    ... )
    >>> result = coordinator.execute(ChunkingRequest(...))
    >>> print(f"Processed {len(result.chunks)} chunks")
"""

# Real example for ChunkingCoordinator:

"""Chunking coordinator for synchronous document chunking operations.

ChunkingCoordinator coordinates synchronous chunking operations by managing job
lifecycle, delegating to ChunkingService, and translating chunking exceptions
to coordinator errors. It implements the Coordinator pattern to provide a
consistent interface for chunking operations across different protocol handlers.

This class implements the Coordinator pattern for chunking operations. It coordinates
between gateway services (upstream) and ChunkingService/JobLifecycleManager (downstream)
to provide reliable, tracked chunking operations with proper error handling.

Attributes:
    _lifecycle: JobLifecycleManager for tracking job state transitions
    _chunker: ChunkingService for actual chunking operations
    _errors: ChunkingErrorTranslator for converting exceptions to HTTP problem details
    _config: CoordinatorConfig with name and runtime settings
    _metrics: CoordinatorMetrics for performance and error tracking

Invariants:
    - self._lifecycle is never None after __init__
    - self._chunker is never None after __init__
    - self._errors is never None after __init__
    - All public methods require valid tenant_id in requests

Thread Safety:
    - Not thread-safe: Coordinator instances are not designed for concurrent use
    - Each request should be handled by a single coordinator instance
    - Shared dependencies (lifecycle, chunker) must be thread-safe

Lifecycle:
    - Created during application startup with injected dependencies
    - Used for request processing throughout application lifetime
    - No explicit cleanup required (dependencies handle their own cleanup)

Example:
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
