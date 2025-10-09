"""Decorator docstring template for pipeline decorators.

This template shows the required structure and content for decorator
docstrings in the Medical_KG_rev pipeline codebase.
"""

# Example decorator docstring structure:

"""[One-line summary: 'Decorator that adds functionality'].

[Detailed explanation of what the decorator does, when to use it,
and how it modifies the decorated function.]

Decorator Behavior:
    [Describe what the decorator adds to the decorated function]
    [List any side effects or modifications]
    [Describe any parameters the decorator accepts]

Usage:
    [Show how to use the decorator]
    [Include examples of common patterns]

Parameters:
    param_name: [Describe decorator parameters if any]
        [Additional details for complex parameters]

Returns:
    [Describe what the decorator returns (usually a wrapper function)]

Note:
    [Any important implementation notes]
    [Performance considerations]
    [Thread safety information]

Example:
    >>> @my_decorator(param="value")
    ... def my_function():
    ...     return "result"
    >>> result = my_function()
    >>> print(result)
    result
"""

# Real example for stage plugin decorator:

def stage_plugin(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    dependencies: list[str] | None = None,
) -> Callable[[Callable[..., StageResult]], Callable[..., StageResult]]:
    """Decorator for registering orchestration stage plugins.

    Registers a function as an orchestration stage plugin that can be
    discovered and executed by the plugin manager. The decorator adds
    metadata about the stage including name, version, and dependencies.

    This decorator enables the plugin system to automatically discover
    and register stage implementations without manual registration code.
    It provides a clean way to mark functions as stage plugins while
    maintaining type safety and documentation.

    Decorator Behavior:
        - Adds stage metadata to the decorated function
        - Registers the function in the global stage registry
        - Validates that the function signature matches StageResult
        - Provides runtime information about the stage

    Usage:
        Use this decorator on functions that implement orchestration
        stages. The function must accept StageContext and return StageResult.

    Parameters:
        name: Unique name for the stage plugin. Used for discovery
            and configuration. Must be unique across all plugins.
        version: Version string for the stage plugin. Used for
            compatibility checking and plugin management.
        description: Human-readable description of what the stage does.
            Used in documentation and error messages.
        dependencies: List of other stage names that this stage depends on.
            Used for dependency resolution and execution ordering.

    Returns:
        Callable: Decorator function that returns the original function
            with added metadata and registration.

    Note:
        Thread safety: Safe to use in multi-threaded environments
        Performance: Minimal overhead, registration happens at import time
        Side effects: Registers stage in global registry

    Example:
        >>> @stage_plugin(
        ...     name="metadata_extraction",
        ...     version="1.0.0",
        ...     description="Extract metadata from documents",
        ...     dependencies=["document_validation"]
        ... )
        ... def extract_metadata(context: StageContext) -> StageResult:
        ...     # Extract metadata logic here
        ...     return StageResult(success=True, data={"metadata": {}})
        >>>
        >>> # Stage is now registered and can be discovered
        >>> stage = plugin_manager.get_stage("metadata_extraction")
        >>> result = stage(context)
    """

# Real example for metrics decorator:

def track_metrics(
    operation: str,
    labels: dict[str, str] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for tracking operation metrics automatically.

    Automatically tracks metrics for decorated functions including
    execution time, success/failure rates, and custom labels.
    Integrates with Prometheus metrics system.

    This decorator provides consistent metrics collection across
    all pipeline operations without requiring manual metric
    emission in each function.

    Decorator Behavior:
        - Measures execution time of decorated function
        - Emits success/failure metrics with labels
        - Handles exceptions and records failure metrics
        - Provides operation-specific labels for filtering

    Usage:
        Use this decorator on functions that represent significant
        operations that should be monitored and tracked.

    Parameters:
        operation: Name of the operation for metrics labeling.
            Used as the 'operation' label in Prometheus metrics.
        labels: Additional labels to include in metrics.
            Useful for adding context like tenant_id, model_name, etc.

    Returns:
        Callable: Decorator function that returns the original function
            with added metrics tracking.

    Note:
        Thread safety: Safe to use in multi-threaded environments
        Performance: Minimal overhead, metrics emission is async
        Side effects: Emits Prometheus metrics

    Example:
        >>> @track_metrics(
        ...     operation="chunking",
        ...     labels={"strategy": "section", "model": "biobert"}
        ... )
        ... def chunk_document(text: str, strategy: str) -> list[Chunk]:
        ...     # Chunking logic here
        ...     return chunks
        >>>
        >>> # Metrics are automatically tracked
        >>> chunks = chunk_document("text", "section")
        >>> # Prometheus metrics emitted: operation_time, operation_success, etc.
    """

# Real example for validation decorator:

def validate_request(
    schema: dict[str, Any],
    strict: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for validating function arguments against JSON schema.

    Validates function arguments against a JSON schema before execution.
    Raises ValidationError if arguments don't match the schema.

    This decorator provides consistent request validation across
    all API endpoints and service methods without requiring
    manual validation code in each function.

    Decorator Behavior:
        - Validates function arguments against provided schema
        - Raises ValidationError for invalid arguments
        - Supports strict and non-strict validation modes
        - Provides detailed error messages for validation failures

    Usage:
        Use this decorator on functions that accept structured
        input data that should be validated.

    Parameters:
        schema: JSON schema for validating function arguments.
            Must be a valid JSON schema object.
        strict: Whether to use strict validation mode.
            Strict mode disallows additional properties not in schema.

    Returns:
        Callable: Decorator function that returns the original function
            with added validation.

    Note:
        Thread safety: Safe to use in multi-threaded environments
        Performance: Validation overhead depends on schema complexity
        Side effects: May raise ValidationError for invalid input

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "tenant_id": {"type": "string"},
        ...         "document_id": {"type": "string"}
        ...     },
        ...     "required": ["tenant_id", "document_id"]
        ... }
        >>>
        >>> @validate_request(schema, strict=True)
        ... def process_document(tenant_id: str, document_id: str) -> str:
        ...     return f"Processing {document_id} for {tenant_id}"
        >>>
        >>> # Validation happens automatically
        >>> result = process_document("tenant1", "doc1")
        >>> # Raises ValidationError if arguments don't match schema
    """
