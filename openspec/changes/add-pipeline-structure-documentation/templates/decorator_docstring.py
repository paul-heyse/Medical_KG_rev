"""Decorator docstring template for pipeline decorators.

This template shows the required structure and content for decorator
docstrings in the Medical_KG_rev pipeline codebase.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

StageResult = TypeVar("StageResult")
T = TypeVar("T")
Chunk = TypeVar("Chunk")

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
    """Register an orchestration stage plugin.

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

    Args:
    ----
        name: Unique identifier for the stage plugin used during discovery.
        version: Semantic version string that enables compatibility checks.
        description: Human-readable explanation of the stage behavior.
        dependencies: Optional list of stage names that must execute first.

    Returns:
    -------
        Callable[..., StageResult]: Decorator that returns the original function
            with added metadata and registration hooks.

    Note:
    ----
        Thread safety: Safe to use in multi-threaded environments
        Performance: Minimal overhead, registration happens at import time
        Side effects: Registers stage in global registry

    Example:
    -------
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
    """Track operation metrics automatically.

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

    Args:
    ----
        operation: Name of the operation for metrics labeling. Used as the
            'operation' label in Prometheus metrics.
        labels: Additional labels to include in metrics. Useful for adding
            context like tenant_id, model_name, or stage.

    Returns:
    -------
        Callable[[Callable[..., T]], Callable[..., T]]: Decorator that wraps the
            original callable with metrics instrumentation.

    Note:
    ----
        Thread safety: Safe to use in multi-threaded environments
        Performance: Minimal overhead, metrics emission is async
        Side effects: Emits Prometheus metrics

    Example:
    -------
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
    """Validate function arguments against a JSON schema.

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

    Args:
    ----
        schema: JSON schema for validating function arguments. Must be a valid
            JSON schema object.
        strict: Whether to use strict validation mode. Strict mode disallows
            additional properties not present in the schema.

    Returns:
    -------
        Callable[[Callable[..., T]], Callable[..., T]]: Decorator that returns
            the original function with automatic validation.

    Note:
    ----
        Thread safety: Safe to use in multi-threaded environments
        Performance: Validation overhead depends on schema complexity
        Side effects: May raise ValidationError for invalid input

    Example:
    -------
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
