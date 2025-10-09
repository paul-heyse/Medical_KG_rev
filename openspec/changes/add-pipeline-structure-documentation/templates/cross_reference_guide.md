# Cross-Reference Guide for Pipeline Documentation

This guide explains how to use Sphinx-style cross-references in docstrings to create links between related components in the Medical_KG_rev pipeline codebase.

## Basic Cross-Reference Syntax

### Classes

```python
:class:`ClassName`           # Link to a class
:class:`~ClassName`          # Link to class, show only class name (no module path)
:class:`module.ClassName`    # Link to class in specific module
```

### Functions and Methods

```python
:func:`function_name`                    # Link to a function
:func:`~function_name`                   # Link to function, show only function name
:func:`module.function_name`             # Link to function in specific module
:meth:`ClassName.method_name`            # Link to a method
:meth:`~ClassName.method_name`           # Link to method, show only method name
:meth:`module.ClassName.method_name`     # Link to method in specific module
```

### Modules

```python
:mod:`module_name`           # Link to a module
:mod:`~module_name`          # Link to module, show only module name
:mod:`package.module_name`   # Link to module in specific package
```

### Exceptions

```python
:exc:`ExceptionName`         # Link to an exception class
:exc:`~ExceptionName`        # Link to exception, show only exception name
:exc:`module.ExceptionName`  # Link to exception in specific module
```

### Data and Constants

```python
:data:`CONSTANT_NAME`        # Link to a constant or variable
:data:`~CONSTANT_NAME`       # Link to constant, show only constant name
:data:`module.CONSTANT_NAME` # Link to constant in specific module
```

### Attributes

```python
:attr:`ClassName.attribute_name`         # Link to a class attribute
:attr:`~ClassName.attribute_name`        # Link to attribute, show only attribute name
:attr:`module.ClassName.attribute_name` # Link to attribute in specific module
```

## Examples in Pipeline Documentation

### Coordinator Cross-References

```python
class ChunkingCoordinator:
    """Chunking coordinator for synchronous document chunking operations.

    This class implements the :class:`BaseCoordinator` pattern for chunking
    operations. It coordinates between gateway services and :class:`ChunkingService`
    to provide reliable chunking operations.

    The coordinator uses :class:`JobLifecycleManager` for job tracking and
    :class:`ChunkingErrorTranslator` for error translation.

    Example:
        >>> coordinator = ChunkingCoordinator(
        ...     lifecycle=:class:`JobLifecycleManager`(),
        ...     chunker=:class:`ChunkingService`(),
        ...     config=:class:`CoordinatorConfig`(name="chunking")
        ... )
        >>> result = coordinator.execute(:class:`ChunkingRequest`(...))
    """

    def _execute(self, request: ChunkingRequest) -> ChunkingResult:
        """Execute chunking operation with comprehensive error handling.

        This method delegates to :meth:`ChunkingService.chunk` and handles
        exceptions using :meth:`ChunkingErrorTranslator.translate`.

        Args:
            request: :class:`ChunkingRequest` with document and chunking parameters

        Returns:
            :class:`ChunkingResult`: Contains chunks and metadata

        Raises:
            :exc:`CoordinatorError`: For all handled errors after translation
        """
```

### Service Cross-References

```python
class ChunkingService:
    """Service layer adapter to chunking library.

    This service provides a facade over the chunking library and handles
    profile loading, chunker configuration, and chunk execution.

    The service uses :class:`ChunkingProfile` for configuration and
    :class:`ChunkCommand` for operation parameters.

    Example:
        >>> service = ChunkingService(config=:class:`ChunkingConfig`())
        >>> command = :class:`ChunkCommand`(text="Sample text", strategy="section")
        >>> chunks = service.chunk(command)
    """

    def chunk(self, command: ChunkCommand) -> list[DocumentChunk]:
        """Execute chunking operation using the configured chunker.

        This method loads the appropriate :class:`ChunkingProfile` and
        executes chunking using the chunking library.

        Args:
            command: :class:`ChunkCommand` with chunking parameters

        Returns:
            list[:class:`DocumentChunk`]: List of chunked document segments

        Raises:
            :exc:`ProfileNotFoundError`: If the specified profile doesn't exist
            :exc:`ChunkingFailedError`: If chunking operation fails
        """
```

### Error Translation Cross-References

```python
class ChunkingErrorTranslator:
    """Translates chunking exceptions to HTTP problem details.

    This class maps chunking library exceptions to standardized
    :class:`ProblemDetail` objects with appropriate HTTP status codes.

    The translator handles exceptions from :class:`ChunkingService` and
    converts them to :exc:`CoordinatorError` instances.

    Example:
        >>> translator = ChunkingErrorTranslator()
        >>> report = translator.translate(
        ...     :exc:`ProfileNotFoundError`("profile_not_found"),
        ...     command=:class:`ChunkCommand`(...)
        ... )
        >>> assert report.problem.status == 400
    """

    def translate(
        self,
        exc: Exception,
        command: ChunkCommand,
        job_id: str
    ) -> ChunkingErrorReport | None:
        """Translate chunking exception to error report.

        This method maps exceptions to :class:`ChunkingErrorReport` objects
        containing :class:`ProblemDetail` information.

        Args:
            exc: Exception to translate
            command: :class:`ChunkCommand` for context
            job_id: Job identifier for correlation

        Returns:
            :class:`ChunkingErrorReport` | None: Error report or None if
                exception is not recognized
        """
```

### Orchestration Cross-References

```python
def metadata_extraction_stage(context: StageContext) -> StageResult:
    """Extract metadata from documents in the orchestration pipeline.

    This stage extracts metadata from documents and prepares it for
    downstream processing stages.

    The stage uses :class:`DocumentMetadataExtractor` for extraction
    and :class:`MetadataValidator` for validation.

    Args:
        context: :class:`StageContext` with document and job information

    Returns:
        :class:`StageResult`: Contains extracted metadata and validation results

    Raises:
        :exc:`StageExecutionError`: If metadata extraction fails
    """
```

## Best Practices

### 1. Use Descriptive Cross-References

```python
# Good: Descriptive and specific
:class:`ChunkingCoordinator` handles synchronous chunking operations

# Avoid: Generic or unclear
:class:`Coordinator` handles operations
```

### 2. Link to Related Components

```python
# Good: Links to related components
This method uses :class:`ChunkingService` for actual chunking and
:class:`JobLifecycleManager` for job tracking.

# Avoid: No cross-references
This method uses the chunking service for actual chunking and
the job lifecycle manager for job tracking.
```

### 3. Use Tilde for Cleaner Display

```python
# Good: Clean display without module path
The :class:`~ChunkingCoordinator` class...

# Avoid: Full module path in display
The :class:`Medical_KG_rev.gateway.coordinators.ChunkingCoordinator` class...
```

### 4. Cross-Reference in Examples

```python
# Good: Cross-references in examples
>>> coordinator = :class:`ChunkingCoordinator`(
...     lifecycle=:class:`JobLifecycleManager`(),
...     chunker=:class:`ChunkingService`()
... )

# Avoid: No cross-references in examples
>>> coordinator = ChunkingCoordinator(
...     lifecycle=JobLifecycleManager(),
...     chunker=ChunkingService()
... )
```

### 5. Link to Exceptions

```python
# Good: Link to specific exceptions
Raises:
    :exc:`ProfileNotFoundError`: If the specified profile doesn't exist
    :exc:`ChunkingFailedError`: If chunking operation fails

# Avoid: Generic exception references
Raises:
    Exception: If something goes wrong
```

## Common Cross-Reference Patterns

### Coordinator Pattern

```python
class MyCoordinator:
    """Coordinator for MyOperation.

    This class implements the :class:`BaseCoordinator` pattern for
    MyOperation. It coordinates between :class:`GatewayService` and
    :class:`MyService` to provide reliable operations.

    The coordinator uses :class:`JobLifecycleManager` for job tracking
    and :class:`MyErrorTranslator` for error translation.
    """
```

### Service Pattern

```python
class MyService:
    """Service layer for MyOperation.

    This service provides a facade over the MyOperation library and
    handles configuration, execution, and error handling.

    The service uses :class:`MyConfig` for configuration and
    :class:`MyCommand` for operation parameters.
    """
```

### Error Translation Pattern

```python
class MyErrorTranslator:
    """Translates MyOperation exceptions to HTTP problem details.

    This class maps MyOperation exceptions to standardized
    :class:`ProblemDetail` objects with appropriate HTTP status codes.

    The translator handles exceptions from :class:`MyService` and
    converts them to :exc:`CoordinatorError` instances.
    """
```

### Stage Plugin Pattern

```python
def my_stage(context: StageContext) -> StageResult:
    """MyStage for the orchestration pipeline.

    This stage performs MyOperation in the orchestration pipeline.

    The stage uses :class:`MyService` for operations and
    :class:`MyValidator` for validation.

    Args:
        context: :class:`StageContext` with operation information

    Returns:
        :class:`StageResult`: Contains operation results
    """
```

## Validation

Cross-references are validated during documentation generation. Common issues:

1. **Broken Links**: Cross-references to non-existent classes, functions, or modules
2. **Incorrect Syntax**: Malformed cross-reference syntax
3. **Missing Imports**: Cross-references to components not imported in the module
4. **Circular References**: Cross-references that create circular dependencies

Use the documentation generation process to validate cross-references:

```bash
# Generate documentation and check for broken links
mkdocs build --strict

# Check specific cross-references
mkdocs build --verbose | grep "WARNING.*cross-reference"
```
