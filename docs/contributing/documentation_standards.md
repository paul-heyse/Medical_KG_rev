# Documentation Standards

This guide outlines the documentation standards for the Medical KG project, including docstring conventions, section headers, and automated quality checks.

## Overview

Documentation is critical for maintaining code quality and enabling effective collaboration. This project follows Google-style docstrings and enforces consistent code organization through section headers.

### Why Documentation Matters

- **Code Understanding**: Clear docstrings help developers understand purpose, parameters, and behavior
- **API Discovery**: Well-documented APIs are easier to discover and use correctly
- **Maintenance**: Documentation reduces the time needed to understand and modify code
- **Quality Assurance**: Automated checks ensure documentation standards are maintained

### Standards We Follow

- **Google-style docstrings** for all modules, classes, and functions
- **Section headers** for consistent code organization
- **Type hints** for all function parameters and return values
- **Inline comments** for complex logic and design decisions

## Google-Style Docstrings

### Module Docstrings

Every module should start with a comprehensive docstring:

```python
"""One-line summary of module purpose.

This module provides detailed explanation of what the module does, its role
in the larger system, and key design decisions.

Key Responsibilities:
    - Responsibility 1: Be specific about what the module handles
    - Responsibility 2: Include data transformations, external calls, etc.
    - Responsibility 3: Mention any caching, rate limiting, etc.

Collaborators:
    - Upstream: List modules/services that call into this one
    - Downstream: List modules/services this one depends on

Side Effects:
    - Database writes, external API calls, file I/O, metric emission
    - Global state modifications, cache updates
    - None if pure/functional

Thread Safety:
    - Thread-safe: All public functions can be called from multiple threads
    - Not thread-safe: Must be called from single thread
    - Conditionally safe: Describe conditions

Performance Characteristics:
    - Time complexity for main operations
    - Memory usage patterns
    - Rate limits or throttling behavior

Example:
    >>> from Medical_KG_rev.gateway.coordinators import ChunkingCoordinator
    >>> coordinator = ChunkingCoordinator(...)
    >>> result = coordinator.execute(request)
"""
```

### Class Docstrings

Classes should document their purpose, attributes, invariants, and usage:

```python
class ChunkingCoordinator:
    """Coordinates synchronous chunking operations.

    This class implements the coordinator pattern for chunking operations.
    It coordinates between gateway services and chunking domain logic to
    provide protocol-agnostic chunking capabilities.

    Attributes:
        _lifecycle: JobLifecycleManager for tracking job states
        _chunker: ChunkingService for actual chunking operations
        _errors: ChunkingErrorTranslator for error translation

    Invariants:
        - self._lifecycle is never None after __init__
        - self._chunker is never None after __init__
        - self._errors is never None after __init__

    Thread Safety:
        - Not thread-safe: Must be called from single thread

    Lifecycle:
        - Created with dependencies injected
        - Used for coordinating chunking operations
        - No explicit cleanup required

    Example:
        >>> coordinator = ChunkingCoordinator(
        ...     lifecycle=JobLifecycleManager(),
        ...     chunker=ChunkingService(),
        ...     config=CoordinatorConfig(name="chunking")
        ... )
        >>> result = coordinator.execute(ChunkingRequest(...))
        >>> print(f"Processed {len(result.chunks)} chunks")
    """
```

### Function/Method Docstrings

Functions should document parameters, return values, exceptions, and behavior:

```python
def execute(self, request: ChunkingRequest) -> ChunkingResult:
    """Execute chunking operation with job lifecycle management.

    Coordinates the full chunking workflow: creates job entry, extracts
    document text, creates chunking command, delegates to chunking service,
    handles exceptions, assembles results, and marks job completed.

    Args:
        request: ChunkingRequest with document and chunking parameters
            - document_id: Unique identifier for document being chunked
            - text: Optional document text (can also be in options["text"])
            - strategy: Chunking strategy name, defaults to "section"
            - chunk_size: Maximum tokens per chunk, defaults to profile setting
            - overlap: Token overlap between chunks, defaults to profile setting
            - options: Additional metadata and configuration

    Returns:
        ChunkingResult with chunks and metadata:
            - chunks: Sequence of DocumentChunk objects with content and metadata
            - job_id: Unique identifier for this chunking job
            - duration_s: Time taken to complete chunking operation
            - metadata: Additional operation metadata

    Raises:
        CoordinatorError: For all handled errors after translation
            - ProfileNotFoundError: When chunking profile doesn't exist
            - InvalidDocumentError: When document text is empty or invalid
            - ChunkingUnavailableError: When chunking service is unavailable

    Note:
        Emits metrics for failures, updates job lifecycle

    Example:
        >>> request = ChunkingRequest(
        ...     document_id="doc1",
        ...     text="Sample document text",
        ...     strategy="section"
        ... )
        >>> result = coordinator.execute(request)
        >>> assert len(result.chunks) > 0
    """
```

### Dataclass Docstrings

Dataclasses should document each field:

```python
@dataclass
class ChunkingRequest:
    """Request for chunking operation.

    Attributes:
        document_id: Unique identifier for document being chunked
        text: Optional document text (can also be in options["text"])
        strategy: Chunking strategy name (e.g., "section", "semantic"), defaults to "section"
        chunk_size: Maximum tokens per chunk, defaults to profile setting
        overlap: Token overlap between chunks, defaults to profile setting
        options: Additional metadata and configuration
    """
    document_id: str
    text: str | None = None
    strategy: str = "section"
    chunk_size: int | None = None
    overlap: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
```

## Section Headers

Code should be organized into labeled sections with consistent ordering:

### Gateway Coordinator Module Structure

```python
# ============================================================================
# IMPORTS
# ============================================================================
# (stdlib imports)
# (blank line)
# (third-party imports)
# (blank line)
# (first-party imports from Medical_KG_rev)
# (blank line)
# (relative imports)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================
# (Dataclasses for request and result types used by coordinator)

# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================
# (Main coordinator class with __init__ and public execute method)

# ============================================================================
# PRIVATE HELPERS
# ============================================================================
# (Private methods for text extraction, metadata handling, etc.)

# ============================================================================
# ERROR TRANSLATION
# ============================================================================
# (Methods for translating exceptions to coordinator errors)

# ============================================================================
# EXPORTS
# ============================================================================
# (__all__ list)
```

### Service Layer Module Structure

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# TYPE DEFINITIONS & CONSTANTS
# ============================================================================

# ============================================================================
# SERVICE CLASS DEFINITION
# ============================================================================

# ============================================================================
# CHUNKING ENDPOINTS
# ============================================================================

# ============================================================================
# EMBEDDING ENDPOINTS
# ============================================================================

# ============================================================================
# RETRIEVAL ENDPOINTS
# ============================================================================

# ============================================================================
# ADAPTER MANAGEMENT ENDPOINTS
# ============================================================================

# ============================================================================
# VALIDATION ENDPOINTS
# ============================================================================

# ============================================================================
# EXTRACTION ENDPOINTS
# ============================================================================

# ============================================================================
# ADMIN & UTILITY ENDPOINTS
# ============================================================================

# ============================================================================
# PRIVATE HELPERS
# ============================================================================
```

### Ordering Rules Within Sections

- **Imports**: stdlib, third-party, first-party, relative (each group alphabetically sorted)
- **Classes**: Base classes before subclasses, interfaces before implementations
- **Class methods**: `__init__` first, public methods (alphabetically), private methods (alphabetically), static/class methods last
- **Functions**: Public functions before private functions, alphabetical within each group

## Running Checks Locally

### Ruff Docstring Check

```bash
ruff check --select D src/Medical_KG_rev/gateway src/Medical_KG_rev/services src/Medical_KG_rev/orchestration
```

### Section Header Check

```bash
python scripts/check_section_headers.py
```

### Docstring Coverage Check

```bash
python scripts/check_docstring_coverage.py --min-coverage 90
```

### Pre-commit Hooks

Install pre-commit hooks to run checks automatically:

```bash
pre-commit install
pre-commit run --all-files
```

## Interpreting Errors

### Common Ruff Docstring Errors

- **D100**: Missing docstring in public module
  - **Fix**: Add module docstring at top of file
- **D101**: Missing docstring in public class
  - **Fix**: Add docstring immediately after class definition
- **D102**: Missing docstring in public method
  - **Fix**: Add docstring immediately after method definition
- **D103**: Missing docstring in public function
  - **Fix**: Add docstring immediately after function definition
- **D107**: Missing docstring in **init**
  - **Fix**: Add docstring to **init** method

### Section Header Errors

- **Missing section header**: Add required section headers per `section_headers.md`
- **Incorrect order**: Reorder sections to match canonical order
- **Missing content**: Ensure each section contains appropriate code

### Docstring Coverage Errors

- **Coverage < 90%**: Add docstrings to modules, classes, and functions
- **Missing Args section**: Add Args section for functions with parameters
- **Missing Returns section**: Add Returns section for functions that return values
- **Missing Raises section**: Add Raises section for functions that raise exceptions

## Templates

Reference templates in `openspec/changes/add-pipeline-structure-documentation/templates/`:

- `module_docstring.py`: Module-level docstring template
- `class_docstring.py`: Class-level docstring template
- `function_docstring.py`: Function-level docstring template
- `dataclass_docstring.py`: Dataclass docstring template
- `protocol_docstring.py`: Protocol docstring template
- `exception_handler_docstring.py`: Exception handler docstring template
- `async_docstring.py`: Async function docstring template
- `decorator_docstring.py`: Decorator docstring template
- `property_docstring.py`: Property docstring template
- `constant_docstring.py`: Constant docstring template
- `test_docstring.py`: Test function docstring template

## Examples

### Before (Poor Documentation)

```python
def chunk_text(text, strategy="section"):
    chunks = []
    # ... implementation
    return chunks
```

### After (Good Documentation)

```python
def chunk_text(text: str, strategy: str = "section") -> list[DocumentChunk]:
    """Chunk document text into smaller segments.

    Splits the input text into chunks using the specified strategy.
    Each chunk contains a portion of the original text with metadata.

    Args:
        text: The document text to chunk. Must be non-empty string.
        strategy: Chunking strategy to use. Valid values: "section", "semantic".
            Defaults to "section".

    Returns:
        List of DocumentChunk objects containing:
            - content: The chunk text content
            - metadata: Chunk metadata including position and strategy

    Raises:
        ValueError: If text is empty or strategy is invalid
        ChunkingError: If chunking operation fails

    Example:
        >>> chunks = chunk_text("This is a sample document.", "section")
        >>> assert len(chunks) > 0
        >>> assert chunks[0].content == "This is a sample document."
    """
    if not text:
        raise ValueError("Text cannot be empty")

    # ... implementation
    return chunks
```

## Best Practices

1. **Write docstrings first**: Document the interface before implementing
2. **Be specific**: Include exact parameter types, valid ranges, and constraints
3. **Include examples**: Show typical usage patterns
4. **Document side effects**: Mention metrics, logging, external calls
5. **Keep docstrings up to date**: Update when changing function signatures
6. **Use type hints**: Complement docstrings with type annotations
7. **Add inline comments**: Explain complex logic and design decisions

## Getting Help

- Check existing code for examples of good documentation
- Refer to the templates in the `templates/` directory
- Run the automated checks to identify specific issues
- Ask team members for review and feedback
