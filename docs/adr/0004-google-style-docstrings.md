# ADR-0004: Google-Style Docstrings

## Status

Accepted

## Context

The Medical KG codebase lacked consistent documentation standards. Different modules used different docstring formats, making it difficult to:

- Generate consistent API documentation
- Understand code interfaces and behavior
- Maintain documentation quality
- Onboard new developers
- Use automated documentation tools effectively

The existing issues included:

- Inconsistent docstring formats across modules
- Missing docstrings for many classes and functions
- Incomplete parameter and return value documentation
- Lack of examples and usage patterns
- No standardized format for different types of code elements

The system needed a standardized docstring format that would:

- Provide consistent documentation across all modules
- Support automated documentation generation
- Enable better code understanding and maintenance
- Facilitate developer onboarding
- Support comprehensive API documentation

## Decision

We will adopt **Google-Style Docstrings** as the standard format for all Python code in the Medical KG project.

### Docstring Format

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """One-line summary of function purpose.

    Detailed explanation of what the function does, why it exists,
    and how it fits into the larger system. Include key design
    decisions and important implementation details.

    Args:
        param1: Description of first parameter, including valid
            values and constraints. Use multiple lines for complex
            parameters with detailed explanations.
        param2: Description of second parameter. Defaults to 10.

    Returns:
        Description of return value, including structure and
        meaning. Specify what None return means if applicable.

    Raises:
        ValueError: When param1 is empty or invalid.
        RuntimeError: When operation fails unexpectedly.

    Note:
        Important implementation notes, performance considerations,
        thread safety, or side effects.

    Warning:
        Any gotchas or surprising behavior users should know about.

    Example:
        >>> result = function_name("test", 20)
        >>> assert result is True
        >>> # Raises ValueError if param1 is empty
    """
```

### Required Sections

- **Summary**: One-line description of purpose
- **Description**: Detailed explanation of functionality
- **Args**: Parameter documentation with types and constraints
- **Returns**: Return value documentation with structure
- **Raises**: Exception documentation with conditions
- **Example**: Usage examples with expected behavior

### Optional Sections

- **Note**: Implementation details, performance, thread safety
- **Warning**: Gotchas, surprising behavior, limitations
- **Attributes**: Class attribute documentation
- **Invariants**: Class invariants that must hold
- **Thread Safety**: Concurrency characteristics
- **Lifecycle**: Object lifecycle and cleanup requirements

### Module-Level Docstrings

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

### Class-Level Docstrings

```python
class MyClass:
    """One-line summary of class purpose.

    Detailed explanation of what the class does, why it exists, and how it fits
    into the larger architecture. Explain the key abstractions it provides.

    This class implements the [pattern name] pattern for [purpose]. It coordinates
    between [upstream components] and [downstream components] to [achieve goal].

    Attributes:
        attribute_name: Type already in code, describe purpose and valid ranges
        _private_attr: Describe internal state and invariants

    Invariants:
        - List any class invariants that must hold throughout object lifetime
        - Example: self._cache is never None after __init__
        - Example: self._count is always >= 0

    Thread Safety:
        - Thread-safe if all methods are thread-safe
        - Not thread-safe: describe which methods are unsafe
        - Conditionally safe: describe locking strategy

    Lifecycle:
        - Describe object lifecycle: creation, usage, cleanup
        - Mention if cleanup is automatic or requires explicit close()

    Example:
        >>> instance = MyClass(param1="value", param2=42)
        >>> result = instance.method()
        >>> print(f"Result: {result}")
    """
```

### Dataclass Docstrings

```python
@dataclass
class MyDataClass:
    """One-line summary of dataclass purpose.

    Detailed explanation of what the dataclass represents and how it's used.

    Attributes:
        field1: Description of first field with valid values and constraints
        field2: Description of second field with default behavior
        field3: Description of third field with type information
    """
    field1: str
    field2: int = 10
    field3: list[str] = field(default_factory=list)
```

## Consequences

### Positive

- **Consistent Documentation**: Uniform docstring format across all modules
- **Better Tooling Support**: Google-style docstrings work well with mkdocstrings, Sphinx, and other tools
- **Improved Readability**: Clear structure makes documentation easier to read and understand
- **Comprehensive Coverage**: Required sections ensure complete documentation
- **Example Support**: Built-in support for usage examples
- **Type Integration**: Works well with Python type hints
- **Automated Generation**: Supports automated API documentation generation

### Negative

- **Verbose Format**: Google-style docstrings are more verbose than other formats
- **Learning Curve**: Developers need to learn the format and required sections
- **Maintenance Overhead**: Need to keep docstrings up-to-date with code changes
- **Enforcement Required**: Need tools to ensure compliance with the format

### Risks

- **Inconsistent Implementation**: Different developers may implement docstrings differently
- **Outdated Documentation**: Docstrings may become outdated as code changes
- **Performance Impact**: Verbose docstrings may impact code readability
- **Tooling Dependencies**: Reliance on specific documentation tools

### Mitigation

- **Comprehensive Examples**: Provide clear examples for all docstring types
- **Automated Checking**: Use tools to validate docstring format and completeness
- **Code Review**: Include docstring quality in code review process
- **Documentation**: Provide comprehensive guidelines and templates

## Implementation

### Phase 1: Standards Definition

- Define Google-style docstring standards
- Create comprehensive templates for all code elements
- Develop docstring validation tools
- Create documentation guidelines

### Phase 2: Existing Code Refactoring

- Refactor coordinator modules to use Google-style docstrings
- Refactor service layer modules
- Refactor orchestration modules
- Update test modules

### Phase 3: Enforcement and Tooling

- Integrate docstring checking into pre-commit hooks
- Add docstring validation to CI pipeline
- Create IDE plugins or extensions for docstring support
- Set up automated documentation generation

## Examples

### Function Docstring

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

    Note:
        Performance: O(n) time complexity where n is text length.
        Memory: O(m) space complexity where m is number of chunks.

    Example:
        >>> chunks = chunk_text("This is a sample document.", "section")
        >>> assert len(chunks) > 0
        >>> assert chunks[0].content == "This is a sample document."
    """
```

### Class Docstring

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

### Dataclass Docstring

```python
@dataclass
class ChunkingRequest:
    """Request for synchronous document chunking operations.

    This dataclass represents a request to chunk a document into smaller
    segments. It contains all the necessary information for the chunking
    operation including document content and chunking parameters.

    Attributes:
        document_id: Unique identifier for the document being chunked.
        text: Optional document text content. If None, text will be extracted
            from the document source. Must be non-empty string if provided.
        strategy: Chunking strategy to use. Valid values: "section", "semantic",
            "paragraph". Defaults to "section".
        chunk_size: Maximum number of tokens per chunk. If None, uses default
            from chunking profile. Must be positive integer if provided.
        overlap: Number of tokens to overlap between consecutive chunks.
            If None, uses default from chunking profile. Must be non-negative
            integer if provided.
        options: Additional metadata and configuration options for the
            chunking operation. May include custom parameters specific to
            the chosen strategy.
    """
    document_id: str
    text: str | None = None
    strategy: str = "section"
    chunk_size: int | None = None
    overlap: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
```

## Validation

### Docstring Format Checking

```python
def check_docstring_format(func: Callable) -> list[str]:
    """Check if function has proper Google-style docstring."""
    violations = []

    if not func.__doc__:
        violations.append("Missing docstring")
        return violations

    docstring = func.__doc__

    # Check for required sections
    required_sections = ["Args", "Returns"]
    for section in required_sections:
        if section not in docstring:
            violations.append(f"Missing {section} section")

    # Check for proper formatting
    if not docstring.startswith(func.__name__):
        violations.append("Docstring should start with function name")

    return violations
```

### Automated Validation

```python
# pyproject.toml
[tool.pydocstyle]
convention = "google"

[tool.ruff.lint]
select = ["D"]  # Enable docstring checks

[tool.ruff.lint.pydocstyle]
convention = "google"
```

### Pre-commit Hook

```yaml
- repo: local
  hooks:
    - id: ruff-docstring-check
      name: Check docstrings with ruff
      entry: ruff check --select D
      language: system
      types: [python]
      files: ^src/Medical_KG_rev/
```

## References

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 257 Docstring Conventions](https://pep257.readthedocs.io/)
- [Sphinx Google Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [mkdocstrings Python Handler](https://mkdocstrings.github.io/python/)
- [pydocstyle](https://pydocstyle.readthedocs.io/)
- [Docstring Templates](https://github.com/Medical_KG_rev/Medical_KG_rev/tree/main/openspec/changes/add-pipeline-structure-documentation/templates)
