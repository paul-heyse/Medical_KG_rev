# ADR-0002: Section Headers

## Status

Accepted

## Context

The Medical KG codebase had inconsistent code organization across modules. Different files used different patterns for organizing imports, classes, functions, and other code elements. This inconsistency made it difficult to:

- Navigate and understand code structure
- Enforce consistent code organization
- Automate code quality checks
- Onboard new developers
- Maintain code readability

The existing issues included:

- Imports scattered throughout files without clear grouping
- Classes and functions in random order
- No clear separation between public and private code
- Inconsistent placement of type definitions and constants
- Missing or inconsistent documentation organization

The system needed a standardized approach to code organization that would:

- Improve code readability and navigation
- Enable automated enforcement of organization rules
- Provide clear patterns for new code
- Support consistent documentation generation
- Facilitate code reviews and maintenance

## Decision

We will implement **Standardized Section Headers** across all pipeline modules. Each module will be organized into clearly labeled sections with consistent ordering rules.

### Section Header Format

```python
# ============================================================================
# SECTION_NAME
# ============================================================================
```

### Standard Sections

#### Gateway Coordinator Modules

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================

# ============================================================================
# PRIVATE HELPERS
# ============================================================================

# ============================================================================
# ERROR TRANSLATION
# ============================================================================

# ============================================================================
# EXPORTS
# ============================================================================
```

#### Service Layer Modules

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

#### Policy/Strategy Modules

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# DATA MODELS
# ============================================================================

# ============================================================================
# INTERFACES (Protocols/ABCs)
# ============================================================================

# ============================================================================
# IMPLEMENTATIONS
# ============================================================================

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

# ============================================================================
# EXPORTS
# ============================================================================
```

#### Orchestration Modules

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# STAGE CONTEXT DATA MODELS
# ============================================================================

# ============================================================================
# STAGE IMPLEMENTATIONS
# ============================================================================

# ============================================================================
# PLUGIN REGISTRATION
# ============================================================================

# ============================================================================
# EXPORTS
# ============================================================================
```

#### Test Modules

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# FIXTURES
# ============================================================================

# ============================================================================
# UNIT TESTS - [ComponentName]
# ============================================================================

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
```

### Ordering Rules Within Sections

- **Imports**: stdlib, third-party, first-party, relative (each group alphabetically sorted)
- **Classes**: Base classes before subclasses, interfaces before implementations
- **Class methods**: `__init__` first, public methods (alphabetically), private methods (alphabetically), static/class methods last
- **Functions**: Public functions before private functions, alphabetical within each group

## Consequences

### Positive

- **Improved Readability**: Clear visual separation of code sections
- **Consistent Navigation**: Developers can quickly find specific types of code
- **Automated Enforcement**: Can be validated with automated tools
- **Better Documentation**: Clear structure supports documentation generation
- **Easier Code Reviews**: Reviewers can focus on specific sections
- **Onboarding Support**: New developers can understand code organization quickly

### Negative

- **Initial Refactoring**: Existing code needs to be reorganized
- **Tooling Overhead**: Need to create and maintain section header checker
- **Strictness**: May feel overly prescriptive for some developers
- **Maintenance**: Need to ensure new code follows the standards

### Risks

- **Over-Engineering**: Risk of making the organization too rigid
- **Tooling Complexity**: Section header checker may be complex to implement
- **Developer Resistance**: Some developers may resist the strict organization
- **Maintenance Burden**: Need to keep the standards updated and enforced

### Mitigation

- **Gradual Implementation**: Implement section headers incrementally
- **Clear Documentation**: Provide comprehensive examples and guidelines
- **Automated Tools**: Create tools to automatically check and fix section headers
- **Team Buy-in**: Ensure team understands the benefits and participates in implementation

## Implementation

### Phase 1: Standards Definition

- Define section header standards for each module type
- Create comprehensive documentation with examples
- Develop section header checker tool

### Phase 2: Existing Code Refactoring

- Refactor coordinator modules to use section headers
- Refactor service layer modules
- Refactor orchestration modules
- Update test modules

### Phase 3: Enforcement and Tooling

- Integrate section header checker into pre-commit hooks
- Add section header validation to CI pipeline
- Create IDE plugins or extensions for section header support

## Examples

### Before (Poor Organization)

```python
import os
from typing import Dict, List
from Medical_KG_rev.gateway.models import DocumentChunk
from Medical_KG_rev.observability.metrics import record_chunking_failure
import logging

logger = logging.getLogger(__name__)

class ChunkingCoordinator:
    def __init__(self, chunker):
        self._chunker = chunker

    def _extract_text(self, request):
        return request.text

    def execute(self, request):
        # Implementation here
        pass

    def _translate_error(self, exc):
        # Implementation here
        pass

@dataclass
class ChunkingRequest:
    document_id: str
    text: str
```

### After (Good Organization)

```python
# ============================================================================
# IMPORTS
# ============================================================================
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from Medical_KG_rev.gateway.coordinators.base import BaseCoordinator, CoordinatorConfig
from Medical_KG_rev.gateway.coordinators.job_lifecycle import JobLifecycleManager
from Medical_KG_rev.gateway.models import DocumentChunk
from Medical_KG_rev.observability.metrics import record_chunking_failure

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

@dataclass
class ChunkingRequest(CoordinatorRequest):
    """Request for synchronous document chunking operations."""
    document_id: str
    text: str | None = None
    strategy: str = "section"
    chunk_size: int | None = None
    overlap: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkingResult(CoordinatorResult):
    """Result of synchronous document chunking operations."""
    chunks: Sequence[DocumentChunk] = ()

# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================

class ChunkingCoordinator(BaseCoordinator[ChunkingRequest, ChunkingResult]):
    """Coordinates synchronous chunking operations."""

    def __init__(self, lifecycle: JobLifecycleManager, chunker: ChunkingService, config: CoordinatorConfig) -> None:
        super().__init__(config)
        self._lifecycle = lifecycle
        self._chunker = chunker

    def _execute(self, request: ChunkingRequest, **kwargs: Any) -> ChunkingResult:
        """Execute chunking operation with job lifecycle management."""
        # Implementation here
        pass

# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _extract_text(job_id: str, request: ChunkingRequest) -> str:
    """Extract document text from request."""
    # Implementation here
    pass

# ============================================================================
# ERROR TRANSLATION
# ============================================================================

def _translate_error(job_id: str, request: ChunkingRequest, exc: Exception) -> CoordinatorError:
    """Translate chunking exceptions to coordinator errors."""
    # Implementation here
    pass

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["ChunkingCoordinator", "ChunkingRequest", "ChunkingResult"]
```

## Validation

### Automated Checking

```python
def check_section_headers(file_path: str) -> list[str]:
    """Check if file has proper section headers."""
    violations = []

    with open(file_path, 'r') as f:
        content = f.read()

    # Check for required sections
    required_sections = ["IMPORTS", "EXPORTS"]
    for section in required_sections:
        if f"# {section}" not in content:
            violations.append(f"Missing {section} section")

    # Check section order
    # Implementation here

    return violations
```

### Pre-commit Hook

```yaml
- repo: local
  hooks:
    - id: section-header-check
      name: Check section headers
      entry: python scripts/check_section_headers.py
      language: system
      types: [python]
      files: ^src/Medical_KG_rev/(gateway|services|orchestration)/
```

## References

- [Python Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Code Organization Best Practices](https://docs.python.org/3/tutorial/modules.html)
- [Section Header Standards](https://github.com/Medical_KG_rev/Medical_KG_rev/blob/main/openspec/changes/add-pipeline-structure-documentation/section_headers.md)
