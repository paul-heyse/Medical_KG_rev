# Section Header Standards for Pipeline Modules

This document defines the canonical section headers and ordering rules for all pipeline modules in the Medical_KG_rev codebase.

## Overview

Section headers provide consistent code organization and make modules easier to navigate and understand. Each module type has specific section headers that must appear in a defined order.

## Section Header Format

All section headers must use the following format:

```python
# ============================================================================
# SECTION_NAME
# ============================================================================
```

The section name should be in UPPERCASE with underscores separating words.

## Module Types and Required Sections

### 1. Gateway Coordinator Modules

**Required Sections (in order):**

1. `IMPORTS`
2. `REQUEST/RESPONSE MODELS`
3. `COORDINATOR IMPLEMENTATION`
4. `ERROR TRANSLATION`
5. `EXPORTS`

**Example Structure:**

```python
"""Module docstring explaining coordinator purpose and responsibilities."""

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
# ERROR TRANSLATION
# ============================================================================
# (Methods for translating exceptions to coordinator errors)

# ============================================================================
# EXPORTS
# ============================================================================
# (__all__ list)
```

### 2. Gateway Service Layer Modules

**Required Sections (in order):**

1. `IMPORTS`
2. `TYPE DEFINITIONS & CONSTANTS`
3. `SERVICE CLASS DEFINITION`
4. `INITIALIZATION & SETUP`
5. `CHUNKING ENDPOINTS`
6. `EMBEDDING ENDPOINTS`
7. `RETRIEVAL ENDPOINTS`
8. `ADAPTER MANAGEMENT ENDPOINTS`
9. `VALIDATION ENDPOINTS`
10. `EXTRACTION ENDPOINTS`
11. `ADMIN & UTILITY ENDPOINTS`
12. `PRIVATE HELPERS`
13. `EXPORTS`

**Example Structure:**

```python
"""Module docstring explaining service layer purpose and responsibilities."""

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
# INITIALIZATION & SETUP
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

# ============================================================================
# EXPORTS
# ============================================================================
```

### 3. Policy/Strategy Modules

**Required Sections (in order):**

1. `IMPORTS`
2. `DATA MODELS`
3. `INTERFACES (Protocols/ABCs)`
4. `IMPLEMENTATIONS`
5. `FACTORY FUNCTIONS`
6. `EXPORTS`

**Example Structure:**

```python
"""Module docstring explaining policy system purpose and responsibilities."""

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

### 4. Orchestration Modules

**Required Sections (in order):**

1. `IMPORTS`
2. `STAGE CONTEXT DATA MODELS`
3. `STAGE IMPLEMENTATIONS`
4. `PLUGIN REGISTRATION`
5. `EXPORTS`

**Example Structure:**

```python
"""Module docstring explaining orchestration purpose and responsibilities."""

# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# STAGE CONTEXT DATA MODELS
# ============================================================================

# ============================================================================
# STAGE IMPLEMENTATIONS
# ============================================================================
# (Grouped by pipeline phase: metadata, PDF, chunk, embed, index)

# ============================================================================
# PLUGIN REGISTRATION
# ============================================================================

# ============================================================================
# EXPORTS
# ============================================================================
```

### 5. Test Modules

**Required Sections (in order):**

1. `IMPORTS`
2. `FIXTURES`
3. `UNIT TESTS - [ComponentName]`
4. `INTEGRATION TESTS`
5. `HELPER FUNCTIONS`

**Example Structure:**

```python
"""Module docstring explaining what component is under test."""

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

## Section Content Rules

### IMPORTS Section

- **Content**: Only import statements
- **Ordering**:
  1. Standard library imports (alphabetically sorted)
  2. Third-party imports (alphabetically sorted)
  3. First-party imports from Medical_KG_rev (alphabetically sorted)
  4. Relative imports (alphabetically sorted)
- **Formatting**: One import per line, grouped by category with blank lines between groups

### REQUEST/RESPONSE MODELS Section

- **Content**: Dataclasses, Pydantic models, and other data structures used for request/response
- **Ordering**: Request models before response models, alphabetically within each group
- **Formatting**: Each model should have comprehensive docstring

### COORDINATOR IMPLEMENTATION Section

- **Content**: Main coordinator class with all its methods
- **Ordering**:
  1. `__init__` method first
  2. Public methods (alphabetically sorted)
  3. Private methods (alphabetically sorted)
  4. Static/class methods last
- **Formatting**: Each method should have comprehensive docstring

### ERROR TRANSLATION Section

- **Content**: Methods for translating exceptions to coordinator errors
- **Ordering**: Alphabetically sorted by method name
- **Formatting**: Each method should have comprehensive docstring

### SERVICE CLASS DEFINITION Section

- **Content**: Main service class with all its methods
- **Ordering**: Same as coordinator implementation
- **Formatting**: Each method should have comprehensive docstring

### ENDPOINT Sections

- **Content**: Methods that handle specific types of endpoints
- **Ordering**: Alphabetically sorted by method name within each section
- **Formatting**: Each method should have comprehensive docstring

### PRIVATE HELPERS Section

- **Content**: Private methods used by the main class
- **Ordering**: Alphabetically sorted by method name
- **Formatting**: Each method should have comprehensive docstring

### EXPORTS Section

- **Content**: `__all__` list defining public API
- **Ordering**: Alphabetically sorted
- **Formatting**: One item per line

## Ordering Rules Within Sections

### Import Ordering

```python
# Standard library imports (alphabetically sorted)
import json
import time
from typing import Any, Dict, List

# Third-party imports (alphabetically sorted)
import structlog
from pydantic import BaseModel

# First-party imports (alphabetically sorted)
from Medical_KG_rev.gateway.models import DocumentChunk
from Medical_KG_rev.observability.metrics import record_chunking_failure
from Medical_KG_rev.services.chunking import ChunkingService

# Relative imports (alphabetically sorted)
from .base import BaseCoordinator
from .errors import ChunkingErrorTranslator
```

### Class Method Ordering

```python
class MyCoordinator:
    def __init__(self, ...):  # Constructor first
        pass

    def execute(self, ...):   # Public methods alphabetically
        pass

    def _helper_method(self, ...):  # Private methods alphabetically
        pass

    @classmethod
    def from_config(cls, ...):  # Class methods last
        pass

    @staticmethod
    def utility_function(...):  # Static methods last
        pass
```

### Function Ordering

```python
# Public functions first (alphabetically)
def public_function():
    pass

def another_public_function():
    pass

# Private functions last (alphabetically)
def _private_function():
    pass

def _another_private_function():
    pass
```

## Validation Rules

### Required Sections

- All required sections must be present
- Sections must appear in the correct order
- Section headers must use exact format with equals signs

### Section Content

- Each section must contain appropriate content
- IMPORTS section must contain only import statements
- EXPORTS section must contain only `__all__` list
- Other sections must contain relevant code

### Ordering Within Sections

- Imports must be grouped and sorted correctly
- Class methods must follow ordering rules
- Functions must be sorted alphabetically

## Examples

### Complete Coordinator Module

```python
"""Chunking coordinator for synchronous document chunking operations.

This module provides the ChunkingCoordinator class that coordinates synchronous
chunking jobs by managing job lifecycle, delegating to ChunkingService, and
translating chunking exceptions to coordinator errors.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import time
from typing import Any, Dict, List

import structlog
from pydantic import BaseModel

from Medical_KG_rev.gateway.models import DocumentChunk
from Medical_KG_rev.observability.metrics import record_chunking_failure
from Medical_KG_rev.services.chunking import ChunkingService

from .base import BaseCoordinator
from .errors import ChunkingErrorTranslator

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChunkingRequest(BaseModel):
    """Request model for synchronous chunking operations."""
    tenant_id: str
    document_id: str
    text: str | None = None
    strategy: str = "section"
    chunk_size: int = 512
    overlap: int = 128
    options: Dict[str, Any] = {}

class ChunkingResult(BaseModel):
    """Result model for synchronous chunking operations."""
    job_id: str
    duration_s: float
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]

# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================

class ChunkingCoordinator(BaseCoordinator[ChunkingRequest, ChunkingResult]):
    """Chunking coordinator for synchronous document chunking operations."""

    def __init__(self, lifecycle, chunker, config, errors=None):
        """Initialize chunking coordinator."""
        super().__init__(lifecycle, config)
        self._chunker = chunker
        self._errors = errors or ChunkingErrorTranslator()

    def execute(self, request: ChunkingRequest) -> ChunkingResult:
        """Execute synchronous chunking operation."""
        # Implementation here
        pass

# ============================================================================
# ERROR TRANSLATION
# ============================================================================

def _translate_error(self, job_id: str, command: ChunkCommand, exc: Exception) -> CoordinatorError:
    """Translate chunking exceptions to coordinator errors."""
    # Implementation here
    pass

# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    "ChunkingCoordinator",
    "ChunkingRequest",
    "ChunkingResult",
]
```

## Enforcement

Section headers are enforced by:

1. **Static Analysis**: Custom AST-based checker validates section presence and ordering
2. **Pre-commit Hooks**: Automatic validation on every commit
3. **CI Pipeline**: Validation in continuous integration
4. **Documentation Generation**: Validation during API documentation generation

### Validation Commands

```bash
# Check section headers
python scripts/check_section_headers.py

# Check with specific module type
python scripts/check_section_headers.py --module-type coordinator

# Check specific file
python scripts/check_section_headers.py src/Medical_KG_rev/gateway/coordinators/chunking.py
```

### Common Validation Errors

1. **Missing Section**: "Section 'IMPORTS' is missing in chunking.py"
2. **Wrong Order**: "Section 'ERROR TRANSLATION' appears before 'COORDINATOR IMPLEMENTATION' in chunking.py:150"
3. **Invalid Content**: "IMPORTS section contains non-import statements in chunking.py:25"
4. **Missing Header**: "Section header format invalid in chunking.py:30"

## Migration Guide

When adding section headers to existing modules:

1. **Identify Module Type**: Determine which module type the file belongs to
2. **Add Section Headers**: Insert section headers in the correct order
3. **Reorganize Content**: Move code into appropriate sections
4. **Validate**: Run section header checker to verify compliance
5. **Test**: Ensure functionality remains unchanged

### Example Migration

**Before:**

```python
"""Chunking coordinator."""

import time
from typing import Any, Dict, List

class ChunkingRequest(BaseModel):
    tenant_id: str

class ChunkingCoordinator:
    def __init__(self):
        pass

def _translate_error():
    pass

__all__ = ["ChunkingCoordinator"]
```

**After:**

```python
"""Chunking coordinator for synchronous document chunking operations."""

# ============================================================================
# IMPORTS
# ============================================================================
import time
from typing import Any, Dict, List

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChunkingRequest(BaseModel):
    tenant_id: str

# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================

class ChunkingCoordinator:
    def __init__(self):
        pass

# ============================================================================
# ERROR TRANSLATION
# ============================================================================

def _translate_error():
    pass

# ============================================================================
# EXPORTS
# ============================================================================
__all__ = ["ChunkingCoordinator"]
```
