# ADR-0008: Type Hint Modernization

## Status

**Accepted** - 2024-01-15

## Context

As part of the repository-wide documentation standards (ADR-0005), we need to modernize type hints throughout the Medical_KG_rev repository to use modern Python conventions. The current state shows inconsistent type hint usage, with some modules using modern syntax while others use deprecated patterns.

Modern Python type hints provide better tooling support, improved readability, and alignment with current Python best practices. The repository contains 360+ Python files with varying levels of type hint coverage and consistency.

## Decision

We will modernize all type hints throughout the repository to use modern Python conventions:

1. **Union Syntax**: Use `Type | None` instead of `Optional[Type]`
2. **Collection Generics**: Use `Mapping`, `Sequence` from `collections.abc` instead of `dict`, `list`
3. **Complete Annotations**: Ensure all public functions and classes have complete type annotations
4. **Modern Generic Syntax**: Use modern generic syntax for type parameters
5. **Type Variable Usage**: Use proper type variables for generic functions and classes
6. **Protocol Usage**: Use `Protocol` for structural subtyping where appropriate

## Implementation Details

### Type Hint Modernization Rules

#### Union Syntax

```python
# Old (deprecated)
from typing import Optional, Union
def process_data(data: Optional[str]) -> Optional[dict]:
    pass

# New (modern)
def process_data(data: str | None) -> dict | None:
    pass

# Complex unions
def handle_value(value: str | int | float | None) -> bool:
    pass
```

#### Collection Generics

```python
# Old (deprecated)
from typing import Dict, List, Tuple
def process_items(items: List[str]) -> Dict[str, int]:
    pass

# New (modern)
from collections.abc import Mapping, Sequence
def process_items(items: Sequence[str]) -> Mapping[str, int]:
    pass

# Specific types when needed
def process_dict(data: dict[str, int]) -> list[str]:
    pass
```

#### Complete Annotations

```python
# Old (incomplete)
def calculate_score(data):
    return data.get("score", 0)

# New (complete)
def calculate_score(data: Mapping[str, Any]) -> int:
    return data.get("score", 0)

# With proper return type
def process_documents(docs: Sequence[Document]) -> list[ProcessedDocument]:
    return [process_doc(doc) for doc in docs]
```

#### Modern Generic Syntax

```python
# Old (deprecated)
from typing import TypeVar, Generic
T = TypeVar('T')
class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

# New (modern)
from typing import TypeVar
T = TypeVar('T')
class Container[T]:
    def __init__(self, value: T) -> None:
        self.value = value
```

#### Protocol Usage

```python
# Old (ABC approach)
from abc import ABC, abstractmethod
class Processor(ABC):
    @abstractmethod
    def process(self, data: str) -> str:
        pass

# New (Protocol approach)
from typing import Protocol
class Processor(Protocol):
    def process(self, data: str) -> str:
        ...
```

### Domain-Specific Type Hints

#### Gateway Modules

```python
from typing import Any
from collections.abc import Mapping, Sequence
from Medical_KG_rev.models import Document, Request, Response

# Request/Response models
def handle_request(request: Request) -> Response:
    pass

# Batch operations
def process_batch(requests: Sequence[Request]) -> list[Response]:
    pass

# Configuration
def load_config(config: Mapping[str, Any]) -> None:
    pass
```

#### Service Modules

```python
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence, Callable
from Medical_KG_rev.models import Document, Embedding

# Generic service operations
T = TypeVar('T')
def process_items(items: Sequence[T], processor: Callable[[T], T]) -> list[T]:
    pass

# Service-specific types
def embed_documents(docs: Sequence[Document]) -> list[Embedding]:
    pass

# Configuration and metadata
def configure_service(config: Mapping[str, Any]) -> None:
    pass
```

#### Adapter Modules

```python
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence, Awaitable
from Medical_KG_rev.adapters.base import AdapterResult

# Adapter operations
T = TypeVar('T')
async def fetch_data(adapter: Callable[[str], Awaitable[AdapterResult[T]]],
                    identifier: str) -> AdapterResult[T]:
    pass

# Batch adapter operations
async def fetch_batch(adapter: Callable[[Sequence[str]], Awaitable[list[AdapterResult[T]]]],
                     identifiers: Sequence[str]) -> list[AdapterResult[T]]:
    pass
```

#### Orchestration Modules

```python
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence, Callable
from Medical_KG_rev.orchestration.ledger import JobLedgerEntry

# Stage operations
T = TypeVar('T')
def execute_stage(stage: Callable[[T], T], data: T) -> T:
    pass

# Job management
def transition_job(job_id: str, new_state: str) -> JobLedgerEntry:
    pass

# Event handling
def emit_event(event_type: str, data: Mapping[str, Any]) -> None:
    pass
```

#### Knowledge Graph Modules

```python
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence
from Medical_KG_rev.kg.schema import NodeSchema, RelationshipSchema

# Graph operations
def create_node(label: str, properties: Mapping[str, Any]) -> dict[str, Any]:
    pass

def create_relationship(from_node: str, to_node: str,
                      rel_type: str, properties: Mapping[str, Any]) -> dict[str, Any]:
    pass

# Schema management
def apply_schema(nodes: Sequence[NodeSchema],
                relationships: Sequence[RelationshipSchema]) -> None:
    pass
```

#### Storage Modules

```python
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence
from Medical_KG_rev.storage.base import ObjectMetadata

# Storage operations
T = TypeVar('T')
def store_object(key: str, data: T, metadata: Mapping[str, Any]) -> ObjectMetadata:
    pass

def retrieve_object(key: str) -> T | None:
    pass

# Batch operations
def store_batch(items: Sequence[tuple[str, T, Mapping[str, Any]]]) -> list[ObjectMetadata]:
    pass
```

#### Validation Modules

```python
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence
from Medical_KG_rev.validation.base import ValidationResult

# Validation operations
T = TypeVar('T')
def validate_data(data: T, schema: Mapping[str, Any]) -> ValidationResult:
    pass

def validate_batch(items: Sequence[T], schema: Mapping[str, Any]) -> list[ValidationResult]:
    pass

# FHIR validation
def validate_fhir_resource(resource_type: str, data: Mapping[str, Any]) -> ValidationResult:
    pass
```

#### Utility Modules

```python
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence, Callable
from Medical_KG_rev.utils.errors import MedicalKGError

# Generic utility functions
T = TypeVar('T')
def safe_execute(func: Callable[[], T], default: T) -> T:
    pass

def retry_operation(func: Callable[[], T], max_retries: int) -> T:
    pass

# HTTP client operations
def make_request(url: str, headers: Mapping[str, str]) -> dict[str, Any]:
    pass
```

### Migration Strategy

#### Phase 1: Analysis and Planning

1. **Type Hint Audit**: Analyze current type hint usage across all modules
2. **Deprecated Pattern Detection**: Identify all deprecated type hint patterns
3. **Modernization Plan**: Create detailed plan for each module
4. **Tool Configuration**: Configure type hint validation tools

#### Phase 2: Automated Migration

1. **Pattern Detection**: Use AST analysis to identify deprecated patterns
2. **Automated Conversion**: Convert mechanical patterns automatically
3. **Validation**: Verify converted type hints are correct
4. **Manual Review**: Human review of automated changes

#### Phase 3: Manual Modernization

1. **Complex Patterns**: Manually modernize complex type hint patterns
2. **Missing Annotations**: Add missing type annotations
3. **Protocol Implementation**: Implement Protocol-based interfaces
4. **Generic Type Variables**: Add proper type variables for generic functions

#### Phase 4: Validation and Testing

1. **Type Checker Validation**: Run mypy on all modules
2. **Runtime Testing**: Ensure type hints don't affect runtime behavior
3. **Performance Testing**: Verify no performance impact
4. **Documentation Updates**: Update documentation with new type hints

### Validation Tools

#### Type Hint Checker

```python
#!/usr/bin/env python3
"""Validate modern type hint usage.

Checks:
- Modern union syntax (Type | None vs Optional[Type])
- Collection generics (Mapping vs dict)
- Complete parameter annotations
- Complete return type annotations
- Generic type parameters
- Protocol usage

Fixes:
- Auto-convert Optional to union syntax
- Auto-convert dict/list to Mapping/Sequence
- Generate missing type annotations (suggestions)
"""
```

#### Mypy Configuration

```ini
# mypy.ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
show_error_codes = True
show_column_numbers = True
show_error_context = True
```

## Consequences

### Positive

- **Better Tooling Support**: Modern type hints work better with IDEs and type checkers
- **Improved Readability**: Modern syntax is more readable and concise
- **Alignment with Standards**: Follows current Python best practices
- **Better Error Messages**: Modern type hints provide clearer error messages
- **Future Compatibility**: Aligns with future Python versions

### Negative

- **Migration Overhead**: Time required to modernize existing type hints
- **Learning Curve**: Developers need to learn modern type hint syntax
- **Tooling Updates**: May require updates to development tools
- **Compatibility Issues**: Some tools may not support all modern features

### Risks and Mitigations

- **Risk**: Modern type hints may not be supported by all tools
  - **Mitigation**: Test with all development tools, provide fallbacks
- **Risk**: Migration may introduce type errors
  - **Mitigation**: Comprehensive testing, gradual migration
- **Risk**: Performance impact of type hints
  - **Mitigation**: Type hints are erased at runtime, no performance impact
- **Risk**: Complex type hints may be harder to understand
  - **Mitigation**: Provide documentation and examples

## Alternatives Considered

### Alternative 1: Keep Existing Type Hints

- **Description**: Maintain current type hint style without modernization
- **Rejected**: Deprecated patterns will become unsupported
- **Reason**: Modern Python type hints provide better tooling and readability

### Alternative 2: Gradual Modernization

- **Description**: Modernize type hints incrementally over time
- **Rejected**: Would create inconsistency and technical debt
- **Reason**: Gradual approach leads to mixed patterns and confusion

### Alternative 3: Remove Type Hints

- **Description**: Remove all type hints to avoid complexity
- **Rejected**: Type hints provide significant value for development
- **Reason**: Type hints improve code quality, tooling, and maintainability

## Success Metrics

- **Modern Syntax Usage**: 100% of type hints use modern syntax
- **Complete Coverage**: 100% of public functions and classes have type annotations
- **Type Checker Compliance**: 0 errors from mypy type checking
- **Tool Compatibility**: All development tools work with modern type hints
- **Developer Satisfaction**: Positive feedback on improved type hints
- **Performance Impact**: Zero performance impact from type hints

## Implementation Timeline

- **Week 1**: Analyze current type hint usage and create migration plan
- **Week 2**: Configure type hint validation tools
- **Week 3**: Implement automated migration tools
- **Week 4**: Apply automated migration to Gateway modules
- **Week 5**: Apply automated migration to Service modules
- **Week 6**: Apply automated migration to Adapter modules
- **Week 7**: Apply automated migration to Orchestration modules
- **Week 8**: Apply automated migration to remaining modules
- **Week 9**: Manual modernization of complex patterns
- **Week 10**: Validation and testing of modernized type hints
- **Week 11**: Documentation updates and final validation

## References

- [Whole Repository Structure Documentation Proposal](../openspec/changes/whole-repo-structure-documentation/proposal.md)
- [Whole Repository Structure Documentation Design](../openspec/changes/whole-repo-structure-documentation/design.md)
- [Python Type Hints Documentation](https://docs.python.org/3/library/typing.html)
- [Mypy Documentation](https://mypy.readthedocs.io/)

## Related ADRs

- ADR-0005: Repository-Wide Documentation Standards
- ADR-0006: Domain-Specific Section Headers
- ADR-0007: Automated Documentation Enforcement
