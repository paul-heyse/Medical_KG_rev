# Section Header Standards for Medical_KG_rev Repository

This document defines the canonical section header standards for all module types in the Medical_KG_rev repository, extending the successful pipeline documentation standards to the entire codebase.

## General Principles

1. **Consistency**: All modules follow the same section ordering principles
2. **Clarity**: Section headers clearly indicate their purpose and contents
3. **Completeness**: All relevant sections are included for each module type
4. **Ordering**: Sections follow a logical flow from imports to exports

## Section Header Format

All section headers use the following format:

```python
# ==============================================================================
# SECTION NAME
# ==============================================================================
```

## Gateway Modules

**Required Sections (in order):**

1. `IMPORTS` - All import statements
2. `TYPE DEFINITIONS` - Type aliases and type definitions
3. `REQUEST/RESPONSE MODELS` - Pydantic models for API requests/responses
4. `COORDINATOR IMPLEMENTATION` - Main coordinator classes
5. `ERROR TRANSLATION` - Error translation functions
6. `FACTORY FUNCTIONS` - Factory functions for creating instances
7. `EXPORTS` - `__all__` list

**Example:**

```python
# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
import asyncio
from typing import Any

# Third-party imports
from fastapi import FastAPI
from pydantic import BaseModel

# First-party imports
from Medical_KG_rev.services import EmbeddingService
from Medical_KG_rev.models import Document

# Relative imports
from .base import BaseCoordinator

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type ConfigDict = dict[str, Any]
type ResultList = list[dict[str, Any]]

# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================
class IngestionRequest(BaseModel):
    """Request model for ingestion operations."""
    pass

# ==============================================================================
# COORDINATOR IMPLEMENTATION
# ==============================================================================
class IngestionCoordinator(BaseCoordinator):
    """Coordinator for ingestion operations."""
    pass

# ==============================================================================
# ERROR TRANSLATION
# ==============================================================================
def translate_ingestion_error(exc: Exception) -> ProblemDetail:
    """Translate domain exceptions to HTTP problem details."""
    pass

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_coordinator() -> IngestionCoordinator:
    """Create and configure an ingestion coordinator."""
    pass

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "IngestionCoordinator",
    "IngestionRequest",
    "translate_ingestion_error",
    "create_coordinator",
]
```

## Service Modules

**Required Sections (in order):**

1. `IMPORTS` - All import statements
2. `TYPE DEFINITIONS` - Type aliases and type definitions
3. `DATA MODELS` - Pydantic models, dataclasses, and data structures
4. `INTERFACES` - Protocols, abstract base classes, and interfaces
5. `IMPLEMENTATIONS` - Concrete implementations of interfaces
6. `FACTORY FUNCTIONS` - Factory functions for creating instances
7. `HELPER FUNCTIONS` - Private helper functions
8. `EXPORTS` - `__all__` list

**Example:**

```python
# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

# Third-party imports
import httpx
from pydantic import BaseModel

# First-party imports
from Medical_KG_rev.adapters import BaseAdapter
from Medical_KG_rev.models import Document
from Medical_KG_rev.utils import logger

# Relative imports
from .base import BaseService

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type EmbeddingVector = list[float]
type EmbeddingMetadata = dict[str, Any]

# ==============================================================================
# DATA MODELS
# ==============================================================================
class EmbeddingRequest(BaseModel):
    """Request model for embedding operations."""
    text: str
    model: str
    metadata: EmbeddingMetadata | None = None

class EmbeddingResponse(BaseModel):
    """Response model for embedding operations."""
    embedding: EmbeddingVector
    model: str
    metadata: EmbeddingMetadata

# ==============================================================================
# INTERFACES
# ==============================================================================
class EmbeddingServiceProtocol(Protocol):
    """Protocol for embedding service implementations."""

    async def embed_text(self, text: str, model: str) -> EmbeddingResponse:
        """Embed text using specified model."""
        ...

# ==============================================================================
# IMPLEMENTATIONS
# ==============================================================================
class EmbeddingService(BaseService):
    """Concrete implementation of embedding service."""

    async def embed_text(self, text: str, model: str) -> EmbeddingResponse:
        """Embed text using specified model."""
        pass

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_embedding_service(config: dict[str, Any]) -> EmbeddingService:
    """Create and configure an embedding service."""
    pass

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _validate_embedding_model(model: str) -> bool:
    """Validate embedding model name."""
    pass

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "EmbeddingService",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "create_embedding_service",
]
```

## Adapter Modules

**Required Sections (in order):**

1. `IMPORTS` - All import statements
2. `TYPE DEFINITIONS` - Type aliases and type definitions
3. `DATA MODELS` - Request/response models and data structures
4. `ADAPTER IMPLEMENTATION` - Main adapter class with fetch/parse/validate methods
5. `ERROR HANDLING` - Error translation and handling functions
6. `FACTORY FUNCTIONS` - Factory functions for creating adapter instances
7. `HELPER FUNCTIONS` - Private helper functions
8. `EXPORTS` - `__all__` list

**Example:**

```python
# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
import asyncio
from collections.abc import Mapping
from typing import Any

# Third-party imports
import httpx
from pydantic import BaseModel

# First-party imports
from Medical_KG_rev.adapters.base import BaseAdapter
from Medical_KG_rev.models import Document
from Medical_KG_rev.utils import logger

# Relative imports
from .errors import AdapterError

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type ApiResponse = dict[str, Any]
type AdapterConfig = Mapping[str, Any]

# ==============================================================================
# DATA MODELS
# ==============================================================================
class OpenAlexRequest(BaseModel):
    """Request model for OpenAlex API calls."""
    doi: str
    include_abstract: bool = True

class OpenAlexResponse(BaseModel):
    """Response model for OpenAlex API calls."""
    title: str
    authors: list[str]
    abstract: str | None = None

# ==============================================================================
# ADAPTER IMPLEMENTATION
# ==============================================================================
class OpenAlexAdapter(BaseAdapter):
    """Adapter for OpenAlex API integration."""

    async def fetch(self, request: OpenAlexRequest) -> OpenAlexResponse:
        """Fetch data from OpenAlex API."""
        pass

    def parse(self, response: ApiResponse) -> OpenAlexResponse:
        """Parse API response into structured data."""
        pass

    def validate(self, data: OpenAlexResponse) -> bool:
        """Validate parsed data."""
        pass

# ==============================================================================
# ERROR HANDLING
# ==============================================================================
def translate_openalex_error(exc: Exception) -> AdapterError:
    """Translate OpenAlex API errors to adapter errors."""
    pass

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_openalex_adapter(config: AdapterConfig) -> OpenAlexAdapter:
    """Create and configure an OpenAlex adapter."""
    pass

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _build_api_url(doi: str) -> str:
    """Build OpenAlex API URL for DOI."""
    pass

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "OpenAlexAdapter",
    "OpenAlexRequest",
    "OpenAlexResponse",
    "create_openalex_adapter",
]
```

## Orchestration Modules

**Required Sections (in order):**

1. `IMPORTS` - All import statements
2. `TYPE DEFINITIONS` - Type aliases and type definitions
3. `STAGE CONTEXT DATA MODELS` - Data models for stage context and state
4. `STAGE IMPLEMENTATIONS` - Concrete stage implementations
5. `PLUGIN REGISTRATION` - Plugin registration and management
6. `FACTORY FUNCTIONS` - Factory functions for creating stage instances
7. `HELPER FUNCTIONS` - Private helper functions
8. `EXPORTS` - `__all__` list

**Example:**

```python
# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
import asyncio
from collections.abc import Sequence
from typing import Any

# Third-party imports
import httpx
from pydantic import BaseModel

# First-party imports
from Medical_KG_rev.orchestration.stages.contracts import StageContract
from Medical_KG_rev.models import Document
from Medical_KG_rev.utils import logger

# Relative imports
from .base import BaseStage

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type StageContext = dict[str, Any]
type StageResult = dict[str, Any]

# ==============================================================================
# STAGE CONTEXT DATA MODELS
# ==============================================================================
class PDFDownloadContext(BaseModel):
    """Context data for PDF download stage."""
    document_id: str
    pdf_url: str
    download_path: str

class PDFDownloadResult(BaseModel):
    """Result data for PDF download stage."""
    success: bool
    file_path: str | None = None
    error_message: str | None = None

# ==============================================================================
# STAGE IMPLEMENTATIONS
# ==============================================================================
class PDFDownloadStage(BaseStage):
    """Stage for downloading PDF documents."""

    async def execute(self, context: PDFDownloadContext) -> PDFDownloadResult:
        """Execute PDF download stage."""
        pass

# ==============================================================================
# PLUGIN REGISTRATION
# ==============================================================================
def register_pdf_download_stage() -> None:
    """Register PDF download stage with orchestration system."""
    pass

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_pdf_download_stage(config: dict[str, Any]) -> PDFDownloadStage:
    """Create and configure a PDF download stage."""
    pass

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _validate_pdf_url(url: str) -> bool:
    """Validate PDF URL format."""
    pass

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "PDFDownloadStage",
    "PDFDownloadContext",
    "PDFDownloadResult",
    "create_pdf_download_stage",
]
```

## Knowledge Graph Modules

**Required Sections (in order):**

1. `IMPORTS` - All import statements
2. `TYPE DEFINITIONS` - Type aliases and type definitions
3. `SCHEMA DATA MODELS` - Graph schema definitions and data models
4. `CLIENT IMPLEMENTATION` - Database client implementations
5. `TEMPLATES` - Query templates and utilities
6. `FACTORY FUNCTIONS` - Factory functions for creating client instances
7. `HELPER FUNCTIONS` - Private helper functions
8. `EXPORTS` - `__all__` list

**Example:**

```python
# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

# Third-party imports
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel

# First-party imports
from Medical_KG_rev.models import Document, Entity
from Medical_KG_rev.utils import logger

# Relative imports
from .base import BaseKGClient

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type NodeProperties = dict[str, Any]
type RelationshipProperties = dict[str, Any]

# ==============================================================================
# SCHEMA DATA MODELS
# ==============================================================================
class GraphSchema(BaseModel):
    """Graph schema definition."""
    nodes: list[str]
    relationships: list[str]
    constraints: list[str]

class NodeModel(BaseModel):
    """Base model for graph nodes."""
    id: str
    labels: list[str]
    properties: NodeProperties

# ==============================================================================
# CLIENT IMPLEMENTATION
# ==============================================================================
class Neo4jClient(BaseKGClient):
    """Neo4j client for graph operations."""

    async def create_node(self, model: NodeModel) -> None:
        """Create a node in the graph."""
        pass

    async def create_relationship(self, from_id: str, to_id: str, rel_type: str) -> None:
        """Create a relationship between nodes."""
        pass

# ==============================================================================
# TEMPLATES
# ==============================================================================
def get_cypher_template(template_name: str) -> str:
    """Get Cypher query template by name."""
    pass

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_neo4j_client(uri: str, auth: tuple[str, str]) -> Neo4jClient:
    """Create and configure a Neo4j client."""
    pass

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _validate_cypher_query(query: str) -> bool:
    """Validate Cypher query syntax."""
    pass

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "Neo4jClient",
    "GraphSchema",
    "NodeModel",
    "create_neo4j_client",
]
```

## Storage Modules

**Required Sections (in order):**

1. `IMPORTS` - All import statements
2. `TYPE DEFINITIONS` - Type aliases and type definitions
3. `DATA MODELS` - Storage-specific data models and structures
4. `INTERFACES` - Storage interface protocols and abstract classes
5. `IMPLEMENTATIONS` - Concrete storage implementations
6. `FACTORY FUNCTIONS` - Factory functions for creating storage instances
7. `HELPER FUNCTIONS` - Private helper functions
8. `EXPORTS` - `__all__` list

**Example:**

```python
# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

# Third-party imports
import faiss
from pydantic import BaseModel

# First-party imports
from Medical_KG_rev.models import Document
from Medical_KG_rev.utils import logger

# Relative imports
from .base import BaseStorage

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type Vector = list[float]
type VectorMetadata = dict[str, Any]

# ==============================================================================
# DATA MODELS
# ==============================================================================
class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    index_path: str
    dimension: int
    index_type: str = "faiss"

class VectorSearchResult(BaseModel):
    """Result of vector search operation."""
    vector_id: str
    similarity: float
    metadata: VectorMetadata

# ==============================================================================
# INTERFACES
# ==============================================================================
class VectorStoreProtocol(Protocol):
    """Protocol for vector store implementations."""

    async def store_vectors(self, vectors: Sequence[Vector], metadata: Sequence[VectorMetadata]) -> None:
        """Store vectors with metadata."""
        ...

    async def search_vectors(self, query_vector: Vector, top_k: int) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        ...

# ==============================================================================
# IMPLEMENTATIONS
# ==============================================================================
class FAISSVectorStore(BaseStorage):
    """FAISS-based vector store implementation."""

    async def store_vectors(self, vectors: Sequence[Vector], metadata: Sequence[VectorMetadata]) -> None:
        """Store vectors with metadata."""
        pass

    async def search_vectors(self, query_vector: Vector, top_k: int) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        pass

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_vector_store(config: VectorStoreConfig) -> FAISSVectorStore:
    """Create and configure a vector store."""
    pass

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _validate_vector_dimension(vector: Vector, expected_dim: int) -> bool:
    """Validate vector dimension."""
    pass

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "FAISSVectorStore",
    "VectorStoreConfig",
    "VectorSearchResult",
    "create_vector_store",
]
```

## Validation Modules

**Required Sections (in order):**

1. `IMPORTS` - All import statements
2. `TYPE DEFINITIONS` - Type aliases and type definitions
3. `DATA MODELS` - Validation-specific data models and structures
4. `VALIDATOR IMPLEMENTATION` - Main validator classes and functions
5. `ERROR HANDLING` - Validation error handling and translation
6. `FACTORY FUNCTIONS` - Factory functions for creating validator instances
7. `HELPER FUNCTIONS` - Private helper functions
8. `EXPORTS` - `__all__` list

**Example:**

```python
# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
import re
from collections.abc import Mapping
from typing import Any

# Third-party imports
import jsonschema
from pydantic import BaseModel

# First-party imports
from Medical_KG_rev.models import Document
from Medical_KG_rev.utils import logger

# Relative imports
from .base import BaseValidator

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type ValidationResult = dict[str, Any]
type ValidationError = dict[str, Any]

# ==============================================================================
# DATA MODELS
# ==============================================================================
class ValidationConfig(BaseModel):
    """Configuration for validation."""
    strict_mode: bool = False
    schema_version: str = "r5"

class ValidationReport(BaseModel):
    """Report of validation results."""
    is_valid: bool
    errors: list[ValidationError]
    warnings: list[str]

# ==============================================================================
# VALIDATOR IMPLEMENTATION
# ==============================================================================
class FHIRValidator(BaseValidator):
    """Validator for FHIR resources."""

    def validate_resource(self, resource_type: str, resource_data: dict[str, Any]) -> ValidationReport:
        """Validate a FHIR resource."""
        pass

# ==============================================================================
# ERROR HANDLING
# ==============================================================================
def translate_validation_error(error: jsonschema.ValidationError) -> ValidationError:
    """Translate JSON schema error to validation error."""
    pass

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_fhir_validator(config: ValidationConfig) -> FHIRValidator:
    """Create and configure a FHIR validator."""
    pass

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _load_fhir_schema(resource_type: str) -> dict[str, Any]:
    """Load FHIR schema for resource type."""
    pass

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "FHIRValidator",
    "ValidationConfig",
    "ValidationReport",
    "create_fhir_validator",
]
```

## Utility Modules

**Required Sections (in order):**

1. `IMPORTS` - All import statements
2. `TYPE DEFINITIONS` - Type aliases and type definitions
3. `UTILITY FUNCTIONS` - Main utility functions
4. `HELPER CLASSES` - Utility classes and helpers
5. `FACTORY FUNCTIONS` - Factory functions for creating utility instances
6. `HELPER FUNCTIONS` - Private helper functions
7. `EXPORTS` - `__all__` list

**Example:**

```python
# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
import logging
import traceback
from collections.abc import Mapping
from typing import Any

# Third-party imports
from pydantic import BaseModel

# First-party imports
from Medical_KG_rev.models.base import BaseModel

# Relative imports
from .base import BaseUtility

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type ErrorContext = dict[str, Any]
type LogLevel = int

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def log_error(error: Exception, context: ErrorContext | None = None) -> None:
    """Log an error with optional context."""
    pass

def translate_error(error: Exception) -> str:
    """Translate an exception to a user-friendly message."""
    pass

# ==============================================================================
# HELPER CLASSES
# ==============================================================================
class ErrorHandler:
    """Utility class for error handling."""

    def handle_error(self, error: Exception) -> None:
        """Handle an error."""
        pass

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_error_handler(config: dict[str, Any]) -> ErrorHandler:
    """Create and configure an error handler."""
    pass

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _format_error_message(error: Exception) -> str:
    """Format error message for logging."""
    pass

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "log_error",
    "translate_error",
    "ErrorHandler",
    "create_error_handler",
]
```

## Test Modules

**Required Sections (in order):**

1. `IMPORTS` - All import statements
2. `TYPE DEFINITIONS` - Type aliases and type definitions
3. `FIXTURES` - Pytest fixtures
4. `UNIT TESTS - [ComponentName]` - Unit tests for specific components
5. `INTEGRATION TESTS` - Integration tests
6. `HELPER FUNCTIONS` - Test helper functions
7. `EXPORTS` - `__all__` list

**Example:**

```python
# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
import asyncio
from collections.abc import Sequence
from typing import Any

# Third-party imports
import pytest
import httpx
from httpx_mock import HTTPXMock

# First-party imports
from Medical_KG_rev.services.embedding import EmbeddingService
from Medical_KG_rev.models import Document

# Relative imports
from .conftest import TestConfig

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type TestData = dict[str, Any]
type MockResponse = dict[str, Any]

# ==============================================================================
# FIXTURES
# ==============================================================================
@pytest.fixture
def embedding_service() -> EmbeddingService:
    """Fixture for embedding service."""
    pass

@pytest.fixture
def test_document() -> Document:
    """Fixture for test document."""
    pass

# ==============================================================================
# UNIT TESTS - EmbeddingService
# ==============================================================================
class TestEmbeddingService:
    """Tests for EmbeddingService."""

    def test_embed_text_success(self, embedding_service: EmbeddingService) -> None:
        """Test successful text embedding."""
        pass

    def test_embed_text_invalid_input(self, embedding_service: EmbeddingService) -> None:
        """Test embedding with invalid input."""
        pass

# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================
class TestEmbeddingIntegration:
    """Integration tests for embedding service."""

    def test_embedding_with_external_api(self, embedding_service: EmbeddingService) -> None:
        """Test embedding with external API."""
        pass

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def create_test_embedding() -> list[float]:
    """Create test embedding vector."""
    pass

def assert_embedding_valid(embedding: list[float]) -> None:
    """Assert that embedding is valid."""
    pass

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "TestEmbeddingService",
    "TestEmbeddingIntegration",
    "create_test_embedding",
]
```

## Validation Rules

### Section Presence

- All required sections must be present for each module type
- Sections must appear in the specified order
- Empty sections are allowed but should be documented

### Section Content

- `IMPORTS`: Only import statements
- `TYPE DEFINITIONS`: Only type aliases and type definitions
- `DATA MODELS`: Only Pydantic models, dataclasses, and data structures
- `IMPLEMENTATIONS`: Only concrete implementations
- `FACTORY FUNCTIONS`: Only factory functions
- `HELPER FUNCTIONS`: Only private helper functions
- `EXPORTS`: Only `__all__` list

### Import Organization

- Imports must be grouped by category (stdlib, third-party, first-party, relative)
- Imports must be sorted alphabetically within each group
- Single blank line between import groups
- No blank lines within import groups

### Method Ordering

- Public methods before private methods
- Methods sorted alphabetically within visibility groups
- Special methods (**init**, **repr**, etc.) first
- Properties after methods
- Class methods and static methods after instance methods

## Enforcement

These standards are enforced through:

1. **Pre-commit hooks**: Validate section headers on every commit
2. **CI workflows**: Validate all files on pull requests
3. **Automated tools**: Section header checker with auto-fix capabilities
4. **Code review**: Manual review of section compliance

## Migration

When applying these standards to existing modules:

1. Identify module type
2. Add missing sections in correct order
3. Move existing code into appropriate sections
4. Add section headers with correct formatting
5. Validate compliance with automated tools
