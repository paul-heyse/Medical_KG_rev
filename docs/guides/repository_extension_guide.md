# Repository Extension Guide

This guide provides comprehensive instructions for extending the Medical_KG_rev repository with new components, following the established patterns and standards.

## Table of Contents

- [Overview](#overview)
- [Adding New Adapters](#adding-new-adapters)
- [Adding New Services](#adding-new-services)
- [Adding New Orchestration Stages](#adding-new-orchestration-stages)
- [Adding New Validation Rules](#adding-new-validation-rules)
- [Adding New Storage Backends](#adding-new-storage-backends)
- [Adding New Utility Functions](#adding-new-utility-functions)
- [Testing Patterns](#testing-patterns)
- [Documentation Standards](#documentation-standards)
- [Code Review Checklist](#code-review-checklist)

## Overview

The Medical_KG_rev repository follows a layered architecture with clear separation of concerns:

- **Gateway Layer**: Multi-protocol API façade (REST, GraphQL, gRPC, SOAP, AsyncAPI)
- **Service Layer**: Core business logic (embedding, chunking, retrieval, evaluation)
- **Adapter Layer**: External data source integrations
- **Orchestration Layer**: Pipeline management and job lifecycle
- **Knowledge Graph Layer**: Graph database operations and schema management
- **Storage Layer**: Object storage, caching, and vector storage backends
- **Validation Layer**: Data validation and compliance checking
- **Utility Layer**: Common functionality and helper functions

Each layer has specific patterns, interfaces, and standards that must be followed when adding new components.

## Adding New Adapters

### Overview

Adapters provide integration with external biomedical data sources through a standardized interface. All adapters must implement the `BaseAdapter` interface and follow the established patterns.

### Implementation Steps

#### 1. Create Adapter Module Structure

```python
# src/Medical_KG_rev/adapters/new_source/adapter.py
"""New data source adapter implementation.

This adapter provides integration with the New Data Source API for
retrieving biomedical research data.

Key Responsibilities:
    - Data fetching from New Data Source API
    - Data parsing and transformation to internal format
    - Error handling and retry logic
    - Rate limiting and backoff

Collaborators:
    - Upstream: Gateway coordinators, orchestration stages
    - Downstream: New Data Source API

Side Effects:
    - External API calls to New Data Source
    - Rate limiting and caching
    - Metric emission and logging

Thread Safety:
    - Thread-safe: All public methods can be called concurrently

Performance Characteristics:
    - Rate limits: 100 requests per minute
    - Timeouts: 30 seconds per request
    - Retry behavior: Exponential backoff with jitter

Example:
    >>> from Medical_KG_rev.adapters.new_source import NewSourceAdapter
    >>> adapter = NewSourceAdapter(api_key="...")
    >>> result = await adapter.fetch("document-id")
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

import asyncio
import logging
from typing import Any
from collections.abc import Mapping, Sequence

import httpx
from pydantic import BaseModel, Field

from Medical_KG_rev.adapters.base import BaseAdapter, AdapterContext, AdapterResult
from Medical_KG_rev.utils.http_client import AsyncHttpClient, RetryConfig
from Medical_KG_rev.utils.errors import AdapterError, NetworkError, RateLimitError

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type ApiResponse = dict[str, Any]
type DocumentData = dict[str, Any]

# ==============================================================================
# DATA MODELS
# ==============================================================================
class NewSourceConfig(BaseModel):
    """Configuration for New Source adapter."""

    api_key: str = Field(..., description="API key for authentication")
    base_url: str = Field(default="https://api.newsource.com/v1", description="Base API URL")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    rate_limit: int = Field(default=100, description="Requests per minute")

class NewSourceRequest(BaseModel):
    """Request model for New Source API."""

    document_id: str = Field(..., description="Document identifier")
    format: str = Field(default="json", description="Response format")
    include_metadata: bool = Field(default=True, description="Include metadata")

class NewSourceResponse(BaseModel):
    """Response model for New Source API."""

    document_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="Document metadata")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

# ==============================================================================
# ADAPTER IMPLEMENTATION
# ==============================================================================
class NewSourceAdapter(BaseAdapter):
    """Adapter for New Data Source API."""

    def __init__(self, config: NewSourceConfig) -> None:
        """Initialize adapter with configuration.

        Args:
            config: Adapter configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize HTTP client with retry configuration
        retry_config = RetryConfig(
            max_retries=config.max_retries,
            timeout=config.timeout,
            retry_on_status=[500, 502, 503, 504, 429]
        )

        self.client = AsyncHttpClient(
            retry_config=retry_config,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "User-Agent": "Medical-KG-Adapter/1.0",
                "Accept": "application/json"
            }
        )

    async def fetch(self, identifier: str, context: AdapterContext) -> AdapterResult[NewSourceResponse]:
        """Fetch document from New Data Source.

        Args:
            identifier: Document identifier.
            context: Adapter context with tenant and correlation information.

        Returns:
            Adapter result containing document data.

        Raises:
            AdapterError: If fetching fails.
        """
        try:
            self.logger.info(
                "Fetching document from New Source",
                extra={
                    "document_id": identifier,
                    "tenant_id": context.tenant_id,
                    "correlation_id": context.correlation_id
                }
            )

            # Make API request
            url = f"{self.config.base_url}/documents/{identifier}"
            response = await self.client.get(url)
            response.raise_for_status()

            # Parse response
            data = response.json()
            document = NewSourceResponse(**data)

            self.logger.info(
                "Successfully fetched document from New Source",
                extra={
                    "document_id": identifier,
                    "tenant_id": context.tenant_id,
                    "correlation_id": context.correlation_id
                }
            )

            return AdapterResult(
                success=True,
                data=document,
                metadata={
                    "source": "new_source",
                    "fetched_at": data.get("fetched_at"),
                    "api_version": data.get("api_version")
                }
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded for New Source API: {e}")
            elif e.response.status_code >= 500:
                raise NetworkError(f"Server error from New Source API: {e}")
            else:
                raise AdapterError(f"HTTP error from New Source API: {e}")

        except httpx.RequestError as e:
            raise NetworkError(f"Network error accessing New Source API: {e}")

        except Exception as e:
            raise AdapterError(f"Unexpected error fetching from New Source: {e}")

    async def search(self, query: str, context: AdapterContext,
                    limit: int = 10) -> AdapterResult[list[NewSourceResponse]]:
        """Search documents in New Data Source.

        Args:
            query: Search query.
            context: Adapter context.
            limit: Maximum number of results.

        Returns:
            Adapter result containing list of documents.
        """
        try:
            self.logger.info(
                "Searching documents in New Source",
                extra={
                    "query": query,
                    "limit": limit,
                    "tenant_id": context.tenant_id,
                    "correlation_id": context.correlation_id
                }
            )

            # Make search request
            url = f"{self.config.base_url}/search"
            params = {"q": query, "limit": limit}
            response = await self.client.get(url, params=params)
            response.raise_for_status()

            # Parse response
            data = response.json()
            documents = [NewSourceResponse(**doc) for doc in data.get("results", [])]

            return AdapterResult(
                success=True,
                data=documents,
                metadata={
                    "source": "new_source",
                    "query": query,
                    "total_results": data.get("total", 0),
                    "limit": limit
                }
            )

        except Exception as e:
            raise AdapterError(f"Error searching New Source: {e}")

    async def health_check(self) -> bool:
        """Check adapter health.

        Returns:
            True if adapter is healthy, False otherwise.
        """
        try:
            response = await self.client.get(f"{self.config.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

# ==============================================================================
# ERROR HANDLING
# ==============================================================================
class NewSourceError(AdapterError):
    """New Source adapter specific error."""
    pass

class NewSourceRateLimitError(NewSourceError, RateLimitError):
    """Rate limit error for New Source API."""
    pass

class NewSourceNetworkError(NewSourceError, NetworkError):
    """Network error for New Source API."""
    pass

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_new_source_adapter(config: NewSourceConfig) -> NewSourceAdapter:
    """Create and configure New Source adapter.

    Args:
        config: Adapter configuration.

    Returns:
        Configured adapter instance.
    """
    return NewSourceAdapter(config)

def load_new_source_config(config_data: Mapping[str, Any]) -> NewSourceConfig:
    """Load adapter configuration from data.

    Args:
        config_data: Configuration data.

    Returns:
        Validated configuration.
    """
    return NewSourceConfig(**config_data)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def validate_document_id(document_id: str) -> bool:
    """Validate document ID format.

    Args:
        document_id: Document identifier to validate.

    Returns:
        True if valid, False otherwise.
    """
    # Implement validation logic
    return len(document_id) > 0 and document_id.isalnum()

def format_api_error(error: httpx.HTTPError) -> str:
    """Format API error for logging.

    Args:
        error: HTTP error to format.

    Returns:
        Formatted error message.
    """
    if hasattr(error, 'response') and error.response is not None:
        return f"HTTP {error.response.status_code}: {error.response.text}"
    return str(error)

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "NewSourceAdapter",
    "NewSourceConfig",
    "NewSourceRequest",
    "NewSourceResponse",
    "NewSourceError",
    "NewSourceRateLimitError",
    "NewSourceNetworkError",
    "create_new_source_adapter",
    "load_new_source_config",
    "validate_document_id",
    "format_api_error",
]
```

#### 2. Create Package Initialization

```python
# src/Medical_KG_rev/adapters/new_source/__init__.py
"""New data source adapter package.

This package provides integration with the New Data Source API for
retrieving biomedical research data.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
from .adapter import (
    NewSourceAdapter,
    NewSourceConfig,
    NewSourceRequest,
    NewSourceResponse,
    NewSourceError,
    NewSourceRateLimitError,
    NewSourceNetworkError,
    create_new_source_adapter,
    load_new_source_config,
)

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "NewSourceAdapter",
    "NewSourceConfig",
    "NewSourceRequest",
    "NewSourceResponse",
    "NewSourceError",
    "NewSourceRateLimitError",
    "NewSourceNetworkError",
    "create_new_source_adapter",
    "load_new_source_config",
]
```

#### 3. Register Adapter

```python
# src/Medical_KG_rev/adapters/registry.py
"""Adapter registry for dynamic adapter loading."""

# Add to existing registry
ADAPTER_REGISTRY = {
    # ... existing adapters ...
    "new_source": {
        "module": "Medical_KG_rev.adapters.new_source",
        "adapter_class": "NewSourceAdapter",
        "config_class": "NewSourceConfig",
        "factory_function": "create_new_source_adapter",
    },
}
```

#### 4. Create Configuration

```yaml
# config/adapters/new_source.yaml
adapter:
  type: "new_source"
  config:
    api_key: "${NEW_SOURCE_API_KEY}"
    base_url: "https://api.newsource.com/v1"
    timeout: 30
    max_retries: 3
    rate_limit: 100
```

#### 5. Add Tests

```python
# tests/adapters/test_new_source.py
"""Tests for New Source adapter."""

import pytest
from unittest.mock import AsyncMock, patch
import httpx

from Medical_KG_rev.adapters.new_source import (
    NewSourceAdapter,
    NewSourceConfig,
    NewSourceError,
    NewSourceRateLimitError,
)
from Medical_KG_rev.adapters.base import AdapterContext

# ==============================================================================
# FIXTURES
# ==============================================================================
@pytest.fixture
def adapter_config():
    """Fixture for adapter configuration."""
    return NewSourceConfig(
        api_key="test-key",
        base_url="https://api.newsource.com/v1",
        timeout=30,
        max_retries=3,
        rate_limit=100
    )

@pytest.fixture
def adapter(adapter_config):
    """Fixture for adapter instance."""
    return NewSourceAdapter(adapter_config)

@pytest.fixture
def context():
    """Fixture for adapter context."""
    return AdapterContext(
        tenant_id="test-tenant",
        correlation_id="test-correlation"
    )

# ==============================================================================
# UNIT TESTS - NewSourceAdapter
# ==============================================================================
class TestNewSourceAdapter:
    """Tests for NewSourceAdapter."""

    @pytest.mark.asyncio
    async def test_fetch_success(self, adapter, context):
        """Test successful document fetch."""
        # Mock API response
        mock_response = {
            "document_id": "doc-123",
            "title": "Test Document",
            "content": "Test content",
            "metadata": {"author": "Test Author"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }

        with patch.object(adapter.client, 'get') as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status.return_value = None

            result = await adapter.fetch("doc-123", context)

            assert result.success
            assert result.data.document_id == "doc-123"
            assert result.data.title == "Test Document"
            assert result.metadata["source"] == "new_source"

    @pytest.mark.asyncio
    async def test_fetch_rate_limit_error(self, adapter, context):
        """Test rate limit error handling."""
        with patch.object(adapter.client, 'get') as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "Rate limit exceeded",
                request=httpx.Request("GET", "https://api.newsource.com/v1/documents/doc-123"),
                response=httpx.Response(429, text="Rate limit exceeded")
            )

            with pytest.raises(NewSourceRateLimitError):
                await adapter.fetch("doc-123", context)

    @pytest.mark.asyncio
    async def test_search_success(self, adapter, context):
        """Test successful document search."""
        # Mock search response
        mock_response = {
            "results": [
                {
                    "document_id": "doc-1",
                    "title": "Document 1",
                    "content": "Content 1",
                    "metadata": {},
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                },
                {
                    "document_id": "doc-2",
                    "title": "Document 2",
                    "content": "Content 2",
                    "metadata": {},
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            ],
            "total": 2
        }

        with patch.object(adapter.client, 'get') as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status.return_value = None

            result = await adapter.search("test query", context, limit=10)

            assert result.success
            assert len(result.data) == 2
            assert result.metadata["total_results"] == 2

    @pytest.mark.asyncio
    async def test_health_check_success(self, adapter):
        """Test successful health check."""
        with patch.object(adapter.client, 'get') as mock_get:
            mock_get.return_value.status_code = 200

            result = await adapter.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, adapter):
        """Test failed health check."""
        with patch.object(adapter.client, 'get') as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection failed")

            result = await adapter.health_check()

            assert result is False

# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================
class TestNewSourceAdapterIntegration:
    """Integration tests for New Source adapter."""

    @pytest.mark.asyncio
    async def test_end_to_end_fetch(self, adapter_config, context):
        """Test end-to-end document fetch."""
        # This test would require actual API access or comprehensive mocking
        # For now, we'll skip it in CI but include it for local testing
        pytest.skip("Requires actual API access")

    @pytest.mark.asyncio
    async def test_end_to_end_search(self, adapter_config, context):
        """Test end-to-end document search."""
        # This test would require actual API access or comprehensive mocking
        # For now, we'll skip it in CI but include it for local testing
        pytest.skip("Requires actual API access")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def create_mock_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Create mock HTTP response for testing."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("GET", "https://api.newsource.com/v1/test")
    )

def create_mock_error(status_code: int, message: str) -> httpx.HTTPStatusError:
    """Create mock HTTP error for testing."""
    return httpx.HTTPStatusError(
        message,
        request=httpx.Request("GET", "https://api.newsource.com/v1/test"),
        response=httpx.Response(status_code, text=message)
    )
```

### Testing Patterns

#### Unit Tests

- Test individual methods with mocked dependencies
- Use fixtures for common test data
- Test both success and error scenarios
- Verify logging and metrics emission

#### Integration Tests

- Test with real API endpoints (when available)
- Test error handling with actual network failures
- Test rate limiting and retry logic
- Test configuration loading and validation

#### Contract Tests

- Test adapter interface compliance
- Test error type inheritance
- Test response format compliance
- Test configuration schema validation

## Adding New Services

### Overview

Services provide core business logic for the Medical KG system. All services must follow the established patterns and implement the appropriate interfaces.

### Implementation Steps

#### 1. Create Service Module Structure

```python
# src/Medical_KG_rev/services/new_service/service.py
"""New service implementation.

This service provides [description of service functionality] for the
Medical KG system.

Key Responsibilities:
    - [Responsibility 1]
    - [Responsibility 2]
    - [Responsibility 3]

Collaborators:
    - Upstream: [List modules that call this service]
    - Downstream: [List modules this service depends on]

Side Effects:
    - [List any side effects]

Thread Safety:
    - [Thread safety description]

Performance Characteristics:
    - [Performance characteristics]

Example:
    >>> from Medical_KG_rev.services.new_service import NewService
    >>> service = NewService(config)
    >>> result = await service.process(data)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

import asyncio
import logging
from typing import Any
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from Medical_KG_rev.utils.errors import ServiceError
from Medical_KG_rev.utils.http_client import AsyncHttpClient

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type ServiceConfig = dict[str, Any]
type ProcessingResult = dict[str, Any]

# ==============================================================================
# DATA MODELS
# ==============================================================================
class NewServiceConfig(BaseModel):
    """Configuration for New Service."""

    enabled: bool = Field(default=True, description="Whether service is enabled")
    timeout: int = Field(default=30, description="Processing timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    batch_size: int = Field(default=100, description="Batch processing size")

class NewServiceRequest(BaseModel):
    """Request model for New Service."""

    data: Sequence[Any] = Field(..., description="Data to process")
    options: Mapping[str, Any] = Field(default_factory=dict, description="Processing options")

class NewServiceResponse(BaseModel):
    """Response model for New Service."""

    results: Sequence[Any] = Field(..., description="Processing results")
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="Response metadata")
    processing_time: float = Field(..., description="Processing time in seconds")

# ==============================================================================
# INTERFACES
# ==============================================================================
class NewServiceInterface:
    """Interface for New Service."""

    async def process(self, request: NewServiceRequest) -> NewServiceResponse:
        """Process data using the service."""
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check service health."""
        raise NotImplementedError

# ==============================================================================
# IMPLEMENTATIONS
# ==============================================================================
class NewService(NewServiceInterface):
    """Implementation of New Service."""

    def __init__(self, config: NewServiceConfig) -> None:
        """Initialize service with configuration.

        Args:
            config: Service configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize service resources."""
        if self._initialized:
            return

        self.logger.info("Initializing New Service")

        # Initialize resources
        # ...

        self._initialized = True
        self.logger.info("New Service initialized successfully")

    async def process(self, request: NewServiceRequest) -> NewServiceResponse:
        """Process data using the service.

        Args:
            request: Service request.

        Returns:
            Service response with results.

        Raises:
            ServiceError: If processing fails.
        """
        if not self._initialized:
            await self.initialize()

        start_time = asyncio.get_event_loop().time()

        try:
            self.logger.info(
                "Processing data with New Service",
                extra={
                    "data_count": len(request.data),
                    "options": request.options
                }
            )

            # Process data
            results = await self._process_batch(request.data, request.options)

            processing_time = asyncio.get_event_loop().time() - start_time

            self.logger.info(
                "Successfully processed data with New Service",
                extra={
                    "result_count": len(results),
                    "processing_time": processing_time
                }
            )

            return NewServiceResponse(
                results=results,
                metadata={
                    "service": "new_service",
                    "version": "1.0.0",
                    "processed_at": asyncio.get_event_loop().time()
                },
                processing_time=processing_time
            )

        except Exception as e:
            self.logger.error(f"Error processing data with New Service: {e}")
            raise ServiceError(f"Processing failed: {e}")

    async def _process_batch(self, data: Sequence[Any],
                           options: Mapping[str, Any]) -> list[Any]:
        """Process a batch of data.

        Args:
            data: Data to process.
            options: Processing options.

        Returns:
            List of processed results.
        """
        results = []

        for item in data:
            try:
                result = await self._process_item(item, options)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to process item: {e}")
                # Continue processing other items
                continue

        return results

    async def _process_item(self, item: Any, options: Mapping[str, Any]) -> Any:
        """Process a single item.

        Args:
            item: Item to process.
            options: Processing options.

        Returns:
            Processed item.
        """
        # Implement item processing logic
        # ...
        return item

    async def health_check(self) -> bool:
        """Check service health.

        Returns:
            True if service is healthy, False otherwise.
        """
        try:
            # Implement health check logic
            # ...
            return True
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup service resources."""
        if not self._initialized:
            return

        self.logger.info("Cleaning up New Service")

        # Cleanup resources
        # ...

        self._initialized = False
        self.logger.info("New Service cleaned up successfully")

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_new_service(config: NewServiceConfig) -> NewService:
    """Create and configure New Service.

    Args:
        config: Service configuration.

    Returns:
        Configured service instance.
    """
    return NewService(config)

def load_new_service_config(config_data: Mapping[str, Any]) -> NewServiceConfig:
    """Load service configuration from data.

    Args:
        config_data: Configuration data.

    Returns:
        Validated configuration.
    """
    return NewServiceConfig(**config_data)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def validate_service_config(config: NewServiceConfig) -> bool:
    """Validate service configuration.

    Args:
        config: Configuration to validate.

    Returns:
        True if valid, False otherwise.
    """
    # Implement validation logic
    return config.enabled and config.timeout > 0

def format_processing_error(error: Exception) -> str:
    """Format processing error for logging.

    Args:
        error: Error to format.

    Returns:
        Formatted error message.
    """
    return f"Processing error: {type(error).__name__}: {error}"

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "NewService",
    "NewServiceConfig",
    "NewServiceRequest",
    "NewServiceResponse",
    "NewServiceInterface",
    "create_new_service",
    "load_new_service_config",
    "validate_service_config",
    "format_processing_error",
]
```

### Testing Patterns

#### Unit Tests

- Test individual methods with mocked dependencies
- Test configuration validation
- Test error handling and recovery
- Test health check functionality

#### Integration Tests

- Test service initialization and cleanup
- Test batch processing with real data
- Test error propagation and handling
- Test performance characteristics

#### Performance Tests

- Test processing throughput
- Test memory usage patterns
- Test scalability with large datasets
- Test timeout and retry behavior

## Adding New Orchestration Stages

### Overview

Orchestration stages provide pipeline processing capabilities. All stages must implement the `StageContract` interface and follow the established patterns.

### Implementation Steps

#### 1. Create Stage Module Structure

```python
# src/Medical_KG_rev/orchestration/stages/new_stage.py
"""New orchestration stage implementation.

This stage provides [description of stage functionality] for the
Medical KG orchestration pipeline.

Key Responsibilities:
    - [Responsibility 1]
    - [Responsibility 2]
    - [Responsibility 3]

Collaborators:
    - Upstream: [Previous stages]
    - Downstream: [Subsequent stages]

Side Effects:
    - [List any side effects]

Thread Safety:
    - [Thread safety description]

Performance Characteristics:
    - [Performance characteristics]

Example:
    >>> from Medical_KG_rev.orchestration.stages.new_stage import NewStage
    >>> stage = NewStage(config)
    >>> result = await stage.execute(input_data)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

import asyncio
import logging
from typing import Any
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from Medical_KG_rev.orchestration.stages.contracts import StageContract, StageInput, StageOutput
from Medical_KG_rev.utils.errors import OrchestrationError

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type StageConfig = dict[str, Any]
type ProcessingResult = dict[str, Any]

# ==============================================================================
# STAGE CONTEXT DATA MODELS
# ==============================================================================
class NewStageConfig(BaseModel):
    """Configuration for New Stage."""

    enabled: bool = Field(default=True, description="Whether stage is enabled")
    timeout: int = Field(default=300, description="Stage timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    batch_size: int = Field(default=100, description="Batch processing size")
    parallel_workers: int = Field(default=4, description="Number of parallel workers")

class NewStageContext(BaseModel):
    """Context for New Stage execution."""

    job_id: str = Field(..., description="Job identifier")
    stage_name: str = Field(..., description="Stage name")
    tenant_id: str = Field(..., description="Tenant identifier")
    correlation_id: str = Field(..., description="Correlation identifier")
    config: NewStageConfig = Field(..., description="Stage configuration")
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="Additional metadata")

# ==============================================================================
# STAGE IMPLEMENTATIONS
# ==============================================================================
class NewStage(StageContract):
    """Implementation of New Stage."""

    def __init__(self, config: NewStageConfig) -> None:
        """Initialize stage with configuration.

        Args:
            config: Stage configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize stage resources."""
        if self._initialized:
            return

        self.logger.info("Initializing New Stage")

        # Initialize resources
        # ...

        self._initialized = True
        self.logger.info("New Stage initialized successfully")

    async def execute(self, input_data: StageInput) -> StageOutput:
        """Execute stage processing.

        Args:
            input_data: Stage input data.

        Returns:
            Stage output data.

        Raises:
            OrchestrationError: If stage execution fails.
        """
        if not self._initialized:
            await self.initialize()

        start_time = asyncio.get_event_loop().time()

        try:
            self.logger.info(
                "Executing New Stage",
                extra={
                    "job_id": input_data.job_id,
                    "stage_name": input_data.stage_name,
                    "tenant_id": input_data.tenant_id,
                    "correlation_id": input_data.correlation_id
                }
            )

            # Create stage context
            context = NewStageContext(
                job_id=input_data.job_id,
                stage_name=input_data.stage_name,
                tenant_id=input_data.tenant_id,
                correlation_id=input_data.correlation_id,
                config=self.config,
                metadata=input_data.metadata
            )

            # Process data
            result = await self._process_data(input_data.data, context)

            processing_time = asyncio.get_event_loop().time() - start_time

            self.logger.info(
                "Successfully executed New Stage",
                extra={
                    "job_id": input_data.job_id,
                    "processing_time": processing_time
                }
            )

            return StageOutput(
                success=True,
                data=result,
                metadata={
                    "stage": "new_stage",
                    "version": "1.0.0",
                    "processing_time": processing_time,
                    "processed_at": asyncio.get_event_loop().time()
                }
            )

        except Exception as e:
            self.logger.error(f"Error executing New Stage: {e}")
            raise OrchestrationError(f"Stage execution failed: {e}")

    async def _process_data(self, data: Any, context: NewStageContext) -> Any:
        """Process input data.

        Args:
            data: Input data to process.
            context: Stage context.

        Returns:
            Processed data.
        """
        # Implement data processing logic
        # ...
        return data

    async def health_check(self) -> bool:
        """Check stage health.

        Returns:
            True if stage is healthy, False otherwise.
        """
        try:
            # Implement health check logic
            # ...
            return True
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup stage resources."""
        if not self._initialized:
            return

        self.logger.info("Cleaning up New Stage")

        # Cleanup resources
        # ...

        self._initialized = False
        self.logger.info("New Stage cleaned up successfully")

# ==============================================================================
# PLUGIN REGISTRATION
# ==============================================================================
def register_new_stage(registry: dict[str, Any]) -> None:
    """Register New Stage in the plugin registry.

    Args:
        registry: Plugin registry to update.
    """
    registry["new_stage"] = {
        "stage_class": NewStage,
        "config_class": NewStageConfig,
        "description": "New orchestration stage",
        "version": "1.0.0"
    }

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_new_stage(config: NewStageConfig) -> NewStage:
    """Create and configure New Stage.

    Args:
        config: Stage configuration.

    Returns:
        Configured stage instance.
    """
    return NewStage(config)

def load_new_stage_config(config_data: Mapping[str, Any]) -> NewStageConfig:
    """Load stage configuration from data.

    Args:
        config_data: Configuration data.

    Returns:
        Validated configuration.
    """
    return NewStageConfig(**config_data)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def validate_stage_config(config: NewStageConfig) -> bool:
    """Validate stage configuration.

    Args:
        config: Configuration to validate.

    Returns:
        True if valid, False otherwise.
    """
    # Implement validation logic
    return config.enabled and config.timeout > 0

def format_stage_error(error: Exception) -> str:
    """Format stage error for logging.

    Args:
        error: Error to format.

    Returns:
        Formatted error message.
    """
    return f"Stage error: {type(error).__name__}: {error}"

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "NewStage",
    "NewStageConfig",
    "NewStageContext",
    "register_new_stage",
    "create_new_stage",
    "load_new_stage_config",
    "validate_stage_config",
    "format_stage_error",
]
```

### Testing Patterns

#### Unit Tests

- Test stage initialization and cleanup
- Test stage execution with various inputs
- Test error handling and recovery
- Test health check functionality

#### Integration Tests

- Test stage within orchestration pipeline
- Test stage dependencies and data flow
- Test error propagation between stages
- Test performance characteristics

#### Contract Tests

- Test stage contract compliance
- Test input/output format validation
- Test error type inheritance
- Test configuration schema validation

## Adding New Validation Rules

### Overview

Validation rules provide data validation and compliance checking. All validation rules must follow the established patterns and implement the appropriate interfaces.

### Implementation Steps

#### 1. Create Validation Module Structure

```python
# src/Medical_KG_rev/validation/new_validation.py
"""New validation rule implementation.

This validation module provides [description of validation functionality]
for the Medical KG system.

Key Responsibilities:
    - [Responsibility 1]
    - [Responsibility 2]
    - [Responsibility 3]

Collaborators:
    - Upstream: [Modules that use this validation]
    - Downstream: [Modules this validation depends on]

Side Effects:
    - [List any side effects]

Thread Safety:
    - [Thread safety description]

Performance Characteristics:
    - [Performance characteristics]

Example:
    >>> from Medical_KG_rev.validation.new_validation import NewValidator
    >>> validator = NewValidator(config)
    >>> result = await validator.validate(data)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

import asyncio
import logging
from typing import Any
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from Medical_KG_rev.validation.base import ValidationResult, ValidationError
from Medical_KG_rev.utils.errors import ValidationError as BaseValidationError

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type ValidationConfig = dict[str, Any]
type ValidationData = dict[str, Any]

# ==============================================================================
# DATA MODELS
# ==============================================================================
class NewValidationConfig(BaseModel):
    """Configuration for New Validation."""

    enabled: bool = Field(default=True, description="Whether validation is enabled")
    strict_mode: bool = Field(default=True, description="Whether to use strict validation")
    timeout: int = Field(default=30, description="Validation timeout in seconds")
    max_errors: int = Field(default=100, description="Maximum number of errors to report")

class NewValidationRequest(BaseModel):
    """Request model for New Validation."""

    data: ValidationData = Field(..., description="Data to validate")
    schema: Mapping[str, Any] = Field(..., description="Validation schema")
    options: Mapping[str, Any] = Field(default_factory=dict, description="Validation options")

class NewValidationResponse(BaseModel):
    """Response model for New Validation."""

    valid: bool = Field(..., description="Whether data is valid")
    errors: Sequence[str] = Field(default_factory=list, description="Validation errors")
    warnings: Sequence[str] = Field(default_factory=list, description="Validation warnings")
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="Response metadata")

# ==============================================================================
# VALIDATOR IMPLEMENTATION
# ==============================================================================
class NewValidator:
    """Implementation of New Validation."""

    def __init__(self, config: NewValidationConfig) -> None:
        """Initialize validator with configuration.

        Args:
            config: Validation configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize validator resources."""
        if self._initialized:
            return

        self.logger.info("Initializing New Validator")

        # Initialize resources
        # ...

        self._initialized = True
        self.logger.info("New Validator initialized successfully")

    async def validate(self, request: NewValidationRequest) -> NewValidationResponse:
        """Validate data using the validator.

        Args:
            request: Validation request.

        Returns:
            Validation response with results.

        Raises:
            ValidationError: If validation fails.
        """
        if not self._initialized:
            await self.initialize()

        start_time = asyncio.get_event_loop().time()

        try:
            self.logger.info(
                "Validating data with New Validator",
                extra={
                    "data_keys": list(request.data.keys()),
                    "schema_keys": list(request.schema.keys()),
                    "options": request.options
                }
            )

            # Validate data
            errors, warnings = await self._validate_data(
                request.data,
                request.schema,
                request.options
            )

            processing_time = asyncio.get_event_loop().time() - start_time

            is_valid = len(errors) == 0

            self.logger.info(
                "Validation completed with New Validator",
                extra={
                    "valid": is_valid,
                    "error_count": len(errors),
                    "warning_count": len(warnings),
                    "processing_time": processing_time
                }
            )

            return NewValidationResponse(
                valid=is_valid,
                errors=errors,
                warnings=warnings,
                metadata={
                    "validator": "new_validator",
                    "version": "1.0.0",
                    "processing_time": processing_time,
                    "validated_at": asyncio.get_event_loop().time()
                }
            )

        except Exception as e:
            self.logger.error(f"Error validating data with New Validator: {e}")
            raise ValidationError(f"Validation failed: {e}")

    async def _validate_data(self, data: ValidationData,
                           schema: Mapping[str, Any],
                           options: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        """Validate data against schema.

        Args:
            data: Data to validate.
            schema: Validation schema.
            options: Validation options.

        Returns:
            Tuple of (errors, warnings).
        """
        errors = []
        warnings = []

        # Implement validation logic
        # ...

        return errors, warnings

    async def health_check(self) -> bool:
        """Check validator health.

        Returns:
            True if validator is healthy, False otherwise.
        """
        try:
            # Implement health check logic
            # ...
            return True
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup validator resources."""
        if not self._initialized:
            return

        self.logger.info("Cleaning up New Validator")

        # Cleanup resources
        # ...

        self._initialized = False
        self.logger.info("New Validator cleaned up successfully")

# ==============================================================================
# ERROR HANDLING
# ==============================================================================
class NewValidationError(BaseValidationError):
    """New validation specific error."""
    pass

class NewValidationSchemaError(NewValidationError):
    """Schema validation error."""
    pass

class NewValidationDataError(NewValidationError):
    """Data validation error."""
    pass

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_new_validator(config: NewValidationConfig) -> NewValidator:
    """Create and configure New Validator.

    Args:
        config: Validation configuration.

    Returns:
        Configured validator instance.
    """
    return NewValidator(config)

def load_new_validation_config(config_data: Mapping[str, Any]) -> NewValidationConfig:
    """Load validation configuration from data.

    Args:
        config_data: Configuration data.

    Returns:
        Validated configuration.
    """
    return NewValidationConfig(**config_data)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def validate_validation_config(config: NewValidationConfig) -> bool:
    """Validate validation configuration.

    Args:
        config: Configuration to validate.

    Returns:
        True if valid, False otherwise.
    """
    # Implement validation logic
    return config.enabled and config.timeout > 0

def format_validation_error(error: Exception) -> str:
    """Format validation error for logging.

    Args:
        error: Error to format.

    Returns:
        Formatted error message.
    """
    return f"Validation error: {type(error).__name__}: {error}"

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "NewValidator",
    "NewValidationConfig",
    "NewValidationRequest",
    "NewValidationResponse",
    "NewValidationError",
    "NewValidationSchemaError",
    "NewValidationDataError",
    "create_new_validator",
    "load_new_validation_config",
    "validate_validation_config",
    "format_validation_error",
]
```

### Testing Patterns

#### Unit Tests

- Test validation logic with various inputs
- Test error handling and recovery
- Test configuration validation
- Test health check functionality

#### Integration Tests

- Test validation within the validation pipeline
- Test validation with real data
- Test error propagation and handling
- Test performance characteristics

#### Contract Tests

- Test validation interface compliance
- Test input/output format validation
- Test error type inheritance
- Test configuration schema validation

## Adding New Storage Backends

### Overview

Storage backends provide data persistence and retrieval capabilities. All storage backends must implement the appropriate interfaces and follow the established patterns.

### Implementation Steps

#### 1. Create Storage Module Structure

```python
# src/Medical_KG_rev/storage/new_backend.py
"""New storage backend implementation.

This storage backend provides [description of storage functionality]
for the Medical KG system.

Key Responsibilities:
    - [Responsibility 1]
    - [Responsibility 2]
    - [Responsibility 3]

Collaborators:
    - Upstream: [Modules that use this storage]
    - Downstream: [Storage systems this backend uses]

Side Effects:
    - [List any side effects]

Thread Safety:
    - [Thread safety description]

Performance Characteristics:
    - [Performance characteristics]

Example:
    >>> from Medical_KG_rev.storage.new_backend import NewStorageBackend
    >>> backend = NewStorageBackend(config)
    >>> await backend.store(key, data)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

import asyncio
import logging
from typing import Any
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from Medical_KG_rev.storage.base import ObjectStore, ObjectMetadata, StorageError
from Medical_KG_rev.utils.errors import StorageError as BaseStorageError

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type StorageConfig = dict[str, Any]
type StorageData = dict[str, Any]

# ==============================================================================
# DATA MODELS
# ==============================================================================
class NewStorageConfig(BaseModel):
    """Configuration for New Storage Backend."""

    enabled: bool = Field(default=True, description="Whether storage is enabled")
    connection_string: str = Field(..., description="Storage connection string")
    timeout: int = Field(default=30, description="Storage timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    batch_size: int = Field(default=100, description="Batch operation size")

class NewStorageMetadata(BaseModel):
    """Metadata for New Storage operations."""

    key: str = Field(..., description="Storage key")
    size: int = Field(..., description="Data size in bytes")
    content_type: str = Field(default="application/json", description="Content type")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="Additional metadata")

# ==============================================================================
# INTERFACES
# ==============================================================================
class NewStorageInterface:
    """Interface for New Storage Backend."""

    async def store(self, key: str, data: Any, metadata: Mapping[str, Any] = None) -> ObjectMetadata:
        """Store data with the given key."""
        raise NotImplementedError

    async def retrieve(self, key: str) -> Any:
        """Retrieve data by key."""
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        """Delete data by key."""
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        raise NotImplementedError

    async def list_keys(self, prefix: str = "") -> Sequence[str]:
        """List keys with optional prefix."""
        raise NotImplementedError

# ==============================================================================
# IMPLEMENTATIONS
# ==============================================================================
class NewStorageBackend(ObjectStore, NewStorageInterface):
    """Implementation of New Storage Backend."""

    def __init__(self, config: NewStorageConfig) -> None:
        """Initialize storage backend with configuration.

        Args:
            config: Storage configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        self._connection = None

    async def initialize(self) -> None:
        """Initialize storage backend resources."""
        if self._initialized:
            return

        self.logger.info("Initializing New Storage Backend")

        # Initialize storage connection
        # ...

        self._initialized = True
        self.logger.info("New Storage Backend initialized successfully")

    async def store(self, key: str, data: Any,
                   metadata: Mapping[str, Any] = None) -> ObjectMetadata:
        """Store data with the given key.

        Args:
            key: Storage key.
            data: Data to store.
            metadata: Optional metadata.

        Returns:
            Object metadata.

        Raises:
            StorageError: If storage fails.
        """
        if not self._initialized:
            await self.initialize()

        try:
            self.logger.info(
                "Storing data with New Storage Backend",
                extra={"key": key, "metadata": metadata}
            )

            # Store data
            # ...

            metadata_obj = ObjectMetadata(
                key=key,
                size=len(str(data)),
                content_type="application/json",
                created_at=asyncio.get_event_loop().time(),
                updated_at=asyncio.get_event_loop().time(),
                metadata=metadata or {}
            )

            self.logger.info(
                "Successfully stored data with New Storage Backend",
                extra={"key": key}
            )

            return metadata_obj

        except Exception as e:
            self.logger.error(f"Error storing data with New Storage Backend: {e}")
            raise StorageError(f"Storage failed: {e}")

    async def retrieve(self, key: str) -> Any:
        """Retrieve data by key.

        Args:
            key: Storage key.

        Returns:
            Retrieved data.

        Raises:
            StorageError: If retrieval fails.
        """
        if not self._initialized:
            await self.initialize()

        try:
            self.logger.info(
                "Retrieving data with New Storage Backend",
                extra={"key": key}
            )

            # Retrieve data
            # ...

            self.logger.info(
                "Successfully retrieved data with New Storage Backend",
                extra={"key": key}
            )

            return data

        except Exception as e:
            self.logger.error(f"Error retrieving data with New Storage Backend: {e}")
            raise StorageError(f"Retrieval failed: {e}")

    async def delete(self, key: str) -> bool:
        """Delete data by key.

        Args:
            key: Storage key.

        Returns:
            True if deleted, False if not found.

        Raises:
            StorageError: If deletion fails.
        """
        if not self._initialized:
            await self.initialize()

        try:
            self.logger.info(
                "Deleting data with New Storage Backend",
                extra={"key": key}
            )

            # Delete data
            # ...

            self.logger.info(
                "Successfully deleted data with New Storage Backend",
                extra={"key": key}
            )

            return True

        except Exception as e:
            self.logger.error(f"Error deleting data with New Storage Backend: {e}")
            raise StorageError(f"Deletion failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Storage key.

        Returns:
            True if exists, False otherwise.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check existence
            # ...
            return True
        except Exception:
            return False

    async def list_keys(self, prefix: str = "") -> Sequence[str]:
        """List keys with optional prefix.

        Args:
            prefix: Optional key prefix.

        Returns:
            List of keys.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # List keys
            # ...
            return []
        except Exception as e:
            self.logger.error(f"Error listing keys with New Storage Backend: {e}")
            raise StorageError(f"Key listing failed: {e}")

    async def health_check(self) -> bool:
        """Check storage backend health.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            # Implement health check logic
            # ...
            return True
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup storage backend resources."""
        if not self._initialized:
            return

        self.logger.info("Cleaning up New Storage Backend")

        # Cleanup resources
        # ...

        self._initialized = False
        self.logger.info("New Storage Backend cleaned up successfully")

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_new_storage_backend(config: NewStorageConfig) -> NewStorageBackend:
    """Create and configure New Storage Backend.

    Args:
        config: Storage configuration.

    Returns:
        Configured storage backend instance.
    """
    return NewStorageBackend(config)

def load_new_storage_config(config_data: Mapping[str, Any]) -> NewStorageConfig:
    """Load storage configuration from data.

    Args:
        config_data: Configuration data.

    Returns:
        Validated configuration.
    """
    return NewStorageConfig(**config_data)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def validate_storage_config(config: NewStorageConfig) -> bool:
    """Validate storage configuration.

    Args:
        config: Configuration to validate.

    Returns:
        True if valid, False otherwise.
    """
    # Implement validation logic
    return config.enabled and config.connection_string

def format_storage_error(error: Exception) -> str:
    """Format storage error for logging.

    Args:
        error: Error to format.

    Returns:
        Formatted error message.
    """
    return f"Storage error: {type(error).__name__}: {error}"

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "NewStorageBackend",
    "NewStorageConfig",
    "NewStorageMetadata",
    "NewStorageInterface",
    "create_new_storage_backend",
    "load_new_storage_config",
    "validate_storage_config",
    "format_storage_error",
]
```

### Testing Patterns

#### Unit Tests

- Test storage operations with mocked backends
- Test error handling and recovery
- Test configuration validation
- Test health check functionality

#### Integration Tests

- Test storage with real backend systems
- Test data persistence and retrieval
- Test error propagation and handling
- Test performance characteristics

#### Contract Tests

- Test storage interface compliance
- Test data format validation
- Test error type inheritance
- Test configuration schema validation

## Adding New Utility Functions

### Overview

Utility functions provide common functionality used across the Medical KG system. All utility functions must follow the established patterns and be properly documented.

### Implementation Steps

#### 1. Create Utility Module Structure

```python
# src/Medical_KG_rev/utils/new_utility.py
"""New utility functions implementation.

This utility module provides [description of utility functionality]
for the Medical KG system.

Key Responsibilities:
    - [Responsibility 1]
    - [Responsibility 2]
    - [Responsibility 3]

Collaborators:
    - Upstream: [Modules that use these utilities]
    - Downstream: [Modules these utilities depend on]

Side Effects:
    - [List any side effects]

Thread Safety:
    - [Thread safety description]

Performance Characteristics:
    - [Performance characteristics]

Example:
    >>> from Medical_KG_rev.utils.new_utility import new_utility_function
    >>> result = new_utility_function(data)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

import asyncio
import logging
from typing import Any
from collections.abc import Mapping, Sequence, Callable

from pydantic import BaseModel, Field

from Medical_KG_rev.utils.errors import UtilityError

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
type UtilityConfig = dict[str, Any]
type UtilityData = dict[str, Any]

# ==============================================================================
# DATA MODELS
# ==============================================================================
class NewUtilityConfig(BaseModel):
    """Configuration for New Utility."""

    enabled: bool = Field(default=True, description="Whether utility is enabled")
    timeout: int = Field(default=30, description="Utility timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    batch_size: int = Field(default=100, description="Batch processing size")

class NewUtilityRequest(BaseModel):
    """Request model for New Utility."""

    data: UtilityData = Field(..., description="Data to process")
    options: Mapping[str, Any] = Field(default_factory=dict, description="Processing options")

class NewUtilityResponse(BaseModel):
    """Response model for New Utility."""

    result: Any = Field(..., description="Processing result")
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="Response metadata")
    processing_time: float = Field(..., description="Processing time in seconds")

# ==============================================================================
# EXCEPTIONS
# ==============================================================================
class NewUtilityError(UtilityError):
    """New utility specific error."""
    pass

class NewUtilityTimeoutError(NewUtilityError):
    """Timeout error for New Utility."""
    pass

class NewUtilityConfigError(NewUtilityError):
    """Configuration error for New Utility."""
    pass

# ==============================================================================
# HELPER CLASSES
# ==============================================================================
class NewUtilityHelper:
    """Helper class for New Utility operations."""

    def __init__(self, config: NewUtilityConfig) -> None:
        """Initialize helper with configuration.

        Args:
            config: Utility configuration.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def process_data(self, data: UtilityData) -> Any:
        """Process data using the utility.

        Args:
            data: Data to process.

        Returns:
            Processed result.
        """
        # Implement processing logic
        # ...
        return data

    async def validate_config(self) -> bool:
        """Validate utility configuration.

        Returns:
            True if valid, False otherwise.
        """
        # Implement validation logic
        # ...
        return True

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
async def new_utility_function(data: UtilityData,
                             config: NewUtilityConfig = None) -> NewUtilityResponse:
    """New utility function for processing data.

    Args:
        data: Data to process.
        config: Optional utility configuration.

    Returns:
        Utility response with result.

    Raises:
        NewUtilityError: If processing fails.
    """
    if config is None:
        config = NewUtilityConfig()

    start_time = asyncio.get_event_loop().time()

    try:
        logger = logging.getLogger(__name__)
        logger.info("Processing data with New Utility Function")

        # Process data
        helper = NewUtilityHelper(config)
        result = await helper.process_data(data)

        processing_time = asyncio.get_event_loop().time() - start_time

        logger.info(
            "Successfully processed data with New Utility Function",
            extra={"processing_time": processing_time}
        )

        return NewUtilityResponse(
            result=result,
            metadata={
                "utility": "new_utility",
                "version": "1.0.0",
                "processed_at": asyncio.get_event_loop().time()
            },
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing data with New Utility Function: {e}")
        raise NewUtilityError(f"Processing failed: {e}")

def new_sync_utility_function(data: UtilityData,
                            config: NewUtilityConfig = None) -> NewUtilityResponse:
    """Synchronous version of new utility function.

    Args:
        data: Data to process.
        config: Optional utility configuration.

    Returns:
        Utility response with result.

    Raises:
        NewUtilityError: If processing fails.
    """
    if config is None:
        config = NewUtilityConfig()

    try:
        logger = logging.getLogger(__name__)
        logger.info("Processing data with New Sync Utility Function")

        # Process data synchronously
        # ...
        result = data

        logger.info("Successfully processed data with New Sync Utility Function")

        return NewUtilityResponse(
            result=result,
            metadata={
                "utility": "new_sync_utility",
                "version": "1.0.0"
            },
            processing_time=0.0
        )

    except Exception as e:
        logger.error(f"Error processing data with New Sync Utility Function: {e}")
        raise NewUtilityError(f"Processing failed: {e}")

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def create_new_utility_config(config_data: Mapping[str, Any]) -> NewUtilityConfig:
    """Create utility configuration from data.

    Args:
        config_data: Configuration data.

    Returns:
        Validated configuration.
    """
    return NewUtilityConfig(**config_data)

def create_new_utility_helper(config: NewUtilityConfig) -> NewUtilityHelper:
    """Create utility helper instance.

    Args:
        config: Utility configuration.

    Returns:
        Configured helper instance.
    """
    return NewUtilityHelper(config)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def validate_utility_config(config: NewUtilityConfig) -> bool:
    """Validate utility configuration.

    Args:
        config: Configuration to validate.

    Returns:
        True if valid, False otherwise.
    """
    # Implement validation logic
    return config.enabled and config.timeout > 0

def format_utility_error(error: Exception) -> str:
    """Format utility error for logging.

    Args:
        error: Error to format.

    Returns:
        Formatted error message.
    """
    return f"Utility error: {type(error).__name__}: {error}"

def safe_utility_call(func: Callable[[], Any], default: Any = None) -> Any:
    """Safely call utility function with error handling.

    Args:
        func: Function to call.
        default: Default value if function fails.

    Returns:
        Function result or default value.
    """
    try:
        return func()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Utility function failed: {e}")
        return default

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "NewUtilityConfig",
    "NewUtilityRequest",
    "NewUtilityResponse",
    "NewUtilityHelper",
    "NewUtilityError",
    "NewUtilityTimeoutError",
    "NewUtilityConfigError",
    "new_utility_function",
    "new_sync_utility_function",
    "create_new_utility_config",
    "create_new_utility_helper",
    "validate_utility_config",
    "format_utility_error",
    "safe_utility_call",
]
```

### Testing Patterns

#### Unit Tests

- Test utility functions with various inputs
- Test error handling and recovery
- Test configuration validation
- Test helper class functionality

#### Integration Tests

- Test utilities within the system
- Test utility dependencies
- Test error propagation and handling
- Test performance characteristics

#### Contract Tests

- Test utility interface compliance
- Test input/output format validation
- Test error type inheritance
- Test configuration schema validation

## Testing Patterns

### General Testing Principles

1. **Test Coverage**: Aim for 100% test coverage for new components
2. **Test Isolation**: Each test should be independent and not affect others
3. **Test Data**: Use realistic test data that reflects production usage
4. **Test Documentation**: Document test purpose and expected behavior
5. **Test Maintenance**: Keep tests up-to-date with code changes

### Unit Testing

#### Test Structure

```python
# tests/domain/test_component.py
"""Tests for Component."""

import pytest
from unittest.mock import AsyncMock, patch
from Medical_KG_rev.domain.component import Component

# ==============================================================================
# FIXTURES
# ==============================================================================
@pytest.fixture
def component_config():
    """Fixture for component configuration."""
    return ComponentConfig(
        enabled=True,
        timeout=30,
        max_retries=3
    )

@pytest.fixture
def component(component_config):
    """Fixture for component instance."""
    return Component(component_config)

# ==============================================================================
# UNIT TESTS - Component
# ==============================================================================
class TestComponent:
    """Tests for Component."""

    def test_initialization(self, component_config):
        """Test component initialization."""
        component = Component(component_config)
        assert component.config == component_config
        assert component.enabled is True

    @pytest.mark.asyncio
    async def test_successful_operation(self, component):
        """Test successful operation."""
        # Test implementation
        pass

    @pytest.mark.asyncio
    async def test_error_handling(self, component):
        """Test error handling."""
        # Test implementation
        pass

    def test_configuration_validation(self, component_config):
        """Test configuration validation."""
        # Test implementation
        pass

# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================
class TestComponentIntegration:
    """Integration tests for Component."""

    @pytest.mark.asyncio
    async def test_end_to_end_operation(self, component):
        """Test end-to-end operation."""
        # Test implementation
        pass

    @pytest.mark.asyncio
    async def test_error_propagation(self, component):
        """Test error propagation."""
        # Test implementation
        pass

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def create_test_data() -> dict[str, Any]:
    """Create test data for assertions."""
    return {
        "id": "test-id",
        "data": "test-data",
        "metadata": {"test": True}
    }
```

#### Mocking Patterns

```python
# Mock external dependencies
with patch('Medical_KG_rev.external.api') as mock_api:
    mock_api.get.return_value = {"status": "success"}
    result = await component.operation()
    assert result.success

# Mock async operations
mock_operation = AsyncMock(return_value={"result": "success"})
with patch.object(component, 'operation', mock_operation):
    result = await component.operation()
    assert result["result"] == "success"

# Mock configuration
with patch('Medical_KG_rev.config.load') as mock_load:
    mock_load.return_value = {"enabled": True}
    config = load_config()
    assert config["enabled"] is True
```

### Integration Testing

#### Test Environment Setup

```python
# tests/integration/conftest.py
"""Integration test configuration."""

import pytest
import asyncio
from Medical_KG_rev.system import System

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_system():
    """Create test system instance."""
    system = System()
    await system.initialize()
    yield system
    await system.cleanup()

@pytest.fixture
async def test_database():
    """Create test database."""
    # Setup test database
    yield
    # Cleanup test database
```

#### Integration Test Structure

```python
# tests/integration/test_system_integration.py
"""Integration tests for system components."""

import pytest
from Medical_KG_rev.system import System

@pytest.mark.asyncio
async def test_system_initialization(test_system):
    """Test system initialization."""
    assert test_system.initialized
    assert test_system.health_check()

@pytest.mark.asyncio
async def test_component_interaction(test_system):
    """Test component interaction."""
    # Test component A calling component B
    result = await test_system.component_a.process()
    assert result.success

    # Verify component B was called
    assert test_system.component_b.was_called

@pytest.mark.asyncio
async def test_error_propagation(test_system):
    """Test error propagation through system."""
    with pytest.raises(SystemError):
        await test_system.failing_operation()
```

### Performance Testing

#### Performance Test Structure

```python
# tests/performance/test_component_performance.py
"""Performance tests for component."""

import pytest
import time
import asyncio
from Medical_KG_rev.component import Component

@pytest.mark.asyncio
async def test_processing_throughput(component):
    """Test processing throughput."""
    # Generate test data
    test_data = [create_test_item() for _ in range(1000)]

    # Measure processing time
    start_time = time.time()
    results = await component.process_batch(test_data)
    end_time = time.time()

    # Verify performance
    processing_time = end_time - start_time
    throughput = len(test_data) / processing_time

    assert throughput > 100  # At least 100 items per second
    assert len(results) == len(test_data)

@pytest.mark.asyncio
async def test_memory_usage(component):
    """Test memory usage."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Process large dataset
    large_data = [create_test_item() for _ in range(10000)]
    await component.process_batch(large_data)

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    # Verify memory usage is reasonable
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
```

### Contract Testing

#### Contract Test Structure

```python
# tests/contract/test_component_contract.py
"""Contract tests for component."""

import pytest
from Medical_KG_rev.component import Component, ComponentInterface

def test_interface_compliance():
    """Test component implements interface correctly."""
    component = Component()

    # Verify all interface methods exist
    assert hasattr(component, 'process')
    assert hasattr(component, 'health_check')
    assert hasattr(component, 'cleanup')

    # Verify method signatures
    import inspect
    process_sig = inspect.signature(component.process)
    assert 'data' in process_sig.parameters
    assert 'options' in process_sig.parameters

def test_error_type_inheritance():
    """Test error type inheritance."""
    from Medical_KG_rev.component import ComponentError

    # Verify error inheritance
    assert issubclass(ComponentError, Exception)

    # Test error instantiation
    error = ComponentError("Test error")
    assert str(error) == "Test error"

def test_response_format_compliance():
    """Test response format compliance."""
    component = Component()

    # Test response format
    response = component.create_response({"result": "success"})
    assert "success" in response
    assert "data" in response
    assert "metadata" in response
```

## Documentation Standards

### Module Documentation

#### Module-Level Docstring

```python
"""Module description.

This module provides [detailed description of functionality, responsibilities, and role in the system].

**Architectural Context:**
- **Layer**: [Gateway/Service/Adapter/Orchestration/KG/Storage/Validation/Utils]
- **Dependencies**: [List of major dependencies]
- **Dependents**: [List of major dependent modules]
- **Design Patterns**: [Patterns used: Factory, Strategy, Observer, etc.]

**Key Components:**
- `[ClassName]`: [Brief description]
- `[FunctionName]`: [Brief description]

**Usage Examples:**
```python
from Medical_KG_rev.module import Component

# Example usage
component = Component()
result = component.operation()
```

**Configuration:**

- Environment variables: `VAR_NAME` ([description])
- Configuration files: `config/file.yaml` ([description])

**Side Effects:**

- [List any side effects: file I/O, network calls, state mutations]

**Thread Safety:**

- [Thread-safe/Not thread-safe/Conditionally thread-safe with explanation]

**Performance Characteristics:**

- Time complexity: [O(n) analysis where applicable]
- Memory usage: [Description of memory patterns]
- Scalability: [Horizontal/vertical scaling characteristics]

**Error Handling:**

- Raises: [List of exceptions with conditions]
- Returns None when: [Conditions]

**Deprecation Warnings:**

- [Any deprecated functionality]

**See Also:**

- Related modules: [Links to related modules]
- Documentation: [Links to relevant docs]

**Authors:**

- [Original author if known]

**Version History:**

- Added in: v[X.Y.Z]
- Last modified: [Date]
"""

```

### Class Documentation

#### Class Docstring
```python
class ExampleClass:
    """Class description.

    [Detailed description of class functionality, design patterns, and usage].

    **Design Pattern:** [Strategy/Factory/Singleton/Observer/etc.]

    **Thread Safety:** [Thread-safe/Not thread-safe with explanation]

    **Lifecycle:**
    1. Initialization via `__init__`
    2. Configuration via `configure()`
    3. Operation via `execute()`
    4. Cleanup via context manager or `close()`

    Attributes:
        attr_name (Type): Description of attribute.
        _private_attr (Type): Description of private attribute.

    Example:
        Basic usage example::

            instance = ExampleClass(param=value)
            result = instance.method()

    Note:
        Important notes about usage or behavior.

    Warning:
        Warnings about potential issues or deprecated usage.

    See Also:
        :class:`RelatedClass`: Related functionality
        :func:`related_function`: Related function
    """
```

### Function Documentation

#### Function Docstring

```python
def example_function(
    param1: str,
    param2: int,
    param3: list[str] | None = None,
) -> dict[str, Any]:
    """Function description.

    [Detailed description of function behavior, algorithm, and usage].

    **Algorithm:**
    1. Step one description
    2. Step two description
    3. Step three description

    **Complexity:**
    - Time: O(n log n)
    - Space: O(n)

    Args:
        param1: Description of param1. Must be non-empty.
        param2: Description of param2. Must be positive.
        param3: Description of param3. Defaults to empty list if None.

    Returns:
        Dictionary containing:
        - 'key1' (str): Description of key1
        - 'key2' (int): Description of key2
        - 'key3' (list): Description of key3

    Raises:
        ValueError: If param1 is empty or param2 is negative.
        TypeError: If param3 contains non-string elements.
        RuntimeError: If external service unavailable.

    Example:
        Basic usage::

            result = example_function("test", 42)
            print(result['key1'])

        Advanced usage::

            result = example_function(
                param1="test",
                param2=100,
                param3=["a", "b", "c"]
            )

    Note:
        This function makes external API calls and may be slow.
        Consider using async variant for better performance.

    Warning:
        Do not call with param2 > 1000 as it may cause timeout.

    See Also:
        :func:`related_function`: Related functionality
        :func:`async_example_function`: Async variant

    .. versionadded:: 0.1.0
    .. versionchanged:: 0.2.0
        Added param3 parameter for extended functionality.
    .. deprecated:: 0.3.0
        Use :func:`new_function` instead.
    """
```

## Code Review Checklist

### Documentation Review

- [ ] **Module Documentation**: Module has comprehensive docstring
- [ ] **Class Documentation**: All classes have detailed docstrings
- [ ] **Function Documentation**: All functions have complete docstrings
- [ ] **Type Hints**: All functions have complete type annotations
- [ ] **Examples**: Code includes usage examples where appropriate
- [ ] **Error Documentation**: All exceptions are documented

### Code Quality Review

- [ ] **Section Headers**: Code follows section header standards
- [ ] **Import Organization**: Imports are properly organized and sorted
- [ ] **Method Ordering**: Methods are ordered according to standards
- [ ] **Error Handling**: Proper error handling and exception types
- [ ] **Logging**: Appropriate logging with correlation IDs
- [ ] **Performance**: No obvious performance issues

### Testing Review

- [ ] **Test Coverage**: Adequate test coverage for new code
- [ ] **Test Quality**: Tests are well-written and comprehensive
- [ ] **Test Isolation**: Tests are independent and don't affect each other
- [ ] **Test Data**: Tests use realistic data
- [ ] **Mock Usage**: Appropriate use of mocks for external dependencies
- [ ] **Integration Tests**: Integration tests for component interactions

### Security Review

- [ ] **Input Validation**: All inputs are properly validated
- [ ] **Authentication**: Proper authentication where required
- [ ] **Authorization**: Proper authorization checks
- [ ] **Data Sanitization**: Data is properly sanitized
- [ ] **Secret Management**: Secrets are properly managed
- [ ] **Audit Logging**: Sensitive operations are logged

### Performance Review

- [ ] **Algorithm Efficiency**: Algorithms are efficient
- [ ] **Memory Usage**: Memory usage is reasonable
- [ ] **Database Queries**: Database queries are optimized
- [ ] **Caching**: Appropriate caching strategies
- [ ] **Async Operations**: Proper use of async operations
- [ ] **Resource Management**: Resources are properly managed

### Architecture Review

- [ ] **Design Patterns**: Appropriate design patterns are used
- [ ] **Interface Compliance**: Components implement interfaces correctly
- [ ] **Dependency Management**: Dependencies are properly managed
- [ ] **Configuration**: Configuration is properly handled
- [ ] **Error Propagation**: Errors are properly propagated
- [ ] **Lifecycle Management**: Component lifecycle is properly managed

### Compliance Review

- [ ] **Standards Compliance**: Code follows established standards
- [ ] **Formatting**: Code is properly formatted
- [ ] **Linting**: Code passes all linting checks
- [ ] **Type Checking**: Code passes type checking
- [ ] **Documentation**: Documentation is complete and accurate
- [ ] **Version Control**: Changes are properly committed

This comprehensive guide provides the foundation for extending the Medical_KG_rev repository with new components while maintaining consistency, quality, and compliance with established standards.
