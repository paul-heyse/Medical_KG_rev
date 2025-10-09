# Adapters Development Guide

## Overview

The Adapters layer provides integration with external data sources and APIs. It handles data fetching, transformation, and error handling for various biomedical data sources including OpenAlex, CORE, PubMed Central, and Unpaywall.

## Architecture

Adapters follow a consistent pattern:

- **Base Adapter** (`base.py`): Abstract base classes and interfaces
- **YAML Parser** (`yaml_parser.py`): Configuration parsing and validation
- **Biomedical Adapters** (`biomedical.py`): Core biomedical data adapters
- **Domain-Specific Adapters**: Specialized adapters for specific data sources

## Key Components

### Base Adapter Interface

All adapters implement the base interface:

```python
from abc import ABC, abstractmethod
from typing import Protocol, Any, Dict, List, Optional

class AdapterInterface(Protocol):
    """Base interface for all adapters."""

    def initialize(self) -> None:
        """Initialize the adapter."""
        ...

    def fetch(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch data from external source."""
        ...

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform external data to internal format."""
        ...

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate transformed data."""
        ...

    def cleanup(self) -> None:
        """Cleanup adapter resources."""
        ...

class BaseAdapter(ABC):
    """Abstract base adapter implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the adapter."""
        pass

    @abstractmethod
    def fetch(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch data from external source."""
        pass

    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform external data to internal format."""
        pass

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate transformed data."""
        return True

    def cleanup(self) -> None:
        """Cleanup adapter resources."""
        if self.client:
            self.client.close()
```

### OpenAlex Adapter

The OpenAlex adapter (`src/Medical_KG_rev/adapters/openalex/adapter.py`) provides:

```python
from pyalex import Works, Authors, Institutions
from Medical_KG_rev.adapters.base import BaseAdapter

class OpenAlexAdapter(BaseAdapter):
    """OpenAlex adapter for scholarly works metadata."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.works_client = None
        self.authors_client = None
        self.institutions_client = None

    def initialize(self) -> None:
        """Initialize OpenAlex clients."""
        self.works_client = Works()
        self.authors_client = Authors()
        self.institutions_client = Institutions()

    def fetch(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch works from OpenAlex."""
        if not self.works_client:
            raise AdapterError("Adapter not initialized")

        try:
            results = self.works_client.search(**query)
            return list(results)
        except Exception as e:
            raise AdapterError(f"Failed to fetch from OpenAlex: {str(e)}")

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenAlex data to internal format."""
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "abstract": data.get("abstract"),
            "authors": [author.get("display_name") for author in data.get("authorships", [])],
            "publication_date": data.get("publication_date"),
            "doi": data.get("doi"),
            "open_access": data.get("open_access", {}).get("is_oa", False)
        }
```

### CORE Adapter

The CORE adapter (`src/Medical_KG_rev/adapters/core/adapter.py`) handles:

```python
import httpx
from Medical_KG_rev.adapters.base import BaseAdapter

class CoreAdapter(BaseAdapter):
    """CORE adapter for open research papers."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = "https://core.ac.uk/api-v2"

    def initialize(self) -> None:
        """Initialize HTTP client."""
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

    async def fetch(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch papers from CORE."""
        if not self.client:
            raise AdapterError("Adapter not initialized")

        try:
            response = await self.client.get(
                f"{self.base_url}/search",
                params=query
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            raise AdapterError(f"Failed to fetch from CORE: {str(e)}")

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform CORE data to internal format."""
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "abstract": data.get("abstract"),
            "authors": data.get("authors", []),
            "publication_date": data.get("publishedDate"),
            "doi": data.get("doi"),
            "url": data.get("downloadUrl")
        }
```

## Development Standards

### Section Headers

All Adapter modules must follow the `ADAPTER_SECTIONS` standard:

```python
# ==============================================================================
# IMPORTS
# ==============================================================================

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

# ==============================================================================
# DATA MODELS
# ==============================================================================

# ==============================================================================
# ERROR HANDLING
# ==============================================================================

# ==============================================================================
# ADAPTER IMPLEMENTATION
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
```

### Error Handling

Adapters must implement comprehensive error handling:

```python
from enum import Enum
from typing import Optional, Dict, Any

class AdapterErrorType(Enum):
    """Adapter error types."""
    INITIALIZATION_ERROR = "initialization_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    DATA_ERROR = "data_error"
    TRANSFORMATION_ERROR = "transformation_error"

class AdapterError(Exception):
    """Base adapter error."""

    def __init__(
        self,
        error_type: AdapterErrorType,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(message)

def handle_adapter_error(func):
    """Decorator for adapter error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except httpx.HTTPError as e:
            raise AdapterError(
                AdapterErrorType.NETWORK_ERROR,
                f"Network error in {func.__name__}: {str(e)}"
            )
        except Exception as e:
            if isinstance(e, AdapterError):
                raise
            else:
                raise AdapterError(
                    AdapterErrorType.DATA_ERROR,
                    f"Unexpected error in {func.__name__}: {str(e)}"
                )
    return wrapper
```

### Rate Limiting

Implement rate limiting for external APIs:

```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, max_requests: int, time_window: timedelta):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[datetime] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            now = datetime.now()
            # Remove old requests
            self.requests = [
                req for req in self.requests
                if now - req < self.time_window
            ]

            if len(self.requests) >= self.max_requests:
                sleep_time = (self.requests[0] + self.time_window - now).total_seconds()
                await asyncio.sleep(sleep_time)

            self.requests.append(now)

class RateLimitedAdapter(BaseAdapter):
    """Adapter with rate limiting."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rate_limiter = RateLimiter(
            max_requests=config.get("max_requests", 100),
            time_window=timedelta(minutes=1)
        )

    async def fetch(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch data with rate limiting."""
        await self.rate_limiter.acquire()
        return await super().fetch(query)
```

## Configuration Management

### YAML Configuration

Adapters use YAML configuration files:

```yaml
# config/adapters/openalex.yaml
adapter:
  name: "openalex"
  type: "biomedical"
  base_url: "https://api.openalex.org"
  timeout: 30
  retry_attempts: 3
  rate_limit:
    max_requests: 100
    time_window: "1m"
  authentication:
    type: "none"
  data_mapping:
    id: "id"
    title: "title"
    abstract: "abstract"
    authors: "authorships"
    publication_date: "publication_date"
    doi: "doi"
```

### Configuration Loading

Load adapter configuration:

```python
import yaml
from pathlib import Path
from typing import Dict, Any

def load_adapter_config(adapter_name: str) -> Dict[str, Any]:
    """Load adapter configuration from YAML file."""
    config_path = Path(f"config/adapters/{adapter_name}.yaml")

    if not config_path.exists():
        raise AdapterError(
            AdapterErrorType.CONFIGURATION_ERROR,
            f"Configuration file not found: {config_path}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config.get("adapter", {})

def create_adapter(adapter_name: str) -> BaseAdapter:
    """Create adapter instance from configuration."""
    config = load_adapter_config(adapter_name)

    if adapter_name == "openalex":
        return OpenAlexAdapter(config)
    elif adapter_name == "core":
        return CoreAdapter(config)
    elif adapter_name == "pmc":
        return PMCAdapter(config)
    elif adapter_name == "unpaywall":
        return UnpaywallAdapter(config)
    else:
        raise AdapterError(
            AdapterErrorType.CONFIGURATION_ERROR,
            f"Unknown adapter: {adapter_name}"
        )
```

## Testing

### Unit Tests

Adapters should have comprehensive unit tests:

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from Medical_KG_rev.adapters.openalex.adapter import OpenAlexAdapter

class TestOpenAlexAdapter:
    """Test cases for OpenAlex adapter."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "base_url": "https://api.openalex.org",
            "timeout": 30
        }
        self.adapter = OpenAlexAdapter(self.config)

    def test_initialization(self):
        """Test adapter initialization."""
        with patch('pyalex.Works') as mock_works:
            self.adapter.initialize()
            mock_works.assert_called_once()

    def test_fetch_success(self):
        """Test successful data fetching."""
        self.adapter.works_client = Mock()
        self.adapter.works_client.search.return_value = [
            {"id": "1", "title": "Test Paper"}
        ]

        result = self.adapter.fetch({"query": "test"})
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_fetch_error(self):
        """Test error handling during fetch."""
        self.adapter.works_client = Mock()
        self.adapter.works_client.search.side_effect = Exception("API Error")

        with pytest.raises(AdapterError) as exc_info:
            self.adapter.fetch({"query": "test"})
        assert exc_info.value.error_type == AdapterErrorType.DATA_ERROR

    def test_transform(self):
        """Test data transformation."""
        input_data = {
            "id": "1",
            "title": "Test Paper",
            "abstract": "Test abstract",
            "authorships": [{"author": {"display_name": "John Doe"}}],
            "publication_date": "2023-01-01",
            "doi": "10.1000/test",
            "open_access": {"is_oa": True}
        }

        result = self.adapter.transform(input_data)
        assert result["id"] == "1"
        assert result["title"] == "Test Paper"
        assert result["authors"] == ["John Doe"]
        assert result["open_access"] is True
```

### Integration Tests

Integration tests should verify external API interactions:

```python
@pytest.mark.integration
async def test_openalex_integration():
    """Test OpenAlex adapter integration."""
    config = load_adapter_config("openalex")
    adapter = OpenAlexAdapter(config)

    try:
        adapter.initialize()
        results = adapter.fetch({"query": "machine learning"})

        assert len(results) > 0
        assert all("id" in result for result in results)

        # Test transformation
        transformed = adapter.transform(results[0])
        assert "id" in transformed
        assert "title" in transformed

    finally:
        adapter.cleanup()
```

## Performance Optimization

### Caching

Implement caching for expensive operations:

```python
from functools import lru_cache
from typing import Dict, Any, List

class CachedAdapter(BaseAdapter):
    """Adapter with caching support."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour

    @lru_cache(maxsize=1000)
    def get_cached_data(self, query_hash: str) -> List[Dict[str, Any]]:
        """Get cached data if available."""
        if query_hash in self._cache:
            return self._cache[query_hash]
        return None

    def cache_data(self, query_hash: str, data: List[Dict[str, Any]]) -> None:
        """Cache data for future use."""
        self._cache[query_hash] = data
```

### Batch Processing

Implement batch processing for efficiency:

```python
from typing import List, Dict, Any
import asyncio

class BatchAdapter(BaseAdapter):
    """Adapter with batch processing support."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.batch_size = config.get("batch_size", 10)

    async def fetch_batch(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch data for multiple queries in batches."""
        results = []

        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_results = await asyncio.gather(
                *[self.fetch(query) for query in batch]
            )
            results.extend(batch_results)

        return results
```

## Monitoring and Observability

### Metrics Collection

Collect adapter-specific metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Adapter metrics
ADAPTER_REQUESTS = Counter('adapter_requests_total', 'Total requests', ['adapter', 'operation'])
ADAPTER_DURATION = Histogram('adapter_duration_seconds', 'Adapter duration', ['adapter', 'operation'])
ADAPTER_ERRORS = Counter('adapter_errors_total', 'Total errors', ['adapter', 'error_type'])
ADAPTER_CACHE_HITS = Counter('adapter_cache_hits_total', 'Cache hits', ['adapter'])

def track_adapter_operation(adapter_name: str, operation: str):
    """Track adapter operation metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            ADAPTER_REQUESTS.labels(adapter=adapter_name, operation=operation).inc()
            with ADAPTER_DURATION.labels(adapter=adapter_name, operation=operation).time():
                try:
                    return func(*args, **kwargs)
                except AdapterError as e:
                    ADAPTER_ERRORS.labels(adapter=adapter_name, error_type=e.error_type.value).inc()
                    raise
        return wrapper
    return decorator
```

### Health Checks

Implement health checks for adapters:

```python
from enum import Enum
from typing import Dict, Any

class AdapterHealth:
    """Adapter health checker."""

    def __init__(self, adapter: BaseAdapter):
        self.adapter = adapter

    async def check_health(self) -> Dict[str, Any]:
        """Check adapter health."""
        try:
            # Test basic functionality
            self.adapter.initialize()
            test_query = {"query": "health_check"}
            result = await self.adapter.fetch(test_query)
            self.adapter.cleanup()

            return {
                "status": "healthy",
                "details": {"test_result": "passed", "result_count": len(result)}
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
```

## Deployment

### Docker Configuration

Adapters can be deployed using Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

# Adapter-specific configuration
ENV ADAPTER_NAME=openalex
ENV API_KEY=your_api_key

CMD ["python", "-m", "Medical_KG_rev.adapters.openalex.adapter"]
```

### Environment Variables

Configure adapters using environment variables:

```bash
# OpenAlex adapter
OPENALEX_BASE_URL=https://api.openalex.org
OPENALEX_TIMEOUT=30
OPENALEX_RATE_LIMIT=100

# CORE adapter
CORE_API_KEY=your_core_api_key
CORE_BASE_URL=https://core.ac.uk/api-v2
CORE_TIMEOUT=30

# PMC adapter
PMC_BASE_URL=https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi
PMC_TIMEOUT=30
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Implement proper rate limiting and backoff strategies
2. **Authentication Failures**: Check API keys and authentication configuration
3. **Network Timeouts**: Adjust timeout settings and implement retry logic
4. **Data Transformation Errors**: Validate data schemas and handle missing fields

### Debugging

Enable debug mode for development:

```python
import logging
import os

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Enable adapter debug mode
os.environ["ADAPTER_DEBUG"] = "true"

# Add debug middleware
def debug_adapter(func):
    def wrapper(*args, **kwargs):
        logging.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.debug(f"{func.__name__} returned: {result}")
        return result
    return wrapper
```

## Best Practices

1. **Error Handling**: Implement comprehensive error handling with specific error types
2. **Rate Limiting**: Respect API rate limits and implement backoff strategies
3. **Caching**: Cache expensive operations to reduce API calls
4. **Configuration**: Use structured configuration with validation
5. **Testing**: Write comprehensive unit and integration tests
6. **Monitoring**: Collect metrics and implement health checks
7. **Documentation**: Maintain up-to-date adapter documentation
8. **Security**: Secure API keys and implement proper authentication
9. **Performance**: Use batch processing and async operations where appropriate
10. **Resilience**: Implement retry logic and circuit breakers
