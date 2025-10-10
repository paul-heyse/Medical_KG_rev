# Utilities API Reference

The Utilities layer provides common functionality, helper functions, and shared components used across the Medical KG system.

## Core Utility Components

### HTTP Client

::: Medical_KG_rev.utils.http_client
    options:
      show_root_heading: true
      members:
        - HttpClient
        - AsyncHttpClient
        - RetryConfig
        - CircuitBreakerConfig
        - RateLimitConfig
        - BackoffStrategy
        - RetryableHTTPStatus
        - SynchronousLimiter

### Error Handling

::: Medical_KG_rev.utils.errors
    options:
      show_root_heading: true
      members:
        - MedicalKGError
        - ValidationError
        - ConfigurationError
        - NetworkError
        - TimeoutError
        - RateLimitError

## Usage Examples

### HTTP Client Usage

```python
from Medical_KG_rev.utils.http_client import AsyncHttpClient, RetryConfig

# Initialize HTTP client with retry configuration
client = AsyncHttpClient(
    retry_config=RetryConfig(
        max_retries=3,
        backoff_factor=2,
        retry_on_status=[500, 502, 503, 504],
        timeout=30
    ),
    circuit_breaker_config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=httpx.HTTPError
    ),
    rate_limit_config=RateLimitConfig(
        requests_per_minute=100,
        burst_limit=10
    )
)

# Make HTTP requests
try:
    response = await client.get("https://api.example.com/data")
    data = response.json()
    print(f"Received data: {data}")
except httpx.HTTPError as e:
    print(f"HTTP error: {e}")
except httpx.TimeoutException as e:
    print(f"Timeout error: {e}")
```

### Synchronous HTTP Client Usage

```python
from Medical_KG_rev.utils.http_client import HttpClient

# Initialize synchronous HTTP client
client = HttpClient(
    retry_config=RetryConfig(
        max_retries=3,
        backoff_factor=2,
        timeout=30
    )
)

# Make synchronous requests
try:
    response = client.get("https://api.example.com/data")
    data = response.json()
    print(f"Received data: {data}")
except httpx.HTTPError as e:
    print(f"HTTP error: {e}")
```

### Advanced HTTP Client Configuration

```python
from Medical_KG_rev.utils.http_client import (
    AsyncHttpClient,
    RetryConfig,
    CircuitBreakerConfig,
    RateLimitConfig,
    BackoffStrategy
)

# Configure advanced retry strategy
retry_config = RetryConfig(
    max_retries=5,
    backoff_factor=2,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    retry_on_status=[500, 502, 503, 504, 429],
    timeout=60,
    jitter=True
)

# Configure circuit breaker
circuit_breaker_config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=httpx.HTTPError,
    half_open_max_calls=3
)

# Configure rate limiting
rate_limit_config = RateLimitConfig(
    requests_per_minute=100,
    burst_limit=10,
    per_endpoint_limits={
        "https://api.example.com/data": 50,
        "https://api.example.com/search": 200
    }
)

# Initialize client with advanced configuration
client = AsyncHttpClient(
    retry_config=retry_config,
    circuit_breaker_config=circuit_breaker_config,
    rate_limit_config=rate_limit_config,
    headers={
        "User-Agent": "Medical-KG/1.0",
        "Accept": "application/json"
    }
)

# Use client for API calls
async def fetch_medical_data():
    """Fetch medical data from external API."""
    try:
        response = await client.get(
            "https://api.example.com/medical-data",
            params={"limit": 100, "format": "json"}
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"API returned status {response.status_code}")
            return None

    except httpx.HTTPError as e:
        print(f"Failed to fetch medical data: {e}")
        return None
```

### Error Handling with Custom Exceptions

```python
from Medical_KG_rev.utils.errors import (
    MedicalKGError,
    ValidationError,
    ConfigurationError,
    NetworkError,
    TimeoutError,
    RateLimitError
)

def validate_config(config: dict) -> None:
    """Validate configuration with custom exceptions."""
    if not config:
        raise ConfigurationError("Configuration cannot be empty")

    if "api_key" not in config:
        raise ConfigurationError("API key is required")

    if "base_url" not in config:
        raise ConfigurationError("Base URL is required")

    if not config["base_url"].startswith("http"):
        raise ValidationError("Base URL must start with http:// or https://")

def make_api_request(url: str, timeout: int = 30) -> dict:
    """Make API request with custom error handling."""
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()

    except httpx.TimeoutException:
        raise TimeoutError(f"Request to {url} timed out after {timeout} seconds")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise RateLimitError(f"Rate limit exceeded for {url}")
        else:
            raise NetworkError(f"HTTP error {e.response.status_code} for {url}")

    except httpx.RequestError as e:
        raise NetworkError(f"Network error for {url}: {e}")

# Usage
try:
    config = {"api_key": "test-key", "base_url": "https://api.example.com"}
    validate_config(config)

    data = make_api_request(f"{config['base_url']}/data")
    print(f"Received data: {data}")

except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except NetworkError as e:
    print(f"Network error: {e}")
except TimeoutError as e:
    print(f"Timeout error: {e}")
except RateLimitError as e:
    print(f"Rate limit error: {e}")
except MedicalKGError as e:
    print(f"Medical KG error: {e}")
```

### Batch HTTP Requests

```python
import asyncio
from Medical_KG_rev.utils.http_client import AsyncHttpClient

async def fetch_multiple_endpoints(client: AsyncHttpClient, urls: list[str]) -> list[dict]:
    """Fetch data from multiple endpoints concurrently."""
    async def fetch_single(url: str) -> dict:
        try:
            response = await client.get(url)
            return {"url": url, "data": response.json(), "success": True}
        except Exception as e:
            return {"url": url, "error": str(e), "success": False}

    # Execute all requests concurrently
    tasks = [fetch_single(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results

# Usage
client = AsyncHttpClient()
urls = [
    "https://api.example.com/patients",
    "https://api.example.com/observations",
    "https://api.example.com/medications"
]

results = await fetch_multiple_endpoints(client, urls)
for result in results:
    if result["success"]:
        print(f"Successfully fetched {result['url']}: {len(result['data'])} items")
    else:
        print(f"Failed to fetch {result['url']}: {result['error']}")
```

### HTTP Client with Authentication

```python
from Medical_KG_rev.utils.http_client import AsyncHttpClient

class AuthenticatedClient(AsyncHttpClient):
    """HTTP client with authentication support."""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Make authenticated GET request."""
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {self.api_key}"
        kwargs["headers"] = headers
        return await super().get(url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Make authenticated POST request."""
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {self.api_key}"
        kwargs["headers"] = headers
        return await super().post(url, **kwargs)

# Usage
client = AuthenticatedClient(
    api_key="your-api-key",
    retry_config=RetryConfig(max_retries=3)
)

# Make authenticated requests
response = await client.get("https://api.example.com/protected-data")
data = response.json()
```

### Rate Limiting and Circuit Breaker Monitoring

```python
from Medical_KG_rev.utils.http_client import AsyncHttpClient

class MonitoredClient(AsyncHttpClient):
    """HTTP client with monitoring capabilities."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.request_count = 0
        self.error_count = 0
        self.circuit_breaker_state = "closed"

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Make GET request with monitoring."""
        self.request_count += 1

        try:
            response = await super().get(url, **kwargs)
            return response
        except Exception as e:
            self.error_count += 1
            raise

    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "circuit_breaker_state": self.circuit_breaker_state
        }

# Usage
client = MonitoredClient()
response = await client.get("https://api.example.com/data")
stats = client.get_stats()
print(f"Client stats: {stats}")
```

## Configuration

### HTTP Client Configuration

```python
# HTTP client configuration
HTTP_CLIENT_CONFIG = {
    "retry_config": {
        "max_retries": 3,
        "backoff_factor": 2,
        "backoff_strategy": "exponential",
        "retry_on_status": [500, 502, 503, 504, 429],
        "timeout": 30,
        "jitter": True
    },
    "circuit_breaker_config": {
        "failure_threshold": 5,
        "recovery_timeout": 60,
        "expected_exception": "httpx.HTTPError",
        "half_open_max_calls": 3
    },
    "rate_limit_config": {
        "requests_per_minute": 100,
        "burst_limit": 10,
        "per_endpoint_limits": {
            "https://api.example.com/data": 50,
            "https://api.example.com/search": 200
        }
    },
    "default_headers": {
        "User-Agent": "Medical-KG/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
}
```

### Environment Variables

- `HTTP_CLIENT_TIMEOUT`: Default HTTP client timeout
- `HTTP_CLIENT_MAX_RETRIES`: Default maximum retry attempts
- `HTTP_CLIENT_RATE_LIMIT`: Default rate limit (requests per minute)
- `HTTP_CLIENT_CIRCUIT_BREAKER_THRESHOLD`: Circuit breaker failure threshold
- `HTTP_CLIENT_BACKOFF_FACTOR`: Backoff factor for retries

## Error Handling

### Utility Error Hierarchy

```python
# Base utility error
class UtilityError(Exception):
    """Base exception for utility errors."""
    pass

# HTTP client errors
class HTTPClientError(UtilityError):
    """HTTP client-specific errors."""
    pass

# Configuration errors
class ConfigurationError(UtilityError):
    """Configuration-related errors."""
    pass

# Validation errors
class ValidationError(UtilityError):
    """Validation-related errors."""
    pass
```

### Error Handling Patterns

```python
from Medical_KG_rev.utils.errors import MedicalKGError

def safe_http_request(client: AsyncHttpClient, url: str) -> dict:
    """Make safe HTTP request with comprehensive error handling."""
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

    except httpx.TimeoutException as e:
        logger.error(f"Request timeout for {url}: {e}")
        raise TimeoutError(f"Request to {url} timed out")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning(f"Rate limit exceeded for {url}")
            raise RateLimitError(f"Rate limit exceeded for {url}")
        elif e.response.status_code >= 500:
            logger.error(f"Server error for {url}: {e.response.status_code}")
            raise NetworkError(f"Server error {e.response.status_code} for {url}")
        else:
            logger.error(f"HTTP error for {url}: {e.response.status_code}")
            raise NetworkError(f"HTTP error {e.response.status_code} for {url}")

    except httpx.RequestError as e:
        logger.error(f"Request error for {url}: {e}")
        raise NetworkError(f"Request error for {url}: {e}")

    except Exception as e:
        logger.error(f"Unexpected error for {url}: {e}")
        raise MedicalKGError(f"Unexpected error for {url}: {e}")
```

## Performance Considerations

- **Connection Pooling**: HTTP clients use connection pooling for efficiency
- **Async Operations**: All HTTP operations are asynchronous
- **Circuit Breakers**: Protection against failing external services
- **Rate Limiting**: Built-in rate limiting to respect API limits
- **Retry Logic**: Configurable retry logic with exponential backoff
- **Caching**: Response caching to reduce API calls

## Monitoring and Observability

- **Request Metrics**: Track request count, latency, and success rate
- **Error Tracking**: Monitor error rates and types
- **Circuit Breaker Status**: Track circuit breaker state changes
- **Rate Limit Usage**: Monitor rate limit consumption
- **Distributed Tracing**: OpenTelemetry spans for HTTP requests
- **Structured Logging**: Comprehensive logging with correlation IDs
- **Health Checks**: HTTP client health check endpoints

## Testing

### Mock HTTP Client

```python
from Medical_KG_rev.utils.http_client import AsyncHttpClient
import httpx_mock

class MockHttpClient(AsyncHttpClient):
    """Mock HTTP client for testing."""

    def __init__(self):
        self.requests = []
        self.responses = {}

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Mock GET request."""
        self.requests.append({"method": "GET", "url": url, "kwargs": kwargs})

        if url in self.responses:
            return self.responses[url]
        else:
            # Default mock response
            return httpx.Response(
                status_code=200,
                json={"message": "Mock response"},
                request=httpx.Request("GET", url)
            )

    def set_response(self, url: str, response: httpx.Response):
        """Set mock response for URL."""
        self.responses[url] = response
```

### Integration Tests

```python
import pytest
from Medical_KG_rev.utils.http_client import AsyncHttpClient

@pytest.mark.asyncio
async def test_http_client():
    """Test HTTP client functionality."""
    client = AsyncHttpClient()

    # Test successful request
    with httpx_mock.mock() as mock:
        mock.get(
            "https://api.example.com/data",
            json={"message": "Success"}
        )

        response = await client.get("https://api.example.com/data")
        assert response.status_code == 200
        assert response.json()["message"] == "Success"

    # Test error handling
    with httpx_mock.mock() as mock:
        mock.get(
            "https://api.example.com/error",
            status_code=500
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.get("https://api.example.com/error")
```
