# Gateway Development Guide

## Overview

The Gateway layer serves as the entry point for all external interactions with the Medical Knowledge Graph system. It handles HTTP requests, protocol translation, authentication, and response formatting.

## Architecture

The Gateway follows a layered architecture:

- **Application Layer** (`app.py`): Main FastAPI application setup
- **Models Layer** (`models.py`): Shared Pydantic models
- **Middleware Layer** (`middleware.py`): Request/response processing
- **REST Layer** (`rest/`): REST API endpoints
- **Presentation Layer** (`presentation/`): Response formatting and lifecycle management

## Key Components

### FastAPI Application

The main application is defined in `src/Medical_KG_rev/gateway/app.py`:

```python
from fastapi import FastAPI
from Medical_KG_rev.gateway.middleware import add_middleware
from Medical_KG_rev.gateway.rest.router import create_router

app = FastAPI(
    title="Medical Knowledge Graph Gateway",
    description="Multi-protocol gateway for medical knowledge graph operations",
    version="1.0.0"
)

# Add middleware
add_middleware(app)

# Include routers
app.include_router(create_router(), prefix="/api/v1")
```

### Middleware Stack

The middleware stack includes:

1. **CORS Middleware**: Cross-origin resource sharing
2. **Security Middleware**: Security headers and HTTPS enforcement
3. **Logging Middleware**: Request/response logging
4. **Tenant Validation Middleware**: Multi-tenant request validation
5. **Caching Middleware**: Response caching

### REST API Endpoints

Endpoints are organized by domain:

- `/api/v1/health`: Health check endpoints
- `/api/v1/embeddings`: Embedding operations
- `/api/v1/chunking`: Text chunking operations
- `/api/v1/retrieval`: Retrieval operations
- `/api/v1/mineru`: MinerU document processing

## Development Standards

### Section Headers

All Gateway modules must follow the `GATEWAY_SECTIONS` standard:

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
# MIDDLEWARE IMPLEMENTATION
# ==============================================================================

# ==============================================================================
# ROUTER IMPLEMENTATION
# ==============================================================================

# ==============================================================================
# PRESENTATION LAYER
# ==============================================================================

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
```

### Docstring Standards

All functions, classes, and modules must have comprehensive docstrings:

```python
def process_request(
    request: Request,
    service: ServiceInterface
) -> Response:
    """Process an incoming request through the service layer.

    Args:
        request: The incoming HTTP request
        service: The service interface to process the request

    Returns:
        A formatted response

    Raises:
        ValidationError: If the request is invalid
        ServiceError: If the service fails

    Example:
        >>> request = Request(method="GET", url="/api/v1/health")
        >>> response = process_request(request, health_service)
        >>> assert response.status_code == 200
    """
```

### Error Handling

Gateway modules must implement consistent error handling:

```python
from fastapi import HTTPException
from Medical_KG_rev.gateway.models import ErrorResponse

def handle_service_error(error: ServiceError) -> HTTPException:
    """Convert service errors to HTTP exceptions.

    Args:
        error: The service error to convert

    Returns:
        An HTTP exception with appropriate status code
    """
    if isinstance(error, ValidationError):
        return HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="validation_error",
                message=str(error)
            ).dict()
        )
    elif isinstance(error, NotFoundError):
        return HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="not_found",
                message=str(error)
            ).dict()
        )
    else:
        return HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="internal_error",
                message="An internal error occurred"
            ).dict()
        )
```

## Testing

### Unit Tests

Gateway modules should have comprehensive unit tests:

```python
import pytest
from fastapi.testclient import TestClient
from Medical_KG_rev.gateway.app import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_invalid_endpoint():
    """Test handling of invalid endpoints."""
    response = client.get("/api/v1/invalid")
    assert response.status_code == 404
```

### Integration Tests

Integration tests should verify end-to-end functionality:

```python
def test_embedding_pipeline():
    """Test the complete embedding pipeline."""
    # Test data
    test_data = {
        "text": "Sample medical text",
        "model": "default"
    }

    # Send request
    response = client.post("/api/v1/embeddings", json=test_data)

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert "embeddings" in result
    assert len(result["embeddings"]) > 0
```

## Performance Considerations

### Caching

Implement appropriate caching strategies:

```python
from functools import lru_cache
from Medical_KG_rev.gateway.middleware import CacheMiddleware

@lru_cache(maxsize=1000)
def get_cached_response(key: str) -> Optional[Response]:
    """Get a cached response if available."""
    # Implementation
    pass
```

### Rate Limiting

Implement rate limiting for API endpoints:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/embeddings")
@limiter.limit("100/minute")
async def get_embeddings(request: Request):
    """Get embeddings with rate limiting."""
    # Implementation
    pass
```

## Security

### Authentication

Implement proper authentication:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    """Validate authentication token."""
    if not validate_token(token.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return get_user_from_token(token.credentials)
```

### Input Validation

Validate all inputs:

```python
from pydantic import BaseModel, validator

class EmbeddingRequest(BaseModel):
    text: str
    model: str = "default"

    @validator('text')
    def validate_text(cls, v):
        if len(v) > 10000:
            raise ValueError('Text too long')
        return v
```

## Monitoring and Observability

### Logging

Implement structured logging:

```python
import structlog

logger = structlog.get_logger()

def log_request(request: Request, response: Response):
    """Log request and response details."""
    logger.info(
        "request_processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration=response.headers.get("X-Process-Time")
    )
```

### Metrics

Collect performance metrics:

```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('gateway_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('gateway_request_duration_seconds', 'Request duration')

def track_request(method: str, endpoint: str, duration: float):
    """Track request metrics."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    REQUEST_DURATION.observe(duration)
```

## Deployment

### Docker

Gateway can be deployed using Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

CMD ["uvicorn", "Medical_KG_rev.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Configure using environment variables:

```bash
# Gateway configuration
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8000
GATEWAY_WORKERS=4

# Service endpoints
EMBEDDING_SERVICE_URL=http://embedding-service:8001
CHUNKING_SERVICE_URL=http://chunking-service:8002
RETRIEVAL_SERVICE_URL=http://retrieval-service:8003
```

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure CORS middleware is properly configured
2. **Authentication Failures**: Check token validation logic
3. **Rate Limiting**: Verify rate limit configuration
4. **Memory Issues**: Monitor memory usage and implement caching

### Debugging

Enable debug mode for development:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Enable FastAPI debug mode
app = FastAPI(debug=True)
```

## Best Practices

1. **Consistent Error Handling**: Use standardized error responses
2. **Input Validation**: Validate all inputs using Pydantic models
3. **Security**: Implement proper authentication and authorization
4. **Performance**: Use caching and rate limiting appropriately
5. **Monitoring**: Implement comprehensive logging and metrics
6. **Testing**: Write comprehensive unit and integration tests
7. **Documentation**: Maintain up-to-date API documentation
