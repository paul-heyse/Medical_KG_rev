# Gateway API Reference

The Gateway layer provides a multi-protocol API façade exposing the Medical KG system through REST, GraphQL, gRPC, SOAP, and AsyncAPI/SSE protocols.

## Core Gateway Components

### Application Factory

::: Medical_KG_rev.gateway.app
    options:
      show_root_heading: true
      members:
        - create_app
        - JSONAPIResponseMiddleware
        - SecurityHeadersMiddleware
        - create_problem_response

### Shared Models

::: Medical_KG_rev.gateway.models
    options:
      show_root_heading: true
      members:
        - OperationRequest
        - OperationResponse
        - RetrievalRequest
        - RetrievalResponse
        - NamespaceRequest
        - NamespaceResponse
        - EvaluationRequest
        - EvaluationResponse
        - build_batch_result

### Command Line Interface

::: Medical_KG_rev.gateway.main
    options:
      show_root_heading: true
      members:
        - export_openapi
        - export_graphql
        - export_asyncapi
        - main

### Middleware Components

::: Medical_KG_rev.gateway.middleware
    options:
      show_root_heading: true
      members:
        - CacheEntry
        - CachePolicy
        - CachingMiddleware
        - TenantValidationMiddleware
        - ResponseCache

## REST API Components

### REST Router

::: Medical_KG_rev.gateway.rest.router
    options:
      show_root_heading: true
      members:
        - router
        - get_adapter_endpoints
        - get_ingestion_endpoints
        - get_job_management_endpoints
        - get_processing_endpoints
        - get_namespace_endpoints
        - get_extraction_endpoints
        - get_audit_endpoints
        - get_health_endpoints

## Presentation Layer

### Presentation Interfaces

::: Medical_KG_rev.gateway.presentation.interface
    options:
      show_root_heading: true
      members:
        - ResponsePresenter
        - RequestParser

### JSON:API Implementation

::: Medical_KG_rev.gateway.presentation.jsonapi
    options:
      show_root_heading: true
      members:
        - JSONAPIPresenter
        - _normalise_payload

### Request Lifecycle Management

::: Medical_KG_rev.gateway.presentation.lifecycle
    options:
      show_root_heading: true
      members:
        - RequestLifecycle
        - current_lifecycle
        - push_lifecycle
        - pop_lifecycle
        - RequestLifecycleMiddleware

## Usage Examples

### Basic Gateway Setup

```python
from Medical_KG_rev.gateway.app import create_app

# Create FastAPI application
app = create_app()

# Start server
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### REST API Usage

```python
import httpx

# Health check
async with httpx.AsyncClient() as client:
    response = await client.get("http://localhost:8000/health")
    print(response.json())

# Namespace operations
response = await client.post(
    "http://localhost:8000/v1/namespaces",
    json={
        "data": {
            "type": "namespace",
            "attributes": {
                "name": "test-namespace",
                "description": "Test namespace"
            }
        }
    }
)
```

### GraphQL Usage

```python
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

# Setup GraphQL client
transport = AIOHTTPTransport(url="http://localhost:8000/graphql")
client = Client(transport=transport, fetch_schema_from_transport=True)

# Query documents
query = gql("""
    query GetDocuments($namespace: String!) {
        documents(namespace: $namespace) {
            id
            title
            content
        }
    }
""")

result = client.execute(query, variable_values={"namespace": "test-namespace"})
```

## Configuration

### Environment Variables

- `GATEWAY_HOST`: Host address (default: "0.0.0.0")
- `GATEWAY_PORT`: Port number (default: 8000)
- `GATEWAY_WORKERS`: Number of worker processes (default: 1)
- `GATEWAY_LOG_LEVEL`: Logging level (default: "info")
- `GATEWAY_CORS_ORIGINS`: Allowed CORS origins (default: "*")

### Middleware Configuration

```python
# Configure caching middleware
CACHING_CONFIG = {
    "default_ttl": 300,  # 5 minutes
    "max_size": 1000,    # Maximum cache entries
    "policy": "lru"      # Least Recently Used eviction
}

# Configure tenant validation
TENANT_CONFIG = {
    "required_scopes": ["read", "write"],
    "validate_tenant": True,
    "tenant_header": "X-Tenant-ID"
}
```

## Error Handling

The Gateway provides comprehensive error handling with:

- **HTTP Problem Details**: RFC 7807 compliant error responses
- **Error Translation**: Domain exceptions translated to HTTP status codes
- **Request Correlation**: All errors include correlation IDs for tracing
- **Structured Logging**: Errors logged with context and stack traces

### Error Response Format

```json
{
    "type": "https://example.com/problems/validation-error",
    "title": "Validation Error",
    "status": 400,
    "detail": "Invalid request parameters",
    "instance": "/v1/documents",
    "correlation_id": "req-123456789"
}
```

## Performance Considerations

- **Connection Pooling**: HTTP clients use connection pooling for external services
- **Response Caching**: Implemented with configurable TTL and eviction policies
- **Request Batching**: Support for batch operations to reduce overhead
- **Async Processing**: All I/O operations are asynchronous for better concurrency

## Security Features

- **Authentication**: OAuth 2.0 with JWT tokens
- **Authorization**: Scope-based access control
- **Multi-tenancy**: Tenant isolation at the application level
- **Rate Limiting**: Per-client and per-endpoint rate limiting
- **Security Headers**: Comprehensive security headers middleware
- **Input Validation**: Pydantic-based request validation
