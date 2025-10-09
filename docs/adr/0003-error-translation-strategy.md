# ADR-0003: Error Translation Strategy

## Status

Accepted

## Context

The Medical KG pipeline needed consistent error responses across different protocols (REST, GraphQL, gRPC). The existing error handling was inconsistent and protocol-specific, leading to:

- Different error formats across protocols
- Inconsistent error messages and codes
- Difficult error debugging and monitoring
- Poor client experience with unpredictable error responses
- Lack of standardized retry strategies
- Inconsistent error logging and metrics

The system required a unified approach to error handling that would:

- Provide consistent error responses across all protocols
- Follow industry standards for HTTP error responses
- Enable proper error monitoring and alerting
- Support client retry logic with appropriate hints
- Centralize error translation logic
- Provide clear error messages for debugging

## Decision

We will implement a **Centralized Error Translation Strategy** that converts domain exceptions to standardized HTTP problem details following RFC 7807.

### Error Translation Architecture

```python
class CoordinatorError(Exception):
    """Domain-specific exception carrying problem detail information."""

    def __init__(
        self,
        message: str,
        status_code: int,
        problem_type: str,
        detail: str | None = None,
        context: dict[str, Any] | None = None
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.problem_type = problem_type
        self.detail = detail or message
        self.context = context or {}
```

### Error Translation Flow

1. **Domain Exception**: Raised by domain services (e.g., `ProfileNotFoundError`)
2. **Error Translator**: Converts domain exception to `CoordinatorError`
3. **Protocol Handler**: Converts `CoordinatorError` to protocol-specific response
4. **Client**: Receives standardized error response with retry hints

### Problem Details Format (RFC 7807)

```json
{
  "type": "https://medical-kg.dev/problems/profile-not-found",
  "title": "Chunking profile not found",
  "status": 400,
  "detail": "Profile 'biomedical' does not exist",
  "instance": "/v1/chunk",
  "retry_after": 30,
  "context": {
    "job_id": "job-123",
    "profile_name": "biomedical"
  }
}
```

### Error Categories and HTTP Status Codes

| Error Category | HTTP Status | Problem Type | Retry Strategy |
|----------------|-------------|--------------|----------------|
| Validation Errors | 400 | validation-error | No retry |
| Authentication Errors | 401 | authentication-error | No retry |
| Authorization Errors | 403 | authorization-error | No retry |
| Resource Not Found | 404 | resource-not-found | No retry |
| Semantic Errors | 422 | semantic-error | No retry |
| Rate Limiting | 429 | rate-limit-exceeded | Retry after delay |
| Internal Errors | 500 | internal-error | No retry |
| Service Unavailable | 503 | service-unavailable | Retry with backoff |

### Error Translation Matrix

| Domain Exception | HTTP Status | Problem Type | Retry Strategy | Metric Name |
|------------------|-------------|--------------|----------------|-------------|
| ProfileNotFoundError | 400 | profile-not-found | No retry | ProfileNotFoundError |
| TokenizerMismatchError | 500 | tokenizer-mismatch | No retry | TokenizerMismatchError |
| ChunkingFailedError | 500 | chunking-failed | No retry | ChunkingFailedError |
| InvalidDocumentError | 400 | invalid-document | No retry | InvalidDocumentError |
| ChunkerConfigurationError | 422 | invalid-configuration | No retry | ChunkerConfigurationError |
| ChunkingUnavailableError | 503 | service-unavailable | Retry with backoff | ChunkingUnavailableError |
| MineruOutOfMemoryError | 503 | gpu-oom | Retry after cooldown | MineruOutOfMemoryError |
| MineruGpuUnavailableError | 503 | gpu-unavailable | Retry after cooldown | MineruGpuUnavailableError |
| MemoryError | 503 | resource-exhausted | Retry after 60s | MemoryError |
| TimeoutError | 503 | timeout | Retry after 30s | TimeoutError |

## Consequences

### Positive

- **Consistent Error Responses**: Uniform error format across all protocols
- **Better Client Experience**: Predictable error responses with retry hints
- **Improved Debugging**: Clear error messages and context information
- **Standard Compliance**: Follows RFC 7807 for HTTP problem details
- **Centralized Logic**: Single place to manage error translation
- **Better Monitoring**: Consistent error metrics and alerting
- **Retry Support**: Clients can implement appropriate retry logic

### Negative

- **Additional Complexity**: Introduces error translation layer
- **Performance Overhead**: Additional processing for error translation
- **Learning Curve**: Developers need to understand error translation patterns
- **Maintenance Burden**: Need to maintain error translation matrix

### Risks

- **Over-Translation**: Risk of losing important error context during translation
- **Performance Impact**: Error translation may impact performance for high-error scenarios
- **Inconsistent Implementation**: Different translators may implement different logic
- **Maintenance Overhead**: Error translation matrix needs to be kept up-to-date

### Mitigation

- **Comprehensive Testing**: Test error translation for all exception types
- **Performance Monitoring**: Track error translation performance
- **Documentation**: Provide clear guidelines for error translation
- **Code Review**: Ensure consistent error translation implementation

## Implementation

### Phase 1: Base Infrastructure

- Implement `CoordinatorError` base class
- Create error translation interfaces
- Define problem details format
- Implement basic error translators

### Phase 2: Domain-Specific Translators

- Implement `ChunkingErrorTranslator`
- Implement `EmbeddingErrorTranslator`
- Implement `RetrievalErrorTranslator`
- Add comprehensive error translation matrix

### Phase 3: Protocol Integration

- Update REST handlers to use error translation
- Update GraphQL resolvers to use error translation
- Update gRPC handlers to use error translation
- Add error translation to orchestration stages

## Examples

### Error Translator Implementation

```python
class ChunkingErrorTranslator:
    """Translates chunking exceptions to coordinator errors."""

    def translate(
        self,
        job_id: str,
        request: ChunkingRequest,
        exc: Exception
    ) -> CoordinatorError:
        """Translate chunking exception to coordinator error."""
        if isinstance(exc, ProfileNotFoundError):
            return CoordinatorError(
                "Chunking profile not found",
                status_code=400,
                problem_type="profile-not-found",
                detail=f"Profile '{exc.profile_name}' does not exist",
                context={
                    "job_id": job_id,
                    "profile_name": exc.profile_name,
                    "available_profiles": exc.available_profiles
                }
            )
        elif isinstance(exc, ChunkingUnavailableError):
            return CoordinatorError(
                "Chunking service unavailable",
                status_code=503,
                problem_type="service-unavailable",
                detail="Chunking service is temporarily unavailable",
                context={
                    "job_id": job_id,
                    "retry_after": 30,
                    "service_status": exc.service_status
                }
            )
        elif isinstance(exc, MineruOutOfMemoryError):
            return CoordinatorError(
                "GPU out of memory",
                status_code=503,
                problem_type="gpu-oom",
                detail="GPU memory exhausted, retry after cooldown",
                context={
                    "job_id": job_id,
                    "retry_after": 60,
                    "gpu_memory_usage": exc.memory_usage
                }
            )
        else:
            return CoordinatorError(
                "Chunking operation failed",
                status_code=500,
                problem_type="internal-error",
                detail="An unexpected error occurred during chunking",
                context={"job_id": job_id}
            )
```

### Protocol Handler Integration

```python
@app.exception_handler(CoordinatorError)
async def handle_coordinator_error(request: Request, exc: CoordinatorError):
    """Handle coordinator errors with problem details."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": f"https://medical-kg.dev/problems/{exc.problem_type}",
            "title": exc.message,
            "status": exc.status_code,
            "detail": exc.detail,
            "instance": str(request.url),
            **exc.context
        }
    )
```

### Client Error Handling

```python
async def chunk_document_with_retry(request: ChunkingRequest) -> ChunkingResult:
    """Chunk document with automatic retry on transient failures."""
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = await client.post("/v1/chunk", json=request.dict())
            return ChunkingResult(**response.json())

        except HTTPStatusError as exc:
            if exc.response.status_code == 503:
                # Service unavailable - retry with backoff
                problem_detail = exc.response.json()
                retry_after = problem_detail.get("retry_after", 30)

                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_after)
                    continue

            # Non-retryable error - re-raise
            raise exc

    raise Exception("Max retries exceeded")
```

## Validation

### Error Translation Testing

```python
def test_error_translation():
    """Test error translation for all exception types."""
    translator = ChunkingErrorTranslator()

    # Test ProfileNotFoundError
    exc = ProfileNotFoundError("biomedical", ["clinical", "research"])
    error = translator.translate("job-123", request, exc)

    assert error.status_code == 400
    assert error.problem_type == "profile-not-found"
    assert "biomedical" in error.detail
    assert error.context["profile_name"] == "biomedical"

    # Test ChunkingUnavailableError
    exc = ChunkingUnavailableError("Service down for maintenance")
    error = translator.translate("job-123", request, exc)

    assert error.status_code == 503
    assert error.problem_type == "service-unavailable"
    assert error.context["retry_after"] == 30
```

### Problem Details Validation

```python
def test_problem_details_format():
    """Test that problem details follow RFC 7807."""
    error = CoordinatorError(
        "Test error",
        status_code=400,
        problem_type="test-error",
        detail="Test error detail",
        context={"key": "value"}
    )

    problem_detail = error.to_problem_detail()

    assert "type" in problem_detail
    assert "title" in problem_detail
    assert "status" in problem_detail
    assert "detail" in problem_detail
    assert problem_detail["status"] == 400
```

## References

- [RFC 7807 Problem Details for HTTP APIs](https://tools.ietf.org/html/rfc7807)
- [HTTP Status Codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)
- [Error Handling Best Practices](https://restfulapi.net/error-handling/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Retry Patterns](https://docs.microsoft.com/en-us/azure/architecture/patterns/retry)
