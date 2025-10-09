# ADR-0001: Coordinator Architecture

## Status

Accepted

## Context

The Medical KG pipeline needed a clear separation between protocol handlers (REST/GraphQL/gRPC) and domain logic. The existing architecture had protocol-specific code mixed with business logic, making it difficult to:

- Test domain logic independently of protocol concerns
- Support multiple protocols without duplicating business logic
- Maintain consistent error handling across different protocols
- Implement protocol-agnostic metrics and monitoring

The system required a layer that could:

- Coordinate operations between protocol handlers and domain services
- Manage job lifecycle (creation, tracking, completion, failure)
- Translate domain exceptions to protocol-appropriate error responses
- Emit consistent metrics and logging across all protocols
- Provide a uniform interface for different operation types (chunking, embedding, retrieval)

## Decision

We will introduce a **Coordinator Layer** that sits between the gateway services and domain logic. This layer will:

1. **Implement the Coordinator Pattern**: Each coordinator manages a specific type of operation (chunking, embedding, retrieval)
2. **Provide Protocol Agnostic Interface**: Coordinators expose a uniform interface that can be used by any protocol handler
3. **Manage Job Lifecycle**: Track job creation, execution, completion, and failure states
4. **Handle Error Translation**: Convert domain exceptions to HTTP problem details (RFC 7807)
5. **Emit Metrics**: Provide consistent metrics emission across all operations
6. **Support Resilience**: Implement circuit breakers, rate limiting, and retry logic

### Coordinator Interface

```python
class BaseCoordinator[RequestType, ResultType]:
    """Base coordinator interface for all operation types."""

    def execute(self, request: RequestType) -> ResultType:
        """Execute operation with job lifecycle management."""
        pass

    def _execute(self, request: RequestType, **kwargs) -> ResultType:
        """Subclass implementation of operation logic."""
        pass
```

### Coordinator Responsibilities

- **Job Lifecycle Management**: Create, track, and update job states
- **Request Validation**: Validate incoming requests before processing
- **Domain Service Coordination**: Delegate to appropriate domain services
- **Error Translation**: Convert domain exceptions to coordinator errors
- **Metrics Emission**: Track operation attempts, failures, and duration
- **Resilience**: Implement circuit breakers and rate limiting

## Consequences

### Positive

- **Better Testability**: Domain logic can be tested independently of protocol concerns
- **Protocol Independence**: Business logic is decoupled from specific protocols
- **Consistent Error Handling**: Uniform error translation across all protocols
- **Centralized Metrics**: Consistent metrics emission and monitoring
- **Improved Maintainability**: Clear separation of concerns and responsibilities
- **Extensibility**: Easy to add new operation types and protocols

### Negative

- **Additional Complexity**: Introduces another layer in the architecture
- **Performance Overhead**: Additional method calls and object creation
- **Learning Curve**: Developers need to understand the coordinator pattern
- **Code Duplication**: Some common logic may be duplicated across coordinators

### Risks

- **Over-Engineering**: Risk of making the architecture too complex for simple operations
- **Performance Impact**: Additional layers may impact performance for high-throughput scenarios
- **Maintenance Burden**: More components to maintain and update

### Mitigation

- **Performance Testing**: Benchmark coordinator overhead and optimize critical paths
- **Documentation**: Provide comprehensive documentation and examples
- **Code Review**: Ensure coordinators follow established patterns and don't duplicate logic
- **Monitoring**: Track coordinator performance and error rates

## Implementation

### Phase 1: Base Infrastructure

- Implement `BaseCoordinator` abstract class
- Create `CoordinatorRequest` and `CoordinatorResult` base classes
- Implement `JobLifecycleManager` for job tracking
- Add error translation infrastructure

### Phase 2: Existing Operations

- Refactor chunking operations to use `ChunkingCoordinator`
- Refactor embedding operations to use `EmbeddingCoordinator`
- Update gateway services to use coordinators
- Add comprehensive documentation

### Phase 3: New Operations

- Implement coordinators for new operation types
- Add orchestration coordinators for pipeline execution
- Extend error translation for new domains

## Examples

### Chunking Coordinator

```python
class ChunkingCoordinator(BaseCoordinator[ChunkingRequest, ChunkingResult]):
    """Coordinates synchronous chunking operations."""

    def _execute(self, request: ChunkingRequest) -> ChunkingResult:
        # Create job entry
        job_id = self._lifecycle.create_job(request.tenant_id, "chunk")

        try:
            # Execute chunking
            chunks = self._chunker.chunk(request)

            # Assemble result
            result = ChunkingResult(
                job_id=job_id,
                chunks=chunks,
                duration_s=0.0
            )

            # Mark job completed
            self._lifecycle.mark_completed(job_id, result.metadata)

            return result

        except Exception as exc:
            # Translate and record error
            error = self._translate_error(job_id, request, exc)
            self._lifecycle.mark_failed(job_id, str(exc))
            raise error
```

### Error Translation

```python
class ChunkingErrorTranslator:
    """Translates chunking exceptions to coordinator errors."""

    def translate(self, job_id: str, request: ChunkingRequest, exc: Exception) -> CoordinatorError:
        if isinstance(exc, ProfileNotFoundError):
            return CoordinatorError(
                "Chunking profile not found",
                status_code=400,
                problem_type="profile-not-found",
                detail=f"Profile '{exc.profile_name}' does not exist"
            )
        elif isinstance(exc, ChunkingUnavailableError):
            return CoordinatorError(
                "Chunking service unavailable",
                status_code=503,
                problem_type="service-unavailable",
                detail="Chunking service is temporarily unavailable",
                context={"retry_after": 30}
            )
        # ... other exception types
```

## References

- [Coordinator Pattern](https://martinfowler.com/eaaCatalog/coordinator.html)
- [RFC 7807 Problem Details for HTTP APIs](https://tools.ietf.org/html/rfc7807)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Job Lifecycle Management](https://docs.aws.amazon.com/batch/latest/userguide/job_lifecycle.html)
