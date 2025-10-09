"""Protocol/ABC docstring template for pipeline interfaces.

This template shows the required structure and content for protocol and abstract
base class docstrings in the Medical_KG_rev pipeline codebase.
"""

# Example protocol docstring structure:

"""[One-line summary of protocol purpose].

[Detailed explanation of what the protocol defines, when to implement it,
and how it fits into the architecture.]

Interface Contract:
    [Describe the contract that implementers must follow]
    [List required methods and their signatures]
    [Describe any invariants that must be maintained]

When to Implement:
    [Describe when you should implement this protocol]
    [Give examples of use cases]
    [Mention any alternatives]

Existing Implementations:
    - [List existing implementations with brief descriptions]
    - [Mention any planned implementations]

Example:
    >>> class MyImplementation(MyProtocol):
    ...     def required_method(self, param: str) -> int:
    ...         return len(param)
    >>> impl = MyImplementation()
    >>> result = impl.required_method("hello")
    >>> print(result)
    5
"""

# Real example for BaseCoordinator:

"""Abstract base class for all coordinator implementations.

BaseCoordinator defines the interface that all coordinators must implement
to provide consistent behavior across different operation types (chunking,
embedding, etc.). It implements the Template Method pattern with _execute
as the main template method.

This protocol ensures that all coordinators follow the same pattern:
1. Validate request and extract parameters
2. Create job in lifecycle manager
3. Delegate to appropriate service
4. Handle exceptions and translate to coordinator errors
5. Assemble results and mark job as completed
6. Return standardized result format

Interface Contract:
    Implementers must:
    - Inherit from BaseCoordinator[RequestType, ResultType]
    - Implement _execute method with the main coordination logic
    - Handle all exceptions and translate to CoordinatorError
    - Use JobLifecycleManager for job tracking
    - Emit appropriate metrics for operations
    - Return ResultType with job_id, duration_s, and operation results

When to Implement:
    Implement this protocol when creating new coordinators for:
    - New operation types (validation, extraction, etc.)
    - Different service backends (new chunking libraries, etc.)
    - Custom error handling or retry logic
    - Integration with new external services

Existing Implementations:
    - ChunkingCoordinator: Coordinates chunking operations via ChunkingService
    - EmbeddingCoordinator: Coordinates embedding operations via embedding services
    - ValidationCoordinator: Coordinates validation operations (planned)

Example:
    >>> class MyCoordinator(BaseCoordinator[MyRequest, MyResult]):
    ...     def _execute(self, request: MyRequest, **kwargs) -> MyResult:
    ...         job_id = self._lifecycle.create_job(request.tenant_id, "my_op")
    ...         try:
    ...             result = self._service.process(request)
    ...             self._lifecycle.mark_completed(job_id)
    ...             return MyResult(job_id=job_id, result=result)
    ...         except Exception as exc:
    ...             self._lifecycle.mark_failed(job_id, str(exc))
    ...             raise CoordinatorError(str(exc))
    >>> coordinator = MyCoordinator(...)
    >>> result = coordinator.execute(MyRequest(...))
"""

# Real example for EmbeddingPersister:

"""Protocol for embedding persistence operations.

EmbeddingPersister defines the interface for persisting embedding vectors
to various storage backends. It provides a consistent interface for
embedding storage regardless of the underlying storage technology.

This protocol enables pluggable persistence backends while maintaining
consistent behavior across different storage solutions (vector databases,
file systems, etc.).

Interface Contract:
    Implementers must:
    - Implement persist method for storing embedding vectors
    - Handle persistence context with tenant isolation
    - Support batch operations for efficiency
    - Provide error handling for storage failures
    - Maintain consistency guarantees appropriate to storage backend
    - Support metadata storage alongside vectors

When to Implement:
    Implement this protocol when creating new persistence backends:
    - New vector databases (Pinecone, Weaviate, etc.)
    - File-based storage (HDF5, Parquet, etc.)
    - Cloud storage solutions (S3, GCS, etc.)
    - Custom storage backends with specific requirements

Existing Implementations:
    - VectorDBPersister: Persists to vector database backends
    - FilePersister: Persists to local file system
    - CloudPersister: Persists to cloud storage (planned)

Example:
    >>> class MyPersister(EmbeddingPersister):
    ...     async def persist(
    ...         self,
    ...         context: PersistenceContext,
    ...         vectors: list[EmbeddingVector],
    ...         metadata: dict[str, Any]
    ...     ) -> None:
    ...         # Store vectors in my custom backend
    ...         await self._backend.store(
    ...             tenant_id=context.tenant_id,
    ...             namespace=context.namespace,
    ...             vectors=vectors,
    ...             metadata=metadata
    ...         )
    >>> persister = MyPersister(...)
    >>> await persister.persist(context, vectors, metadata)
"""
