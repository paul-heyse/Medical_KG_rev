# Pipeline Extension Guide

This guide provides comprehensive instructions for extending the Medical KG pipeline architecture. It covers adding new coordinators, orchestration stages, embedding policies, persisters, and error handling mechanisms.

## Overview

The Medical KG pipeline follows a layered architecture with clear separation of concerns:

- **Gateway Layer**: Protocol-agnostic service layer between protocol handlers (REST/GraphQL/gRPC) and domain logic
- **Coordinator Layer**: Manages job lifecycle, error translation, and metrics emission for specific operations
- **Service Layer**: Domain-specific implementations for chunking, embedding, and retrieval operations
- **Orchestration Layer**: Stage-based pipeline execution with Dagster integration

### Data Flow

```
Request → Validation → Execution → Response
    ↓
Gateway Service → Coordinator → Domain Service → External Library
    ↓
Job Lifecycle Management + Error Translation + Metrics Emission
```

### Error Handling Strategy

The pipeline implements a comprehensive error handling strategy:

- **Exception Translation**: Domain exceptions are translated to HTTP problem details
- **Problem Details**: RFC 7807 compliant error responses with retry hints
- **Metrics**: All errors are tracked with appropriate metric names
- **Circuit Breakers**: Prevent cascading failures in distributed systems
- **Rate Limiting**: Control request rate to prevent resource exhaustion

## Adding a New Coordinator

Coordinators provide a protocol-agnostic interface between gateway services and domain logic. Each coordinator manages a specific type of operation (chunking, embedding, retrieval, etc.).

### Step 1: Define Request/Response Dataclasses

Create dataclasses that inherit from the base coordinator interfaces:

```python
from dataclasses import dataclass, field
from typing import Any, Sequence
from Medical_KG_rev.gateway.coordinators.base import CoordinatorRequest, CoordinatorResult

@dataclass
class MyOperationRequest(CoordinatorRequest):
    """Request for my custom operation.

    Attributes:
        operation_id: Unique identifier for the operation.
        parameters: Operation-specific parameters.
        options: Additional configuration options.
    """
    operation_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

@dataclass
class MyOperationResult(CoordinatorResult):
    """Result of my custom operation.

    Attributes:
        output: The operation output data.
        metadata: Additional operation metadata.
    """
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Step 2: Implement Coordinator Class

Create a coordinator class that inherits from `BaseCoordinator`:

```python
from typing import Any
from Medical_KG_rev.gateway.coordinators.base import BaseCoordinator, CoordinatorConfig
from Medical_KG_rev.gateway.coordinators.job_lifecycle import JobLifecycleManager

class MyOperationCoordinator(BaseCoordinator[MyOperationRequest, MyOperationResult]):
    """Coordinates my custom operations with job lifecycle management.

    This coordinator implements the coordinator pattern for my custom operations.
    It coordinates between gateway services and domain logic to provide
    protocol-agnostic operation capabilities.

    Attributes:
        _lifecycle: JobLifecycleManager for tracking job states
        _service: MyOperationService for actual operation execution
        _errors: MyOperationErrorTranslator for error translation

    Invariants:
        - self._lifecycle is never None after __init__
        - self._service is never None after __init__
        - self._errors is never None after __init__

    Thread Safety:
        - Not thread-safe: Must be called from single thread

    Lifecycle:
        - Created with dependencies injected
        - Used for coordinating operations
        - No explicit cleanup required

    Example:
        >>> coordinator = MyOperationCoordinator(
        ...     lifecycle=JobLifecycleManager(),
        ...     service=MyOperationService(),
        ...     config=CoordinatorConfig(name="my_operation")
        ... )
        >>> result = coordinator.execute(MyOperationRequest(...))
        >>> print(f"Operation completed: {result.output}")
    """

    def __init__(
        self,
        lifecycle: JobLifecycleManager,
        service: MyOperationService,
        config: CoordinatorConfig,
        errors: MyOperationErrorTranslator | None = None,
    ) -> None:
        """Initialize my operation coordinator.

        Args:
            lifecycle: JobLifecycleManager for tracking job states.
            service: MyOperationService for actual operation execution.
            config: CoordinatorConfig with coordinator settings.
            errors: Optional error translator, auto-created if not provided.
        """
        super().__init__(config)
        self._lifecycle = lifecycle
        self._service = service
        self._errors = errors or MyOperationErrorTranslator()

    def _execute(self, request: MyOperationRequest, **kwargs: Any) -> MyOperationResult:
        """Execute my custom operation with job lifecycle management.

        Coordinates the full operation workflow: creates job entry, validates
        parameters, delegates to operation service, handles exceptions,
        assembles results, and marks job completed.

        Args:
            request: MyOperationRequest with operation parameters.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            MyOperationResult with operation output and metadata.

        Raises:
            CoordinatorError: For all handled errors after translation.
        """
        # Create job entry
        job_id = self._lifecycle.create_job(
            tenant_id=request.tenant_id,
            operation="my_operation",
            metadata={"operation_id": request.operation_id}
        )

        try:
            # Execute operation
            output = self._service.execute(request.parameters)

            # Assemble result
            result = MyOperationResult(
                job_id=job_id,
                duration_s=0.0,  # Calculate actual duration
                output=output,
                metadata={"status": "completed"}
            )

            # Mark job completed
            self._lifecycle.mark_completed(job_id, result.metadata)

            return result

        except Exception as exc:
            # Translate and record error
            error = self._errors.translate(job_id, request, exc)
            self._lifecycle.mark_failed(job_id, str(exc))
            raise error
```

### Step 3: Add Error Translation Logic

Create an error translator for your coordinator:

```python
from typing import Any
from Medical_KG_rev.gateway.coordinators.base import CoordinatorError
from Medical_KG_rev.gateway.chunking_errors import ChunkingErrorReport

class MyOperationErrorTranslator:
    """Translates my operation exceptions to coordinator errors.

    This class provides centralized exception-to-HTTP error mapping
    for my operation domain. It maps each operation exception type
    to appropriate status codes, problem types, and retry strategies.
    """

    def translate(
        self,
        job_id: str,
        request: MyOperationRequest,
        exc: Exception
    ) -> CoordinatorError:
        """Translate operation exception to coordinator error.

        Args:
            job_id: Job identifier for error context.
            request: Original request that failed.
            exc: Exception to translate.

        Returns:
            CoordinatorError with problem detail and context.
        """
        if isinstance(exc, MyOperationValidationError):
            return CoordinatorError(
                "Operation validation failed",
                status_code=400,
                problem_type="validation-error",
                detail=f"Invalid parameters: {exc.message}",
                context={"job_id": job_id, "operation_id": request.operation_id}
            )
        elif isinstance(exc, MyOperationServiceError):
            return CoordinatorError(
                "Operation service unavailable",
                status_code=503,
                problem_type="service-unavailable",
                detail="Operation service is temporarily unavailable",
                context={"job_id": job_id, "retry_after": 30}
            )
        else:
            return CoordinatorError(
                "Operation failed unexpectedly",
                status_code=500,
                problem_type="internal-error",
                detail="An unexpected error occurred",
                context={"job_id": job_id}
            )
```

### Step 4: Write Comprehensive Docstrings

Follow the Google-style docstring format with all required sections:

- **Module docstring**: Explain the coordinator's role and responsibilities
- **Class docstring**: Document purpose, attributes, invariants, thread safety, lifecycle
- **Method docstrings**: Include Args, Returns, Raises, Example sections
- **Dataclass docstrings**: Document each field with valid ranges and constraints

### Step 5: Add Unit Tests

Create comprehensive unit tests for your coordinator:

```python
import pytest
from unittest.mock import Mock, patch
from Medical_KG_rev.gateway.coordinators.my_operation import MyOperationCoordinator, MyOperationRequest

class TestMyOperationCoordinator:
    """Test suite for MyOperationCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator instance for testing."""
        lifecycle = Mock()
        service = Mock()
        config = CoordinatorConfig(name="test")
        return MyOperationCoordinator(lifecycle, service, config)

    def test_execute_success(self, coordinator):
        """Test that coordinator executes successfully with valid request."""
        # Arrange
        request = MyOperationRequest(
            tenant_id="test",
            operation_id="op1",
            parameters={"param1": "value1"}
        )
        coordinator._service.execute.return_value = {"result": "success"}

        # Act
        result = coordinator._execute(request)

        # Assert
        assert result.output == {"result": "success"}
        assert result.job_id is not None
        coordinator._lifecycle.create_job.assert_called_once()
        coordinator._lifecycle.mark_completed.assert_called_once()

    def test_execute_validation_error(self, coordinator):
        """Test that coordinator raises CoordinatorError for validation failures."""
        # Arrange
        request = MyOperationRequest(
            tenant_id="test",
            operation_id="op1",
            parameters={}
        )
        coordinator._service.execute.side_effect = MyOperationValidationError("Invalid parameters")

        # Act & Assert
        with pytest.raises(CoordinatorError) as exc_info:
            coordinator._execute(request)

        assert exc_info.value.status_code == 400
        assert exc_info.value.problem_type == "validation-error"
```

## Adding a New Orchestration Stage

Orchestration stages provide modular pipeline execution with Dagster integration. Each stage performs a specific transformation or operation.

### Step 1: Implement Stage Function

Create a stage function that accepts `StageContext` and returns `StageResult`:

```python
from typing import Any, Sequence
from Medical_KG_rev.orchestration.stages.contracts import StageContext, StageResult

def my_custom_stage(context: StageContext) -> StageResult:
    """Perform my custom transformation on the pipeline data.

    This stage implements a specific transformation or operation
    as part of the larger pipeline. It processes input data from
    the context and produces output data for downstream stages.

    Args:
        context: StageContext containing input data and configuration.
            - context.data: Input data from previous stages
            - context.config: Stage-specific configuration
            - context.metadata: Additional stage metadata

    Returns:
        StageResult containing transformed data and metadata.
            - result.data: Transformed output data
            - result.metadata: Stage execution metadata
            - result.metrics: Performance and quality metrics

    Raises:
        StageExecutionError: If stage execution fails.
        ValidationError: If input data is invalid.

    Side Effects:
        - Logs stage execution progress
        - Emits performance metrics
        - Updates pipeline state

    Example:
        >>> context = StageContext(
        ...     data={"input": "sample data"},
        ...     config={"param": "value"},
        ...     metadata={"stage": "my_custom"}
        ... )
        >>> result = my_custom_stage(context)
        >>> assert "transformed" in result.data
    """
    try:
        # Extract input data
        input_data = context.data.get("input", "")

        # Perform transformation
        transformed_data = transform_input(input_data, context.config)

        # Assemble result
        result = StageResult(
            data={"output": transformed_data},
            metadata={
                "stage": "my_custom",
                "input_size": len(input_data),
                "output_size": len(transformed_data)
            },
            metrics={
                "processing_time": 0.1,
                "memory_usage": 1024
            }
        )

        return result

    except Exception as exc:
        raise StageExecutionError(f"My custom stage failed: {exc}") from exc

def transform_input(data: str, config: dict[str, Any]) -> str:
    """Transform input data according to configuration.

    Args:
        data: Input data to transform.
        config: Transformation configuration.

    Returns:
        Transformed data.
    """
    # Implementation here
    return f"transformed_{data}"
```

### Step 2: Register Stage Using Decorator

Use the `@stage_plugin` decorator to register your stage:

```python
from Medical_KG_rev.orchestration.stages.plugins import stage_plugin

@stage_plugin("my_custom_stage")
def my_custom_stage(context: StageContext) -> StageResult:
    """Perform my custom transformation on the pipeline data."""
    # Implementation here
    pass
```

### Step 3: Add Stage to Pipeline Configuration

Update the pipeline configuration YAML to include your stage:

```yaml
# config/orchestration/pipelines/my_pipeline.yaml
stages:
  - name: "my_custom_stage"
    type: "transformation"
    config:
      param1: "value1"
      param2: "value2"
    dependencies:
      - "previous_stage"
    retry_policy:
      max_attempts: 3
      backoff_factor: 2.0
```

### Step 4: Write Stage Documentation

Document your stage with comprehensive docstrings:

- **Function docstring**: Explain purpose, inputs, outputs, side effects
- **Parameter documentation**: Describe each parameter and valid values
- **Return value documentation**: Explain result structure and meaning
- **Exception documentation**: List all possible exceptions and conditions
- **Example usage**: Show typical usage patterns

### Step 5: Add Unit Tests

Create unit tests for your stage:

```python
import pytest
from Medical_KG_rev.orchestration.stages.contracts import StageContext, StageResult
from Medical_KG_rev.orchestration.stages.my_custom import my_custom_stage

class TestMyCustomStage:
    """Test suite for my_custom_stage."""

    def test_stage_execution_success(self):
        """Test that stage executes successfully with valid input."""
        # Arrange
        context = StageContext(
            data={"input": "test data"},
            config={"param": "value"},
            metadata={"stage": "my_custom"}
        )

        # Act
        result = my_custom_stage(context)

        # Assert
        assert isinstance(result, StageResult)
        assert "output" in result.data
        assert result.metadata["stage"] == "my_custom"
        assert "processing_time" in result.metrics

    def test_stage_execution_invalid_input(self):
        """Test that stage raises ValidationError for invalid input."""
        # Arrange
        context = StageContext(
            data={},  # Missing required input
            config={"param": "value"},
            metadata={"stage": "my_custom"}
        )

        # Act & Assert
        with pytest.raises(ValidationError):
            my_custom_stage(context)
```

## Adding a New Embedding Policy

Embedding policies enforce access control and configuration for embedding operations. They implement the `NamespaceAccessPolicy` interface.

### Step 1: Create Policy Class

Implement a class that inherits from `NamespaceAccessPolicy`:

```python
from abc import ABC, abstractmethod
from typing import Any, Mapping
from Medical_KG_rev.services.embedding.policy import NamespaceAccessPolicy, NamespaceAccessDecision

class MyCustomPolicy(NamespaceAccessPolicy):
    """Custom namespace access policy implementation.

    This policy implements custom access control logic for embedding
    operations. It evaluates tenant access requests against custom
    business rules and configuration.

    Attributes:
        _config: Policy configuration settings.
        _cache: Access decision cache for performance.

    Invariants:
        - self._config is never None after __init__
        - self._cache is never None after __init__

    Thread Safety:
        - Thread-safe: All methods are thread-safe

    Lifecycle:
        - Created with configuration
        - Used for policy evaluation
        - No explicit cleanup required

    Example:
        >>> policy = MyCustomPolicy(config={"rule": "value"})
        >>> decision = policy.evaluate("tenant1", "namespace1")
        >>> assert decision.allowed in [True, False]
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize custom policy with configuration.

        Args:
            config: Policy configuration settings.
        """
        self._config = config
        self._cache = {}  # Simple cache implementation

    def evaluate(
        self,
        tenant_id: str,
        namespace: str,
        context: Mapping[str, Any] | None = None
    ) -> NamespaceAccessDecision:
        """Evaluate access request for tenant and namespace.

        Args:
            tenant_id: Tenant identifier requesting access.
            namespace: Namespace identifier being accessed.
            context: Additional context for evaluation.

        Returns:
            NamespaceAccessDecision with access result and metadata.
        """
        # Check cache first
        cache_key = f"{tenant_id}:{namespace}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Evaluate access
        decision = self._evaluate_access(tenant_id, namespace, context)

        # Cache result
        self._cache[cache_key] = decision

        return decision

    def _evaluate_access(
        self,
        tenant_id: str,
        namespace: str,
        context: Mapping[str, Any] | None
    ) -> NamespaceAccessDecision:
        """Internal access evaluation logic.

        Args:
            tenant_id: Tenant identifier.
            namespace: Namespace identifier.
            context: Evaluation context.

        Returns:
            NamespaceAccessDecision with evaluation result.
        """
        # Implement custom business logic here
        if self._is_tenant_allowed(tenant_id, namespace):
            return NamespaceAccessDecision(
                allowed=True,
                reason="Tenant has access to namespace",
                metadata={"policy": "my_custom", "evaluated_at": "now"}
            )
        else:
            return NamespaceAccessDecision(
                allowed=False,
                reason="Tenant does not have access to namespace",
                metadata={"policy": "my_custom", "evaluated_at": "now"}
            )

    def _is_tenant_allowed(self, tenant_id: str, namespace: str) -> bool:
        """Check if tenant is allowed access to namespace.

        Args:
            tenant_id: Tenant identifier.
            namespace: Namespace identifier.

        Returns:
            True if access is allowed, False otherwise.
        """
        # Implement custom logic here
        return tenant_id in self._config.get("allowed_tenants", [])

    def invalidate(self, namespace: str | None = None) -> None:
        """Invalidate policy cache.

        Args:
            namespace: Optional namespace to invalidate (all if None).
        """
        if namespace is None:
            self._cache.clear()
        else:
            # Remove entries for specific namespace
            keys_to_remove = [k for k in self._cache.keys() if k.endswith(f":{namespace}")]
            for key in keys_to_remove:
                del self._cache[key]
```

### Step 2: Register Policy in Policy Chain

Add your policy to the policy chain via `build_policy_chain`:

```python
from Medical_KG_rev.services.embedding.policy import build_policy_chain

def build_custom_policy_chain(config: dict[str, Any]) -> NamespaceAccessPolicy:
    """Build policy chain with custom policy.

    Args:
        config: Policy configuration.

    Returns:
        Combined policy chain.
    """
    policies = [
        MyCustomPolicy(config["my_custom"]),
        # Add other policies here
    ]

    return build_policy_chain(policies)
```

### Step 3: Write Policy Documentation

Document your policy with comprehensive docstrings:

- **Class docstring**: Explain policy purpose, attributes, invariants, thread safety
- **Method docstrings**: Document evaluation logic, parameters, return values
- **Configuration documentation**: Explain required configuration parameters
- **Example usage**: Show typical policy evaluation scenarios

### Step 4: Add Unit Tests

Create unit tests for your policy:

```python
import pytest
from Medical_KG_rev.services.embedding.policy import NamespaceAccessDecision
from Medical_KG_rev.services.embedding.my_custom import MyCustomPolicy

class TestMyCustomPolicy:
    """Test suite for MyCustomPolicy."""

    @pytest.fixture
    def policy(self):
        """Create policy instance for testing."""
        config = {
            "allowed_tenants": ["tenant1", "tenant2"],
            "rule": "value"
        }
        return MyCustomPolicy(config)

    def test_evaluate_allowed_tenant(self, policy):
        """Test that allowed tenant gets access."""
        # Act
        decision = policy.evaluate("tenant1", "namespace1")

        # Assert
        assert decision.allowed is True
        assert "Tenant has access" in decision.reason
        assert decision.metadata["policy"] == "my_custom"

    def test_evaluate_denied_tenant(self, policy):
        """Test that denied tenant gets no access."""
        # Act
        decision = policy.evaluate("tenant3", "namespace1")

        # Assert
        assert decision.allowed is False
        assert "does not have access" in decision.reason

    def test_cache_invalidation(self, policy):
        """Test that cache invalidation works correctly."""
        # Arrange
        policy.evaluate("tenant1", "namespace1")  # Populate cache

        # Act
        policy.invalidate("namespace1")

        # Assert
        assert len(policy._cache) == 0
```

## Adding a New Persister

Persisters handle embedding storage and retrieval operations. They implement the `EmbeddingPersister` protocol.

### Step 1: Implement EmbeddingPersister Protocol

Create a class that implements the `EmbeddingPersister` protocol:

```python
from typing import Any, Sequence
from Medical_KG_rev.services.embedding.persister import EmbeddingPersister, PersistenceContext

class MyCustomPersister(EmbeddingPersister):
    """Custom embedding persister implementation.

    This persister implements custom storage and retrieval logic
    for embedding vectors. It provides persistence operations
    with specific consistency guarantees and performance characteristics.

    Attributes:
        _storage_backend: Underlying storage backend.
        _config: Persister configuration settings.

    Invariants:
        - self._storage_backend is never None after __init__
        - self._config is never None after __init__

    Thread Safety:
        - Thread-safe: All methods are thread-safe

    Lifecycle:
        - Created with storage backend and configuration
        - Used for persistence operations
        - No explicit cleanup required

    Example:
        >>> persister = MyCustomPersister(backend, config)
        >>> context = PersistenceContext(namespace="ns1", tenant_id="t1")
        >>> persister.persist(context, embeddings)
        >>> retrieved = persister.retrieve(context, ids)
    """

    def __init__(self, storage_backend: Any, config: dict[str, Any]) -> None:
        """Initialize custom persister.

        Args:
            storage_backend: Underlying storage backend.
            config: Persister configuration settings.
        """
        self._storage_backend = storage_backend
        self._config = config

    def persist(
        self,
        context: PersistenceContext,
        embeddings: Sequence[dict[str, Any]]
    ) -> None:
        """Persist embedding vectors to storage.

        Args:
            context: Persistence context with namespace and tenant info.
            embeddings: Sequence of embedding vectors to persist.

        Raises:
            PersistenceError: If persistence operation fails.
            ValidationError: If embedding data is invalid.
        """
        try:
            # Validate embeddings
            self._validate_embeddings(embeddings)

            # Persist to storage backend
            self._storage_backend.store(
                namespace=context.namespace,
                tenant_id=context.tenant_id,
                embeddings=embeddings
            )

        except Exception as exc:
            raise PersistenceError(f"Failed to persist embeddings: {exc}") from exc

    def retrieve(
        self,
        context: PersistenceContext,
        ids: Sequence[str]
    ) -> Sequence[dict[str, Any]]:
        """Retrieve embedding vectors from storage.

        Args:
            context: Persistence context with namespace and tenant info.
            ids: Sequence of embedding IDs to retrieve.

        Returns:
            Sequence of retrieved embedding vectors.

        Raises:
            PersistenceError: If retrieval operation fails.
            NotFoundError: If requested embeddings are not found.
        """
        try:
            # Retrieve from storage backend
            embeddings = self._storage_backend.retrieve(
                namespace=context.namespace,
                tenant_id=context.tenant_id,
                ids=ids
            )

            return embeddings

        except Exception as exc:
            raise PersistenceError(f"Failed to retrieve embeddings: {exc}") from exc

    def _validate_embeddings(self, embeddings: Sequence[dict[str, Any]]) -> None:
        """Validate embedding data before persistence.

        Args:
            embeddings: Embedding vectors to validate.

        Raises:
            ValidationError: If embedding data is invalid.
        """
        for i, embedding in enumerate(embeddings):
            if "id" not in embedding:
                raise ValidationError(f"Embedding {i} missing required 'id' field")
            if "vector" not in embedding:
                raise ValidationError(f"Embedding {i} missing required 'vector' field")
            if not isinstance(embedding["vector"], (list, tuple)):
                raise ValidationError(f"Embedding {i} vector must be list or tuple")
```

### Step 2: Register Persister in Factory

Add your persister to the `build_persister` factory:

```python
from Medical_KG_rev.services.embedding.persister import build_persister

def build_custom_persister(config: dict[str, Any]) -> EmbeddingPersister:
    """Build persister with custom implementation.

    Args:
        config: Persister configuration.

    Returns:
        Configured persister instance.
    """
    storage_backend = create_storage_backend(config["storage"])
    return MyCustomPersister(storage_backend, config["my_custom"])
```

### Step 3: Write Persister Documentation

Document your persister with comprehensive docstrings:

- **Class docstring**: Explain storage semantics, consistency guarantees, performance characteristics
- **Method docstrings**: Document persistence operations, parameters, return values, exceptions
- **Configuration documentation**: Explain required configuration parameters
- **Example usage**: Show typical persistence and retrieval scenarios

### Step 4: Add Unit and Integration Tests

Create comprehensive tests for your persister:

```python
import pytest
from unittest.mock import Mock
from Medical_KG_rev.services.embedding.persister import PersistenceContext
from Medical_KG_rev.services.embedding.my_custom import MyCustomPersister

class TestMyCustomPersister:
    """Test suite for MyCustomPersister."""

    @pytest.fixture
    def persister(self):
        """Create persister instance for testing."""
        storage_backend = Mock()
        config = {"setting": "value"}
        return MyCustomPersister(storage_backend, config)

    def test_persist_success(self, persister):
        """Test successful embedding persistence."""
        # Arrange
        context = PersistenceContext(namespace="ns1", tenant_id="t1")
        embeddings = [
            {"id": "emb1", "vector": [0.1, 0.2, 0.3]},
            {"id": "emb2", "vector": [0.4, 0.5, 0.6]}
        ]

        # Act
        persister.persist(context, embeddings)

        # Assert
        persister._storage_backend.store.assert_called_once_with(
            namespace="ns1",
            tenant_id="t1",
            embeddings=embeddings
        )

    def test_persist_validation_error(self, persister):
        """Test that invalid embeddings raise ValidationError."""
        # Arrange
        context = PersistenceContext(namespace="ns1", tenant_id="t1")
        embeddings = [{"id": "emb1"}]  # Missing vector field

        # Act & Assert
        with pytest.raises(ValidationError):
            persister.persist(context, embeddings)

    def test_retrieve_success(self, persister):
        """Test successful embedding retrieval."""
        # Arrange
        context = PersistenceContext(namespace="ns1", tenant_id="t1")
        ids = ["emb1", "emb2"]
        expected_embeddings = [
            {"id": "emb1", "vector": [0.1, 0.2, 0.3]},
            {"id": "emb2", "vector": [0.4, 0.5, 0.6]}
        ]
        persister._storage_backend.retrieve.return_value = expected_embeddings

        # Act
        result = persister.retrieve(context, ids)

        # Assert
        assert result == expected_embeddings
        persister._storage_backend.retrieve.assert_called_once_with(
            namespace="ns1",
            tenant_id="t1",
            ids=ids
        )
```

## Error Handling

The pipeline implements comprehensive error handling with exception translation, problem details, and retry strategies.

### Adding New Exception Types

To add new exception types to the error translator:

1. **Define Exception Class**:

```python
class MyOperationError(Exception):
    """Base exception for my operation domain.

    This exception serves as the base class for all
    my operation-related errors.
    """
    pass

class MyOperationValidationError(MyOperationError):
    """Raised when operation validation fails.

    This exception is raised when operation parameters
    fail validation checks.
    """
    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field

class MyOperationServiceError(MyOperationError):
    """Raised when operation service is unavailable.

    This exception is raised when the underlying
    operation service cannot be reached or is
    experiencing issues.
    """
    def __init__(self, message: str, retry_after: int = 30) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after
```

2. **Add to Error Translator**:

```python
def translate(
    self,
    job_id: str,
    request: MyOperationRequest,
    exc: Exception
) -> CoordinatorError:
    """Translate operation exception to coordinator error."""
    if isinstance(exc, MyOperationValidationError):
        return CoordinatorError(
            "Operation validation failed",
            status_code=400,
            problem_type="validation-error",
            detail=f"Invalid {exc.field}: {exc.message}",
            context={"job_id": job_id, "field": exc.field}
        )
    elif isinstance(exc, MyOperationServiceError):
        return CoordinatorError(
            "Operation service unavailable",
            status_code=503,
            problem_type="service-unavailable",
            detail=exc.message,
            context={"job_id": job_id, "retry_after": exc.retry_after}
        )
    # ... other exception types
```

### Mapping Exceptions to HTTP Problem Details

Follow RFC 7807 for problem details:

```python
# HTTP Status Codes
400 - Bad Request (validation errors)
401 - Unauthorized (authentication errors)
403 - Forbidden (authorization errors)
404 - Not Found (resource not found)
422 - Unprocessable Entity (semantic errors)
429 - Too Many Requests (rate limiting)
500 - Internal Server Error (unexpected errors)
503 - Service Unavailable (temporary failures)

# Problem Types
validation-error - Input validation failed
authentication-error - Authentication failed
authorization-error - Authorization failed
resource-not-found - Requested resource not found
semantic-error - Request semantics invalid
rate-limit-exceeded - Rate limit exceeded
internal-error - Internal server error
service-unavailable - Service temporarily unavailable
```

### Defining Retry Strategies

Implement retry strategies for different error types:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def operation_with_retry(self, request: MyOperationRequest) -> MyOperationResult:
    """Operation with automatic retry on failure.

    This method automatically retries on transient failures
    with exponential backoff.
    """
    try:
        return self._execute_operation(request)
    except MyOperationServiceError:
        # This will trigger retry
        raise
    except MyOperationValidationError:
        # This will not trigger retry
        raise
```

## Testing

The pipeline provides comprehensive testing patterns for all components.

### Fixture Patterns for Coordinators

```python
import pytest
from unittest.mock import Mock
from Medical_KG_rev.gateway.coordinators.job_lifecycle import JobLifecycleManager

@pytest.fixture
def mock_lifecycle():
    """Create mock job lifecycle manager."""
    lifecycle = Mock(spec=JobLifecycleManager)
    lifecycle.create_job.return_value = "job-123"
    lifecycle.mark_completed.return_value = None
    lifecycle.mark_failed.return_value = None
    return lifecycle

@pytest.fixture
def mock_service():
    """Create mock operation service."""
    service = Mock()
    service.execute.return_value = {"result": "success"}
    return service

@pytest.fixture
def coordinator(mock_lifecycle, mock_service):
    """Create coordinator instance for testing."""
    config = CoordinatorConfig(name="test")
    return MyOperationCoordinator(mock_lifecycle, mock_service, config)
```

### Fixture Patterns for Services

```python
@pytest.fixture
def mock_storage_backend():
    """Create mock storage backend."""
    backend = Mock()
    backend.store.return_value = None
    backend.retrieve.return_value = []
    return backend

@pytest.fixture
def persister(mock_storage_backend):
    """Create persister instance for testing."""
    config = {"setting": "value"}
    return MyCustomPersister(mock_storage_backend, config)
```

### Fixture Patterns for Stages

```python
@pytest.fixture
def stage_context():
    """Create stage context for testing."""
    return StageContext(
        data={"input": "test data"},
        config={"param": "value"},
        metadata={"stage": "test"}
    )
```

### Assertion Patterns

```python
def test_coordinator_execution(coordinator):
    """Test coordinator execution with assertions."""
    # Arrange
    request = MyOperationRequest(tenant_id="test", operation_id="op1")

    # Act
    result = coordinator._execute(request)

    # Assert
    assert isinstance(result, MyOperationResult)
    assert result.job_id == "job-123"
    assert result.output == {"result": "success"}

    # Verify interactions
    coordinator._lifecycle.create_job.assert_called_once_with(
        tenant_id="test",
        operation="my_operation",
        metadata={"operation_id": "op1"}
    )
    coordinator._lifecycle.mark_completed.assert_called_once()
```

## Documentation

Follow the established documentation standards for all new components.

### Docstring Standards

- **Google-style docstrings** for all modules, classes, and functions
- **Required sections**: Args, Returns, Raises, Example
- **Optional sections**: Note, Warning, Attributes, Invariants, Thread Safety, Lifecycle
- **Cross-references** using Sphinx-style syntax (`:class:`, `:func:`, `:meth:`)

### Section Headers

Organize code with consistent section headers:

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# DATA MODELS
# ============================================================================

# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================

# ============================================================================
# ERROR TRANSLATION
# ============================================================================

# ============================================================================
# EXPORTS
# ============================================================================
```

### Running Documentation Checks

Use the automated documentation quality checks:

```bash
# Check docstrings
ruff check --select D src/Medical_KG_rev/

# Check section headers
python scripts/check_section_headers.py

# Check docstring coverage
python scripts/check_docstring_coverage.py --min-coverage 90

# Build documentation
mkdocs build --strict
```

## Best Practices

1. **Follow Established Patterns**: Use existing coordinators, stages, and policies as templates
2. **Comprehensive Documentation**: Document all public interfaces with Google-style docstrings
3. **Error Handling**: Implement proper exception translation and problem details
4. **Testing**: Write comprehensive unit and integration tests
5. **Performance**: Consider performance implications and implement appropriate caching
6. **Security**: Follow security best practices for access control and data handling
7. **Monitoring**: Add appropriate metrics and logging for observability
8. **Configuration**: Make components configurable through YAML configuration files

## Getting Help

- **Existing Code**: Study existing coordinators, stages, and policies for patterns
- **Documentation**: Refer to the comprehensive docstrings in refactored modules
- **Templates**: Use the templates in `openspec/changes/add-pipeline-structure-documentation/templates/`
- **Standards**: Follow the documentation standards in `docs/contributing/documentation_standards.md`
- **Team**: Ask team members for review and feedback on your implementation
