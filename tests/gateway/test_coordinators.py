from types import SimpleNamespace

import pytest

pytest.importorskip("aiolimiter")
pytest.importorskip("pybreaker")

from aiolimiter import AsyncLimiter
from pybreaker import CircuitBreaker

from Medical_KG_rev.chunking.exceptions import ChunkingFailedError
from Medical_KG_rev.embeddings.ports import EmbeddingRecord
from Medical_KG_rev.gateway.coordinators import (
    ChunkingCoordinator,
    ChunkingRequest,
    CoordinatorConfig,
    CoordinatorError,
    EmbeddingCoordinator,
    EmbeddingRequest,
    JobLifecycleManager,
)
from Medical_KG_rev.gateway.models import DocumentChunk, ProblemDetail
from Medical_KG_rev.gateway.sse.manager import EventStreamManager
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.gateway.chunking_errors import ChunkingErrorTranslator
from Medical_KG_rev.services.retrieval.chunking import ChunkCommand
from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from Medical_KG_rev.services.embedding.namespace.schema import EmbeddingKind, NamespaceConfig
from Medical_KG_rev.services.embedding.policy import NamespaceAccessDecision


class _StubChunk:
    def __init__(self, body: str, *, granularity: str = "paragraph", chunker: str = "test") -> None:
        self.body = body
        self.granularity = granularity
        self.chunker = chunker
        self.meta = {"token_count": len(body.split())}


class _StubChunker:
    def __init__(self, *, raise_error: Exception | None = None) -> None:
        self._raise = raise_error

    def chunk(self, command: ChunkCommand) -> list[_StubChunk]:  # noqa: D401
        if self._raise:
            raise self._raise
        return [_StubChunk(command.text)]
    def chunk(self, tenant_id: str, document_id: str, text: str, options) -> list[_StubChunk]:  # noqa: D401
        if self._raise:
            raise self._raise
        return [_StubChunk(text)]


class _StubEmbeddingPersister:
    def __init__(self) -> None:
        self.calls: list[tuple[list[EmbeddingRecord], object]] = []

    def persist_batch(self, records, context):
        self.calls.append((list(records), context))
        return SimpleNamespace(persisted=len(records))


class _StubTelemetry:
    def __init__(self) -> None:
        self.started: list[dict[str, str]] = []
        self.completed: list[dict[str, object]] = []
        self.failures: list[Exception] = []

    def record_embedding_started(self, *, namespace: str, tenant_id: str, model: str) -> None:
        self.started.append({"namespace": namespace, "tenant_id": tenant_id, "model": model})

    def record_embedding_completed(
        self,
        *,
        namespace: str,
        tenant_id: str,
        model: str,
        provider: str,
        duration_ms: float,
        embeddings: int,
    ) -> None:
        self.completed.append(
            {
                "namespace": namespace,
                "tenant_id": tenant_id,
                "model": model,
                "provider": provider,
                "duration_ms": duration_ms,
                "embeddings": embeddings,
            }
        )

    def record_embedding_failure(self, *, namespace: str, tenant_id: str, error: Exception) -> None:
        self.failures.append(error)


class _StubEmbedder:
    def __init__(self, records: list[EmbeddingRecord]) -> None:
        self._records = records

    def embed_documents(self, request) -> list[EmbeddingRecord]:  # noqa: D401
        return self._records


class _StubEmbeddingRegistry:
    def __init__(self, namespace: str, embedder: _StubEmbedder, registry: EmbeddingNamespaceRegistry) -> None:
        self._namespace = namespace
        self._embedder = embedder
        self.namespace_registry = registry

    def get(self, namespace: str) -> _StubEmbedder:  # noqa: D401
        if namespace != self._namespace:
            raise ValueError(f"Unknown namespace {namespace}")
        return self._embedder


class _AllowAllPolicy:
    def __init__(self, namespace: str, config: NamespaceConfig) -> None:
        self._namespace = namespace
        self._config = config

    def evaluate(self, *, namespace: str, tenant_id: str, required_scope: str) -> NamespaceAccessDecision:
        if namespace != self._namespace:
            raise ValueError("unexpected namespace")
        return NamespaceAccessDecision(
            namespace=namespace,
            tenant_id=tenant_id,
            scope=required_scope,
            allowed=True,
            config=self._config,
            policy="allow-all",
        )


@pytest.fixture
def lifecycle() -> JobLifecycleManager:
    ledger = JobLedger()
    events = EventStreamManager()
    return JobLifecycleManager(ledger, events)


@pytest.fixture
def chunk_config() -> CoordinatorConfig:
    return CoordinatorConfig(
        name="chunking-test",
        retry_attempts=2,
        retry_wait_base=0.01,
        retry_wait_max=0.05,
        breaker=CircuitBreaker("chunking-test"),
        limiter=AsyncLimiter(5, 1),
    )


@pytest.fixture
def embed_config() -> CoordinatorConfig:
    return CoordinatorConfig(
        name="embedding-test",
        retry_attempts=2,
        retry_wait_base=0.01,
        retry_wait_max=0.05,
        breaker=CircuitBreaker("embedding-test"),
        limiter=AsyncLimiter(5, 1),
    )


def test_job_lifecycle_creates_and_completes(lifecycle: JobLifecycleManager) -> None:
    job_id = lifecycle.create_job("tenant-a", "chunk", metadata={"document": "doc-1"})
    lifecycle.update_metadata(job_id, {"chunks": 2})
    lifecycle.mark_completed(job_id, payload={"chunks": 2})

    entry = lifecycle.ledger.get(job_id)
    assert entry.status == "completed"
    assert entry.metadata["chunks"] == 2

    history = lifecycle.events.history(job_id)
    assert [event.type for event in history] == ["jobs.started", "jobs.completed"]


def test_job_lifecycle_failure_records_event(lifecycle: JobLifecycleManager) -> None:
    job_id = lifecycle.create_job("tenant-a", "chunk")
    lifecycle.mark_failed(job_id, reason="boom", stage="chunk")

    entry = lifecycle.ledger.get(job_id)
    assert entry.status == "failed"
    assert entry.error_reason == "boom"

    events = lifecycle.events.history(job_id)
    assert events[-1].payload["reason"] == "boom"


def test_chunking_coordinator_success(lifecycle: JobLifecycleManager, chunk_config: CoordinatorConfig) -> None:
    chunker = _StubChunker()
    translator = ChunkingErrorTranslator(strategies=["section"])
    coordinator = ChunkingCoordinator(
        lifecycle=lifecycle,
        chunker=chunker,
        config=chunk_config,
        errors=translator,
    )
    coordinator = ChunkingCoordinator(lifecycle=lifecycle, chunker=chunker, config=chunk_config)

    request = ChunkingRequest(
        tenant_id="tenant-a",
        document_id="doc-1",
        text="hello world",
        strategy="section",
        chunk_size=256,
        overlap=0.1,
        options={"text": "hello world"},
    )
    result = coordinator(request)

    assert len(result.chunks) == 1
    assert isinstance(result.chunks[0], DocumentChunk)
    assert result.metadata == {"chunks": 1, "strategy": "section"}
    assert result.metadata == {"chunks": 1}


def test_chunking_coordinator_error_maps_problem(lifecycle: JobLifecycleManager, chunk_config: CoordinatorConfig) -> None:
    chunker = _StubChunker(raise_error=ChunkingFailedError("failed"))
    translator = ChunkingErrorTranslator(strategies=["section"])
    coordinator = ChunkingCoordinator(
        lifecycle=lifecycle,
        chunker=chunker,
        config=chunk_config,
        errors=translator,
    )
    coordinator = ChunkingCoordinator(lifecycle=lifecycle, chunker=chunker, config=chunk_config)

    request = ChunkingRequest(
        tenant_id="tenant-a",
        document_id="doc-1",
        text="boom",
        strategy="section",
        chunk_size=256,
        overlap=0.1,
        options={"text": "boom"},
    )

    with pytest.raises(CoordinatorError) as err:
        coordinator(request)

    problem = err.value.context.get("problem")
    assert isinstance(problem, ProblemDetail)
    assert problem.title == "Chunking failed"


def test_embedding_coordinator_success(lifecycle: JobLifecycleManager, embed_config: CoordinatorConfig) -> None:
    registry = EmbeddingNamespaceRegistry()
    namespace = "single_vector.demo.3.v1"
    ns_config = NamespaceConfig(
        name="demo",
        kind=EmbeddingKind.SINGLE_VECTOR,
        model_id="demo-model",
        dim=3,
        provider="unit-test",
    )
    registry.register(namespace, ns_config)

    vectors = [[1.0, 0.0, 0.0]]
    records = [
        EmbeddingRecord(
            id="vec-1",
            tenant_id="tenant-a",
            namespace=namespace,
            model_id="demo-model",
            model_version="v1",
            kind="single_vector",
            dim=3,
            vectors=vectors,
            metadata={},
        )
    ]
    embedder = _StubEmbedder(records)
    stub_registry = _StubEmbeddingRegistry(namespace, embedder, registry)
    persister = _StubEmbeddingPersister()
    telemetry = _StubTelemetry()
    policy = _AllowAllPolicy(namespace, ns_config)
    coordinator = EmbeddingCoordinator(
        lifecycle=lifecycle,
        registry=stub_registry,
        namespace_registry=registry,
        policy=policy,
        persister=persister,
        telemetry=telemetry,
        config=embed_config,
    )

    request = EmbeddingRequest(
        tenant_id="tenant-a",
        namespace=namespace,
        texts=["alpha"],
        options=None,
    )
    result = coordinator(request)

    assert result.response is not None
    assert len(result.response.embeddings) == 1
    assert persister.calls[0][0][0].metadata["namespace"] == namespace
    assert telemetry.started and telemetry.completed


def test_embedding_coordinator_empty_input_returns_completed(
    lifecycle: JobLifecycleManager, embed_config: CoordinatorConfig
) -> None:
    registry = EmbeddingNamespaceRegistry()
    namespace = "single_vector.demo.3.v1"
    ns_config = NamespaceConfig(
        name="demo",
        kind=EmbeddingKind.SINGLE_VECTOR,
        model_id="demo-model",
        dim=3,
        provider="unit-test",
    )
    registry.register(namespace, ns_config)

    embedder = _StubEmbedder([])
    stub_registry = _StubEmbeddingRegistry(namespace, embedder, registry)
    persister = _StubEmbeddingPersister()
    telemetry = _StubTelemetry()
    policy = _AllowAllPolicy(namespace, ns_config)
    coordinator = EmbeddingCoordinator(
        lifecycle=lifecycle,
        registry=stub_registry,
        namespace_registry=registry,
        policy=policy,
        persister=persister,
        telemetry=telemetry,
        config=embed_config,
    )

    request = EmbeddingRequest(
        tenant_id="tenant-a",
        namespace=namespace,
        texts=[],
        options=None,
    )
    result = coordinator(request)

    assert result.response is not None
    assert len(result.response.embeddings) == 0
    assert persister.calls == []


def test_embedding_coordinator_invalid_text_raises(
    lifecycle: JobLifecycleManager, embed_config: CoordinatorConfig
) -> None:
    registry = EmbeddingNamespaceRegistry()
    namespace = "single_vector.demo.3.v1"
    ns_config = NamespaceConfig(
        name="demo",
        kind=EmbeddingKind.SINGLE_VECTOR,
        model_id="demo-model",
        dim=3,
        provider="unit-test",
    )
    registry.register(namespace, ns_config)

    embedder = _StubEmbedder([])
    stub_registry = _StubEmbeddingRegistry(namespace, embedder, registry)
    persister = _StubEmbeddingPersister()
    telemetry = _StubTelemetry()
    policy = _AllowAllPolicy(namespace, ns_config)
    coordinator = EmbeddingCoordinator(
        lifecycle=lifecycle,
        registry=stub_registry,
        namespace_registry=registry,
        policy=policy,
        persister=persister,
        telemetry=telemetry,
        config=embed_config,
    )

    request = EmbeddingRequest(
        tenant_id="tenant-a",
        namespace=namespace,
        texts=["   "],
        options=None,
    )

    with pytest.raises(CoordinatorError) as err:
        coordinator(request)

    problem = err.value.context.get("problem")
    assert isinstance(problem, ProblemDetail)
    assert problem.status == 400
