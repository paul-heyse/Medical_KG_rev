# Embedding Namespace Contracts

This guide summarises the new abstractions that decouple namespace governance from gateway orchestration. It should be referenced by teams extending embedding behaviour or onboarding new storage implementations.

## NamespaceAccessPolicy

`NamespaceAccessPolicy` is an abstract base class that centralises namespace validation and routing decisions. Implementations return a `NamespaceAccessDecision` describing whether a tenant/scope pair may interact with a namespace.

### Key capabilities

- Cached evaluations with eviction to prevent redundant registry lookups.
- Optional dry-run wrapping via `DryRunNamespacePolicy` for migrations.
- Mock/custom policies to support testing and tenant-specific rules.
- Health/debug snapshots for observability tooling.

### Usage example

```python
from Medical_KG_rev.services.embedding.namespace.registry import EmbeddingNamespaceRegistry
from Medical_KG_rev.services.embedding.policy import StandardNamespacePolicy

registry = EmbeddingNamespaceRegistry()
registry.register("single_vector.demo.1024.v1", config)
policy = StandardNamespacePolicy(registry)

decision = policy.evaluate(
    namespace="single_vector.demo.1024.v1",
    tenant_id="tenant-a",
    required_scope="embed:write",
)
if not decision.allowed:
    raise PermissionError(decision.reason)
```

## EmbeddingPersister

`EmbeddingPersister` abstracts persistence operations, allowing the gateway to call `persist_batch` without knowledge of the underlying storage topology.

### Built-in implementations

- `VectorStorePersister` – persists through the shared `StorageRouter`.
- `DatabasePersister` – maintains Neo4j-compatible snapshots.
- `DryRunPersister`/`MockPersister` – instrumentation for tests.
- `HybridPersister` – delegates by embedding kind for hybrid storage strategies.

### Usage example

```python
from Medical_KG_rev.embeddings.storage import StorageRouter
from Medical_KG_rev.services.embedding.persister import PersistenceContext, VectorStorePersister

router = StorageRouter()
persister = VectorStorePersister(router)
context = PersistenceContext(
    tenant_id="tenant-a",
    namespace="single_vector.demo.1024.v1",
    model="demo-model",
    provider="demo",
)
report = persister.persist_batch(records, context)
```

## EmbeddingTelemetry

`EmbeddingTelemetry` provides a single instrumentation surface for policies, persisters, and the gateway service. `StandardEmbeddingTelemetry` logs structured events and records Prometheus metrics (with safe fallbacks when Prometheus is unavailable).

### Usage example

```python
from Medical_KG_rev.services.embedding.telemetry import StandardEmbeddingTelemetry, TelemetrySettings

telemetry = StandardEmbeddingTelemetry(TelemetrySettings(enable_logging=False))
telemetry.record_embedding_completed(
    namespace="single_vector.demo.1024.v1",
    tenant_id="tenant-a",
    model="demo-model",
    provider="demo",
    duration_ms=32.4,
    embeddings=8,
)
```

## Gateway Integration

`GatewayService.embed` now consumes these abstractions. Namespace access decisions, persistence, and telemetry are injected during `GatewayService` construction, unlocking clean configuration overrides and simpler testing of embedding flows.
