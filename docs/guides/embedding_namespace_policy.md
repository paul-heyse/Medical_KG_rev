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

## Runtime configuration

Namespace policy and persister behaviour are exposed via `AppSettings.embedding`. Defaults can be overridden through environment variables (`MK_EMBEDDING__...`) or configuration files.

```yaml
embedding:
  policy:
    cache_ttl_seconds: 60.0
    max_cache_entries: 512
    dry_run: false
  persister:
    backend: vector_store  # vector_store | database | dry_run | hybrid
    cache_limit: 256
    hybrid_backends:
      sparse: database
      neural_sparse: vector_store
```

## Administrative API endpoints

The REST router exposes observability and configuration management endpoints secured by the `embed:admin` scope:

- `GET /v1/namespaces/policy` – current policy status and cache metrics.
- `PATCH /v1/namespaces/policy` – update cache TTL, entry limits, or toggle dry-run.
- `GET /v1/namespaces/policy/diagnostics` – detailed cache keys and statistics.
- `GET /v1/namespaces/policy/health` – health summary suitable for alerts.
- `GET /v1/namespaces/policy/metrics` – operational metrics snapshot.
- `POST /v1/namespaces/policy/cache/invalidate` – clear cached decisions globally or per-namespace.

## Migration utility

The `scripts/migrate_namespace_policy.py` helper exports the active policy and persister configuration for auditing or for seeding infrastructure-as-code manifests:

```bash
python scripts/migrate_namespace_policy.py --output embedding-runtime.json
```

The script serialises the runtime configuration using the same abstractions as the gateway, ensuring drift between code and deployment manifests is easy to detect.

## Testing and troubleshooting

- Unit tests (`tests/services/embedding/test_namespace_policy.py`, `tests/services/embedding/test_embedding_persister.py`) validate caching, dry-run behaviour, and persister routing.
- Integration tests (`tests/gateway/test_gateway_namespace_policy_admin.py`) cover the administrative surfaces exposed by `GatewayService`.
- The diagnostics endpoint returns cache keys in `namespace:tenant:scope` form to accelerate debugging of unexpected denials.
