# Adapter SDK Guide

The adapter SDK makes it straightforward to add new biomedical data sources by
standardising the ingestion lifecycle.

## Lifecycle

1. `fetch(request)` → Pull raw payloads from an upstream service and return an
   `AdapterResponse` envelope.
2. `parse(response, request)` → Transform payloads into canonical IR objects
   such as `Document` instances.
3. `validate(response, request)` → Enforce structural rules before the
   orchestrator persists or forwards the documents.

The `AdapterPluginManager` materialises this lifecycle as an
`AdapterPipeline`. When you call `manager.invoke(name, request)` the manager
returns an `AdapterInvocationResult` containing the underlying
`AdapterExecutionContext`, canonical response, validation outcome, and detailed
stage timings. The historic `execute`/`run` helpers now build on top of
`invoke`—`execute` returns the context while `run` continues to raise when the
pipeline cannot produce a valid `AdapterResponse`.

## Registry

Adapters are registered with the pluggy-backed
`Medical_KG_rev.adapters.AdapterPluginManager`. Registration may happen at
runtime (e.g. `manager.register(ClinicalTrialsAdapterPlugin())`) or via entry
points declared in `pyproject.toml`. The manager groups adapters by domain,
exposes metadata through the gateway, and drives orchestration execution.

## YAML Configuration

The SDK includes `load_adapter_config` to parse YAML descriptors that map HTTP
requests to IR fields. These descriptors produce legacy adapter classes that can
be wrapped by domain-specific plugins (e.g. `ClinicalTrialsAdapterPlugin`) while
teams incrementally migrate business logic into first-class plugin
implementations.

## Testing

Instantiate a plugin directly (or use `AdapterPluginManager`) to exercise the
full lifecycle in tests. Using the manager exposes the pipeline context and
telemetry so you can assert on intermediate artefacts and timings:

```python
request = AdapterRequest(
    tenant_id="tenant", correlation_id="corr", domain=AdapterDomain.BIOMEDICAL
)
manager = AdapterPluginManager()
manager.register(ClinicalTrialsAdapterPlugin())
result = manager.invoke("clinicaltrials", request)
assert result.ok
assert result.metrics.duration_ms > 0
assert result.response is not None
```
