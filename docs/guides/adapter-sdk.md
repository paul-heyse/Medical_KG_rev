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
`AdapterPipeline`. When you call `manager.execute(name, request)` the manager
creates an `AdapterExecutionState`, runs it through each pipeline stage, and
returns the enriched state object. `manager.run(...)` is a thin wrapper that
enforces validation and returns the final `AdapterResponse` for convenience.

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
full lifecycle in tests. Using the manager exposes the pipeline state so you
can assert on intermediate artefacts:

```python
request = AdapterRequest(
    tenant_id="tenant", correlation_id="corr", domain=AdapterDomain.BIOMEDICAL
)
manager = AdapterPluginManager()
manager.register(ClinicalTrialsAdapterPlugin())
state = manager.execute("clinicaltrials", request)
state.raise_for_validation()  # optional in tests
response = state.ensure_response()
assert response.items
```
