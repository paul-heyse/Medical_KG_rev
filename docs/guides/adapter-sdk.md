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

The `AdapterPluginManager` calls the hooks in order via `manager.run(name,
request)`, ensuring a consistent lifecycle for all plugins.

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
full lifecycle in tests:

```python
request = AdapterRequest(
    tenant_id="tenant", correlation_id="corr", domain=AdapterDomain.BIOMEDICAL
)
plugin = ClinicalTrialsAdapterPlugin()
response = plugin.fetch(request)
response = plugin.parse(response, request)
outcome = plugin.validate(response, request)
assert outcome.valid
```
