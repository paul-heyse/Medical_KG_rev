# Adapter SDK Guide

The adapter SDK makes it straightforward to add new biomedical data sources by
standardising the ingestion lifecycle.

## Lifecycle

1. `fetch(context)` → Pull raw payloads from an upstream service.
2. `parse(payloads, context)` → Transform payloads into IR `Document` objects.
3. `validate(documents, context)` → Enforce structural rules (can return
   warnings but should raise on fatal errors).
4. `write(documents, context)` → Persist results to storage or downstream
   pipelines.

`BaseAdapter.run()` orchestrates the sequence, ensuring consistency across
protocols.

## Registry

Adapters can be registered for dynamic discovery via
`Medical_KG_rev.adapters.registry`. This enables configuration-driven
initialisation based on domain metadata.

## YAML Configuration

The SDK includes `load_adapter_config` to parse YAML descriptors that map HTTP
requests to IR fields. Complex transformations can still use Python code via the
example adapter pattern documented in `Medical_KG_rev.adapters.example`.

## Testing

`Medical_KG_rev.adapters.testing.run_adapter` executes adapters using an in-memory
context, simplifying unit tests without requiring external services.
