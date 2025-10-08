## Why

The embedding flow currently reaches into the namespace registry, storage router, and metrics directly, applying normalization rules and persistence logic inline. The namespace validation in the gateway constructs responses and calls registry helpers directly. This creates tight coupling between the gateway layer and storage/namespace concerns, making it difficult to test embedding workflows in isolation and complicating the introduction of alternative implementations like dry-run or mock storage.

## What Changes

- **Define `NamespaceAccessPolicy` interface**: Create explicit contracts for namespace validation, routing, and access control that can be implemented by different policy types
- **Extract `EmbeddingPersister` interface**: Abstract embedding storage operations behind a clean interface that supports different persistence strategies (database, vector store, cache)
- **Create `EmbeddingTelemetry` interface**: Encapsulate metrics collection, tracing, and monitoring for embedding operations
- **Simplify `GatewayService.embed`**: Allow the method to focus on orchestrating collaborators rather than handling validation, normalization, persistence, and business metrics inline
- **Enable alternative implementations**: Support dry-run, mock, and alternative storage implementations through interface substitution

## Impact

- **Affected specs**: `specs/gateway/spec.md` - Namespace and storage policy interface requirements
- **Affected code**:
  - `src/Medical_KG_rev/gateway/services.py` - Extract storage and namespace logic into dedicated interfaces
  - `src/Medical_KG_rev/services/embedding/` - Update to use new policy and persister interfaces
  - `src/Medical_KG_rev/gateway/rest/router.py` - Update namespace validation to use policy interface
- **Affected systems**: Embedding pipeline, storage abstraction, namespace management, testing isolation
