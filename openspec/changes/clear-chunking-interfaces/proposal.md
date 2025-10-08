## Why

The `ChunkingService.chunk` method currently relies on positional/keyword juggling, manual metadata parsing, and inline Dagster invocation, obscuring the contract that downstream callers must satisfy. The gateway's `chunk_document` method contains numerous detailed `ProblemDetail` branches that could be centralized for reuse across protocols. This creates tight coupling between the chunking service and its callers, making it difficult to understand the required inputs and test chunking workflows in isolation.

## What Changes

- **Introduce `ChunkCommand` dataclass**: Create a structured request object with explicit fields for tenant, document, text, options, and context to document the chunking interface clearly
- **Extract `ChunkingErrorTranslator`**: Lift the detailed `ProblemDetail` branches from `GatewayService.chunk_document` into a reusable error mapper that converts domain exceptions into API-ready responses
- **Simplify service interface**: Allow the chunking service to hide argument normalization and stage execution details behind the `ChunkCommand` interface
- **Centralize error handling**: Make error mappings reusable across REST, GraphQL, gRPC, and other protocol handlers
- **Improve testability**: Enable focused testing of chunking logic without protocol-specific concerns

## Impact

- **Affected specs**: `specs/gateway/spec.md` - Chunking interface and error handling requirements
- **Affected code**:
  - `src/Medical_KG_rev/services/retrieval/chunking.py` - Update to use ChunkCommand interface
  - `src/Medical_KG_rev/gateway/services.py` - Extract error mapping into ChunkingErrorTranslator
  - `src/Medical_KG_rev/gateway/rest/router.py` - Update chunking endpoints to use new error translator
- **Affected systems**: Chunking pipeline, error handling, protocol integration, testing isolation
