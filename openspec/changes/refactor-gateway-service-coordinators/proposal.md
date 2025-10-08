## Why

The `GatewayService` class currently serves as a monolithic service locator that wires together adapter discovery, ingestion, chunking, embedding, retrieval, entity linking, extraction, namespace management, and ledger book-keeping. With twelve collaborators in the constructor and additional defaults setup, the class spreads responsibilities across every protocol-facing operation. This violates the Single Responsibility Principle and creates tight coupling between unrelated domain concerns, making the codebase difficult to understand, test, and maintain.

The existing helper methods (`_new_job`, `_complete_job`, `_fail_job`) demonstrate reusable patterns for job lifecycle management that are currently duplicated across workflow methods. By extracting these patterns into a dedicated `JobLifecycleManager`, we can eliminate duplication while providing a clean interface for job state transitions and event streaming.

## What Changes

- **Split into focused coordinators**: Decompose `GatewayService` into use-case-specific coordinators (`IngestionCoordinator`, `EmbeddingCoordinator`, `RetrievalCoordinator`) that each encapsulate domain logic, ledger updates, and error mapping for their respective workflows
- **Extract `JobLifecycleManager`**: Promote the existing helper methods into a dedicated service that handles job creation, completion, failure states, and associated ledger/event-stream operations
- **Narrow interface dependencies**: Replace the monolithic `GatewayService` dependency with narrow interfaces (e.g., `EmbedderGateway.embed(request)`) that protocol handlers can depend on
- **Maintain backward compatibility**: Ensure existing protocol handlers continue to work during the transition by providing adapter/facade patterns where needed
- **Improve testability**: Enable focused unit testing of individual coordinators and easier mocking of dependencies

## Impact

- **Affected specs**: `specs/gateway/spec.md` - Gateway service architecture and coordinator interfaces
- **Affected code**:
  - `src/Medical_KG_rev/gateway/services.py` - Refactor into coordinator classes and JobLifecycleManager
  - `src/Medical_KG_rev/gateway/rest/router.py` - Update to use coordinator interfaces
  - `src/Medical_KG_rev/gateway/graphql/` - Update resolvers to use coordinator interfaces
  - `src/Medical_KG_rev/gateway/grpc/` - Update service implementations to use coordinator interfaces
  - `src/Medical_KG_rev/gateway/soap/` - Update handlers to use coordinator interfaces
- **Affected systems**: Multi-protocol API gateway, job orchestration, error handling, service composition
