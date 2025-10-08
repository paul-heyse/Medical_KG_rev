## Why

The REST router currently mixes JSON:API formatting helpers, request/tenant validation, and gateway service orchestration inside each endpoint. This creates tightly coupled route handlers that are difficult to test, maintain, and reuse across different protocols. The formatting and security concerns are intertwined with business logic, making it hard to ensure consistent response shapes across REST, GraphQL, and gRPC protocols.

## What Changes

- **Extract presentation layer**: Create a thin presenter layer responsible for JSON:API envelopes and OData parsing
- **Dependency injection for routes**: Make each route an orchestration of dependencies rather than a mix of formatting and security concerns
- **Consistent response formatting**: Reuse the same presentation logic across all protocols that need JSON:API response shapes
- **Separation of concerns**: Clearly separate HTTP formatting, validation, and business logic orchestration
- **Improved testability**: Make it easier to test route logic independently of formatting concerns

## Impact

- **Affected specs**: `specs/multi-protocol-gateway/spec.md` - Gateway architecture and response formatting requirements
- **Affected code**:
  - `src/Medical_KG_rev/gateway/rest/router.py` - Extract presentation layer and update route handlers
  - `src/Medical_KG_rev/gateway/` - Create presentation layer module
  - `src/Medical_KG_rev/gateway/graphql/` - Update to use shared presentation logic
  - `src/Medical_KG_rev/gateway/grpc/` - Update to use shared presentation logic
- **Affected systems**: Multi-protocol API gateway, response formatting, request validation
