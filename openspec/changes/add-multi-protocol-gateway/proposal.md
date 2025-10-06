# Change Proposal: Multi-Protocol API Gateway

## Why

Implement a unified API gateway that exposes the system through multiple standard protocols (REST/OpenAPI, GraphQL, gRPC, SOAP, AsyncAPI/SSE) to meet clients where they are. Each protocol must adhere to its specification while sharing a common backend, enabling RESTful integration, graph-based querying, high-speed internal calls, legacy enterprise needs, and reactive event-driven patterns.

## What Changes

- Implement OpenAPI 3.1 REST endpoints with JSON:API formatting and OData filtering
- Build GraphQL API with typed schema, resolvers, and introspection
- Create gRPC service definitions (Protocol Buffers) for internal microservices
- Add SOAP adapter for legacy integration
- Implement AsyncAPI-documented Server-Sent Events (SSE) for job status streaming
- Create comprehensive OpenAPI specification document
- Generate GraphQL SDL from Pydantic models
- Define gRPC .proto files with Buf validation
- Build AsyncAPI specification for event channels
- Add protocol-specific middleware (content negotiation, error formatting)
- Implement API versioning strategy

## Impact

- **Affected specs**: NEW capabilities `rest-api`, `graphql-api`, `grpc-services`, `asyncapi-events`
- **Affected code**:
  - `src/Medical_KG_rev/gateway/` - FastAPI application and routers
  - `src/Medical_KG_rev/gateway/rest/` - REST endpoint implementations
  - `src/Medical_KG_rev/gateway/graphql/` - GraphQL schema and resolvers
  - `src/Medical_KG_rev/gateway/sse/` - Server-Sent Events implementation
  - `src/Medical_KG_rev/proto/` - gRPC Protocol Buffer definitions
  - `docs/openapi.yaml` - Complete OpenAPI 3.1 specification
  - `docs/schema.graphql` - GraphQL SDL
  - `docs/asyncapi.yaml` - AsyncAPI specification
  - `tests/contract/` - Schemathesis and GraphQL Inspector tests
