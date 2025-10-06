# Design Document: Multi-Protocol API Gateway

## Context

The system must serve diverse clients with varying technical requirements: web frontends prefer GraphQL, internal microservices need gRPC performance, legacy systems require SOAP, and REST remains the standard for broad integration. Providing all protocols over a single backend maximizes flexibility while minimizing development overhead.

## Goals / Non-Goals

### Goals

- Single unified backend exposed through 5 protocols
- Full spec compliance (OpenAPI 3.1, GraphQL, gRPC, SOAP, AsyncAPI)
- Consistent auth/error handling across all protocols
- Shared business logic (no protocol-specific duplication)
- Interactive documentation for all APIs

### Non-Goals

- Not building separate services per protocol (unified gateway)
- Not GraphQL Federation v1 (may add later)
- Not WebSocket (SSE sufficient for v1)
- Not rate limiting per protocol (handled at gateway level)

## Decisions

### Decision 1: FastAPI for REST + GraphQL + SSE

**What**: Use FastAPI as primary framework, add Strawberry GraphQL
**Why**: FastAPI auto-generates OpenAPI, supports SSE natively, excellent async performance
**Alternatives**: Flask (less async support), Django (overkill), pure ASGI (too low-level)

### Decision 2: JSON:API for REST Responses

**What**: All REST responses follow JSON:API v1.1 specification
**Why**: Standardized resource format, client libraries available, relationship handling
**Implementation**: Middleware wraps responses in `{data: [...], meta: {...}}` format

### Decision 3: OData Query Syntax

**What**: Support `$filter`, `$select`, `$expand` query parameters
**Why**: Powerful querying without custom DSL, familiar to enterprise developers
**Example**: `GET /documents?$filter=status eq 'active'&$select=title,date`

### Decision 4: GraphQL Schema Generation from Pydantic

**What**: Auto-generate GraphQL types from existing Pydantic models
**Why**: Single source of truth, reduces maintenance
**Library**: Strawberry with Pydantic integration

### Decision 5: Buf for gRPC Management

**What**: Use Buf for proto linting, breaking change detection, code generation
**Why**: Industry best practice, prevents accidental breaking changes
**CI Integration**: `buf lint`, `buf breaking`, `buf generate` in pipeline

### Decision 6: Minimal SOAP Support

**What**: Thin SOAP adapter that forwards to REST/gRPC
**Why**: Legacy compatibility without maintaining duplicate logic
**Scope**: Only critical operations (ingest, retrieve), not full feature parity

### Decision 7: Server-Sent Events over WebSocket

**What**: Use SSE for real-time job updates
**Why**: Simpler than WebSocket, works over HTTP, sufficient for server-to-client push
**When to WebSocket**: Only if bidirectional real-time needed (not in v1)

## Architecture

### Request Flow

```
Client Request
    ↓
Protocol Layer (FastAPI/gRPC/SOAP handler)
    ↓
Auth Middleware (JWT validation, scope check)
    ↓
Service Layer (protocol-agnostic business logic)
    ↓
Storage/External APIs
    ↓
Response Formatter (protocol-specific)
    ↓
Client Response
```

### Protocol-to-Service Mapping

- REST `/ingest/clinicaltrials` → IngestionService.ingest_trials()
- GraphQL `mutation { startIngestion(...) }` → IngestionService.ingest_trials()
- gRPC `IngestionService.StartIngest()` → IngestionService.ingest_trials()

All protocols call the same underlying service methods.

## Risks / Trade-offs

### Risk 1: GraphQL N+1 Query Problem

**Mitigation**: Use DataLoader pattern for batching, add query cost analysis

### Risk 2: Protocol-Specific Bugs

**Mitigation**: Comprehensive integration tests per protocol, shared test fixtures

### Risk 3: gRPC Learning Curve

**Mitigation**: Provide example .proto files, thorough documentation

### Risk 4: SSE Connection Limits

**Mitigation**: Implement connection pooling, add reconnection logic in clients

## Migration Plan

New capability (no migration). Future API versions use `/v2/` prefix for breaking changes.

## Open Questions

1. **Q**: Should GraphQL support subscriptions via WebSocket?
   **A**: Defer to v2 if demand exists; SSE sufficient for job updates

2. **Q**: How to version gRPC services?
   **A**: Use proto package versioning (ai.mercor.embed.v1, v2, etc.)

3. **Q**: Support GraphQL Federation?
   **A**: Not initially; revisit if microservices need independent GraphQL schemas
