# Implementation Tasks: Multi-Protocol API Gateway

## 1. REST API (OpenAPI 3.1 + JSON:API + OData)

- [x] 1.1 Create FastAPI application with CORS, middleware
- [x] 1.2 Implement `/ingest/clinicaltrials` endpoint (POST)
- [x] 1.3 Implement `/ingest/dailymed` endpoint (POST)
- [x] 1.4 Implement `/ingest/pmc` endpoint (POST)
- [x] 1.5 Implement `/chunk` endpoint (POST)
- [x] 1.6 Implement `/embed` endpoint (POST)
- [x] 1.7 Implement `/retrieve` endpoint (POST)
- [x] 1.8 Implement `/map/el` entity linking endpoint (POST)
- [x] 1.9 Implement `/extract/{kind}` endpoints (pico, effects, ae, dose, eligibility)
- [x] 1.10 Implement `/kg/write` knowledge graph write endpoint (POST)
- [x] 1.11 Add JSON:API response formatting middleware
- [x] 1.12 Add OData query parameter parsing ($filter, $select, $expand, $top, $skip)
- [x] 1.13 Implement RFC 7807 Problem Details error responses
- [x] 1.14 Add 207 Multi-Status responses for batch operations
- [x] 1.15 Generate OpenAPI 3.1 spec from FastAPI
- [x] 1.16 Add Swagger UI at `/docs/openapi`
- [x] 1.17 Write Schemathesis contract tests

## 2. GraphQL API

- [x] 2.1 Set up Strawberry GraphQL or Ariadne
- [x] 2.2 Generate GraphQL types from Pydantic models
- [x] 2.3 Define Query type with document(), organization(), search() resolvers
- [x] 2.4 Define Mutation type for ingest, chunk, embed, extract, kg write operations
- [x] 2.5 Implement DataLoader pattern for efficient batching
- [x] 2.6 Add relationship resolvers (Document.organization, Document.claims)
- [x] 2.7 Implement filtering and pagination arguments
- [x] 2.8 Add GraphQL Playground at `/docs/graphql`
- [x] 2.9 Export GraphQL SDL to `docs/schema.graphql`
- [x] 2.10 Write GraphQL Inspector CI checks for breaking changes
- [x] 2.11 Add comprehensive GraphQL query tests

## 3. gRPC Service Definitions

- [x] 3.1 Set up Buf for proto management
- [x] 3.2 Define `mineru.proto` (PDF processing service)
- [x] 3.3 Define `embedding.proto` (embedding generation service)
- [x] 3.4 Define `extraction.proto` (information extraction service)
- [x] 3.5 Define `ingestion.proto` (ingestion orchestration service)
- [x] 3.6 Generate Python code from protos with Buf
- [x] 3.7 Add Buf lint and breaking change detection to CI
- [x] 3.8 Create gRPC server stub templates
- [x] 3.9 Add gRPC health check implementation
- [x] 3.10 Write gRPC service tests

## 4. AsyncAPI and Server-Sent Events

- [x] 4.1 Implement SSE endpoint `/jobs/{id}/events`
- [x] 4.2 Create event stream manager with pub/sub
- [x] 4.3 Define event payloads (jobs.started, jobs.progress, jobs.completed, jobs.failed)
- [x] 4.4 Write AsyncAPI specification to `docs/asyncapi.yaml`
- [x] 4.5 Add AsyncAPI UI at `/docs/asyncapi`
- [x] 4.6 Implement event authentication and authorization
- [x] 4.7 Write SSE integration tests

## 5. SOAP Adapter (Legacy)

- [x] 5.1 Create minimal SOAP wrapper using Zeep or Spyne
- [x] 5.2 Define WSDL for key operations (ingest, retrieve)
- [x] 5.3 Map SOAP operations to internal REST/gRPC calls
- [x] 5.4 Add SOAP endpoint at `/soap`
- [x] 5.5 Write SOAP integration tests

## 6. API Documentation Portal

- [x] 6.1 Set up static docs site (MkDocs or Docusaurus)
- [x] 6.2 Embed Swagger UI for REST API
- [x] 6.3 Embed GraphQL Playground
- [x] 6.4 Embed AsyncAPI UI
- [x] 6.5 Add authentication guide
- [x] 6.6 Add example workflows and tutorials
- [x] 6.7 Deploy docs to GitHub Pages or container

## 7. Cross-Protocol Integration

- [x] 7.1 Ensure REST, GraphQL, gRPC share same business logic
- [x] 7.2 Add protocol-agnostic service layer
- [x] 7.3 Implement consistent error handling across protocols
- [x] 7.4 Add request/response logging for all protocols
- [x] 7.5 Performance test all protocol endpoints with k6
