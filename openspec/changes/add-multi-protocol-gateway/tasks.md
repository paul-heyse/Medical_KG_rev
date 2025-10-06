# Implementation Tasks: Multi-Protocol API Gateway

## 1. REST API (OpenAPI 3.1 + JSON:API + OData)

- [ ] 1.1 Create FastAPI application with CORS, middleware
- [ ] 1.2 Implement `/ingest/clinicaltrials` endpoint (POST)
- [ ] 1.3 Implement `/ingest/dailymed` endpoint (POST)
- [ ] 1.4 Implement `/ingest/pmc` endpoint (POST)
- [ ] 1.5 Implement `/chunk` endpoint (POST)
- [ ] 1.6 Implement `/embed` endpoint (POST)
- [ ] 1.7 Implement `/retrieve` endpoint (POST)
- [ ] 1.8 Implement `/map/el` entity linking endpoint (POST)
- [ ] 1.9 Implement `/extract/{kind}` endpoints (pico, effects, ae, dose, eligibility)
- [ ] 1.10 Implement `/kg/write` knowledge graph write endpoint (POST)
- [ ] 1.11 Add JSON:API response formatting middleware
- [ ] 1.12 Add OData query parameter parsing ($filter, $select, $expand, $top, $skip)
- [ ] 1.13 Implement RFC 7807 Problem Details error responses
- [ ] 1.14 Add 207 Multi-Status responses for batch operations
- [ ] 1.15 Generate OpenAPI 3.1 spec from FastAPI
- [ ] 1.16 Add Swagger UI at `/docs/openapi`
- [ ] 1.17 Write Schemathesis contract tests

## 2. GraphQL API

- [ ] 2.1 Set up Strawberry GraphQL or Ariadne
- [ ] 2.2 Generate GraphQL types from Pydantic models
- [ ] 2.3 Define Query type with document(), organization(), search() resolvers
- [ ] 2.4 Define Mutation type for ingest, chunk, embed, extract, kg write operations
- [ ] 2.5 Implement DataLoader pattern for efficient batching
- [ ] 2.6 Add relationship resolvers (Document.organization, Document.claims)
- [ ] 2.7 Implement filtering and pagination arguments
- [ ] 2.8 Add GraphQL Playground at `/docs/graphql`
- [ ] 2.9 Export GraphQL SDL to `docs/schema.graphql`
- [ ] 2.10 Write GraphQL Inspector CI checks for breaking changes
- [ ] 2.11 Add comprehensive GraphQL query tests

## 3. gRPC Service Definitions

- [ ] 3.1 Set up Buf for proto management
- [ ] 3.2 Define `mineru.proto` (PDF processing service)
- [ ] 3.3 Define `embedding.proto` (embedding generation service)
- [ ] 3.4 Define `extraction.proto` (information extraction service)
- [ ] 3.5 Define `ingestion.proto` (ingestion orchestration service)
- [ ] 3.6 Generate Python code from protos with Buf
- [ ] 3.7 Add Buf lint and breaking change detection to CI
- [ ] 3.8 Create gRPC server stub templates
- [ ] 3.9 Add gRPC health check implementation
- [ ] 3.10 Write gRPC service tests

## 4. AsyncAPI and Server-Sent Events

- [ ] 4.1 Implement SSE endpoint `/jobs/{id}/events`
- [ ] 4.2 Create event stream manager with pub/sub
- [ ] 4.3 Define event payloads (jobs.started, jobs.progress, jobs.completed, jobs.failed)
- [ ] 4.4 Write AsyncAPI specification to `docs/asyncapi.yaml`
- [ ] 4.5 Add AsyncAPI UI at `/docs/asyncapi`
- [ ] 4.6 Implement event authentication and authorization
- [ ] 4.7 Write SSE integration tests

## 5. SOAP Adapter (Legacy)

- [ ] 5.1 Create minimal SOAP wrapper using Zeep or Spyne
- [ ] 5.2 Define WSDL for key operations (ingest, retrieve)
- [ ] 5.3 Map SOAP operations to internal REST/gRPC calls
- [ ] 5.4 Add SOAP endpoint at `/soap`
- [ ] 5.5 Write SOAP integration tests

## 6. API Documentation Portal

- [ ] 6.1 Set up static docs site (MkDocs or Docusaurus)
- [ ] 6.2 Embed Swagger UI for REST API
- [ ] 6.3 Embed GraphQL Playground
- [ ] 6.4 Embed AsyncAPI UI
- [ ] 6.5 Add authentication guide
- [ ] 6.6 Add example workflows and tutorials
- [ ] 6.7 Deploy docs to GitHub Pages or container

## 7. Cross-Protocol Integration

- [ ] 7.1 Ensure REST, GraphQL, gRPC share same business logic
- [ ] 7.2 Add protocol-agnostic service layer
- [ ] 7.3 Implement consistent error handling across protocols
- [ ] 7.4 Add request/response logging for all protocols
- [ ] 7.5 Performance test all protocol endpoints with k6
