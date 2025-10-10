# API Reference

This comprehensive API reference provides detailed documentation for all endpoints, request/response formats, and usage examples for the Medical_KG_rev system.

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Error Handling](#error-handling)
5. [REST API](#rest-api)
6. [GraphQL API](#graphql-api)
7. [gRPC API](#grpc-api)
8. [WebSocket API](#websocket-api)
9. [SDKs and Libraries](#sdks-and-libraries)
10. [Examples](#examples)

## API Overview

The Medical_KG_rev system provides a multi-protocol API gateway supporting:

- **REST API**: OpenAPI 3.0 compliant with JSON:API and OData support
- **GraphQL API**: Flexible query language for complex data retrieval
- **gRPC API**: High-performance binary protocol for microservices
- **WebSocket API**: Real-time updates and streaming data

### Base URLs

- **Production**: `https://api.medical-kg-rev.com`
- **Staging**: `https://staging-api.medical-kg-rev.com`
- **Development**: `https://dev-api.medical-kg-rev.com`

### API Versioning

The API uses URL-based versioning:

- **v1**: Current stable version
- **v2**: Beta version (subject to change)

## Authentication

### API Key Authentication

Include your API key in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.medical-kg-rev.com/v1/documents
```

### OAuth 2.0 Authentication

#### Client Credentials Flow

```bash
# Get access token
curl -X POST https://api.medical-kg-rev.com/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET"

# Response
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "read write"
}
```

#### Authorization Code Flow

```bash
# 1. Redirect user to authorization endpoint
https://api.medical-kg-rev.com/oauth/authorize?response_type=code&client_id=YOUR_CLIENT_ID&redirect_uri=YOUR_REDIRECT_URI&scope=read+write

# 2. Exchange authorization code for access token
curl -X POST https://api.medical-kg-rev.com/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code" \
  -d "code=AUTHORIZATION_CODE" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "redirect_uri=YOUR_REDIRECT_URI"
```

### JWT Token Structure

```json
{
  "sub": "user123",
  "tenant_id": "tenant456",
  "roles": ["researcher", "analyst"],
  "scopes": ["read", "write"],
  "iat": 1640995200,
  "exp": 1640998800,
  "iss": "medical-kg-rev"
}
```

## Rate Limiting

### Limits

- **Per User**: 1000 requests/hour
- **Per IP**: 5000 requests/hour
- **Burst**: 100 requests/minute

### Headers

All responses include rate limiting headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Retry-After: 3600
```

### Rate Limit Exceeded

When rate limit is exceeded, the API returns:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Try again in 3600 seconds.",
    "retry_after": 3600
  }
}
```

## Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "error": {
    "code": "error_code",
    "message": "Human readable error message",
    "details": {
      "field": "field_name",
      "issue": "Specific issue description"
    },
    "request_id": "req_123456789"
  }
}
```

### HTTP Status Codes

- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource conflict
- **422 Unprocessable Entity**: Validation error
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **502 Bad Gateway**: Gateway error
- **503 Service Unavailable**: Service temporarily unavailable

### Common Error Codes

- **validation_error**: Request validation failed
- **authentication_error**: Authentication failed
- **authorization_error**: Insufficient permissions
- **not_found**: Resource not found
- **conflict**: Resource conflict
- **rate_limit_exceeded**: Rate limit exceeded
- **server_error**: Internal server error

## REST API

### OpenAPI Specification

The REST API follows OpenAPI 3.0 specification and is available at:

- **Swagger UI**: `https://api.medical-kg-rev.com/docs`
- **OpenAPI JSON**: `https://api.medical-kg-rev.com/openapi.json`

### Content Types

- **Request**: `application/json`, `application/x-www-form-urlencoded`
- **Response**: `application/json`

### Pagination

List endpoints support pagination:

```bash
GET /v1/documents?page=1&per_page=20&sort=created_at&order=desc
```

Response includes pagination metadata:

```json
{
  "data": [...],
  "meta": {
    "pagination": {
      "page": 1,
      "per_page": 20,
      "total": 1000,
      "total_pages": 50,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

### Filtering and Sorting

#### OData Support

The API supports OData query syntax:

```bash
# Filter by date range
GET /v1/documents?$filter=created_at ge 2024-01-01 and created_at le 2024-12-31

# Select specific fields
GET /v1/documents?$select=id,title,created_at

# Order by field
GET /v1/documents?$orderby=created_at desc

# Top N results
GET /v1/documents?$top=10

# Skip N results
GET /v1/documents?$skip=20
```

#### JSON:API Format

Responses follow JSON:API specification:

```json
{
  "data": {
    "type": "documents",
    "id": "doc_123",
    "attributes": {
      "title": "Clinical Trial Results",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    },
    "relationships": {
      "entities": {
        "data": [
          {"type": "entities", "id": "ent_456"}
        ]
      }
    }
  },
  "included": [
    {
      "type": "entities",
      "id": "ent_456",
      "attributes": {
        "name": "metformin",
        "type": "drug"
      }
    }
  ]
}
```

### Core Endpoints

#### Documents

##### List Documents

```bash
GET /v1/documents
```

**Parameters:**

- `page` (integer): Page number (default: 1)
- `per_page` (integer): Items per page (default: 20, max: 100)
- `sort` (string): Sort field (default: created_at)
- `order` (string): Sort order (asc/desc, default: desc)
- `filter[source]` (string): Filter by source
- `filter[type]` (string): Filter by document type
- `filter[date_from]` (string): Filter by date from (ISO 8601)
- `filter[date_to]` (string): Filter by date to (ISO 8601)

**Response:**

```json
{
  "data": [
    {
      "type": "documents",
      "id": "doc_123",
      "attributes": {
        "title": "Efficacy of Metformin in Type 2 Diabetes",
        "source": "pubmed",
        "type": "research_paper",
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T10:30:00Z",
        "status": "processed",
        "metadata": {
          "authors": ["Smith J", "Doe A"],
          "journal": "Diabetes Care",
          "year": 2024,
          "doi": "10.1234/diabetes.2024.001"
        }
      },
      "relationships": {
        "entities": {
          "data": [
            {"type": "entities", "id": "ent_456"}
          ]
        }
      }
    }
  ],
  "meta": {
    "pagination": {
      "page": 1,
      "per_page": 20,
      "total": 1000,
      "total_pages": 50
    }
  }
}
```

##### Get Document

```bash
GET /v1/documents/{id}
```

**Response:**

```json
{
  "data": {
    "type": "documents",
    "id": "doc_123",
    "attributes": {
      "title": "Efficacy of Metformin in Type 2 Diabetes",
      "content": "Full document content...",
      "source": "pubmed",
      "type": "research_paper",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z",
      "status": "processed",
      "metadata": {
        "authors": ["Smith J", "Doe A"],
        "journal": "Diabetes Care",
        "year": 2024,
        "doi": "10.1234/diabetes.2024.001",
        "abstract": "This study evaluates...",
        "keywords": ["metformin", "diabetes", "efficacy"]
      },
      "extractions": {
        "pico": [
          {
            "population": "Adults with type 2 diabetes",
            "intervention": "Metformin 1000mg twice daily",
            "comparison": "Placebo",
            "outcome": "HbA1c reduction"
          }
        ],
        "effects": [
          {
            "effect_type": "mean_difference",
            "value": -0.5,
            "unit": "%",
            "confidence_interval": [-0.8, -0.2],
            "p_value": 0.001
          }
        ]
      }
    },
    "relationships": {
      "entities": {
        "data": [
          {"type": "entities", "id": "ent_456"},
          {"type": "entities", "id": "ent_789"}
        ]
      },
      "claims": {
        "data": [
          {"type": "claims", "id": "claim_123"}
        ]
      }
    }
  }
}
```

##### Create Document

```bash
POST /v1/documents
```

**Request Body:**

```json
{
  "data": {
    "type": "documents",
    "attributes": {
      "title": "New Research Paper",
      "content": "Document content...",
      "source": "manual_upload",
      "type": "research_paper",
      "metadata": {
        "authors": ["Author Name"],
        "journal": "Journal Name",
        "year": 2024
      }
    }
  }
}
```

**Response:**

```json
{
  "data": {
    "type": "documents",
    "id": "doc_new_123",
    "attributes": {
      "title": "New Research Paper",
      "status": "processing",
      "created_at": "2024-01-15T10:30:00Z"
    }
  }
}
```

#### Entities

##### List Entities

```bash
GET /v1/entities
```

**Parameters:**

- `type` (string): Filter by entity type (drug, disease, gene, protein)
- `name` (string): Filter by entity name
- `page` (integer): Page number
- `per_page` (integer): Items per page

**Response:**

```json
{
  "data": [
    {
      "type": "entities",
      "id": "ent_456",
      "attributes": {
        "name": "metformin",
        "type": "drug",
        "synonyms": ["Glucophage", "Fortamet"],
        "description": "Biguanide antidiabetic medication",
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T10:30:00Z"
      },
      "relationships": {
        "documents": {
          "data": [
            {"type": "documents", "id": "doc_123"}
          ]
        }
      }
    }
  ]
}
```

##### Get Entity

```bash
GET /v1/entities/{id}
```

**Response:**

```json
{
  "data": {
    "type": "entities",
    "id": "ent_456",
    "attributes": {
      "name": "metformin",
      "type": "drug",
      "synonyms": ["Glucophage", "Fortamet"],
      "description": "Biguanide antidiabetic medication",
      "ontology_mappings": {
        "rxnorm": "6809",
        "atc": "A10BA02",
        "unii": "9100L32L2N"
      },
      "properties": {
        "mechanism_of_action": "Inhibits hepatic gluconeogenesis",
        "indications": ["Type 2 diabetes"],
        "contraindications": ["Severe renal impairment"],
        "adverse_events": ["nausea", "diarrhea", "metallic taste"]
      },
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    },
    "relationships": {
      "documents": {
        "data": [
          {"type": "documents", "id": "doc_123"}
        ]
      },
      "relationships": {
        "data": [
          {
            "type": "entity_relationships",
            "id": "rel_789",
            "attributes": {
              "target_entity": "ent_123",
              "relationship_type": "TREATS",
              "evidence_strength": 0.95
            }
          }
        ]
      }
    }
  }
}
```

#### Search

##### Search Documents

```bash
GET /v1/search/documents
```

**Parameters:**

- `q` (string): Search query (required)
- `strategy` (string): Search strategy (bm25, dense, sparse, hybrid)
- `limit` (integer): Maximum results (default: 20, max: 100)
- `offset` (integer): Result offset (default: 0)
- `filters` (object): Additional filters

**Response:**

```json
{
  "data": [
    {
      "type": "search_results",
      "id": "result_123",
      "attributes": {
        "score": 0.95,
        "document": {
          "id": "doc_123",
          "title": "Efficacy of Metformin in Type 2 Diabetes",
          "snippet": "Metformin significantly reduced HbA1c levels...",
          "highlighted_spans": [
            {
              "start": 45,
              "end": 53,
              "text": "Metformin"
            }
          ]
        }
      }
    }
  ],
  "meta": {
    "query": "metformin diabetes",
    "strategy": "hybrid",
    "total_results": 150,
    "search_time_ms": 245
  }
}
```

##### Search Entities

```bash
GET /v1/search/entities
```

**Parameters:**

- `q` (string): Search query (required)
- `type` (string): Entity type filter
- `limit` (integer): Maximum results
- `offset` (integer): Result offset

**Response:**

```json
{
  "data": [
    {
      "type": "search_results",
      "id": "result_456",
      "attributes": {
        "score": 0.98,
        "entity": {
          "id": "ent_456",
          "name": "metformin",
          "type": "drug",
          "description": "Biguanide antidiabetic medication",
          "synonyms": ["Glucophage", "Fortamet"]
        }
      }
    }
  ]
}
```

#### Knowledge Graph

##### Query Knowledge Graph

```bash
POST /v1/kg/query
```

**Request Body:**

```json
{
  "query": "MATCH (d:Drug)-[r:TREATS]->(diabetes:Disease) RETURN d.name, r.evidence_strength ORDER BY r.evidence_strength DESC LIMIT 10",
  "parameters": {}
}
```

**Response:**

```json
{
  "data": [
    {
      "d.name": "metformin",
      "r.evidence_strength": 0.95
    },
    {
      "d.name": "insulin",
      "r.evidence_strength": 0.92
    }
  ],
  "meta": {
    "query_time_ms": 156,
    "result_count": 10
  }
}
```

##### Get Entity Relationships

```bash
GET /v1/kg/entities/{id}/relationships
```

**Parameters:**

- `type` (string): Relationship type filter
- `direction` (string): Relationship direction (incoming, outgoing, both)
- `limit` (integer): Maximum results

**Response:**

```json
{
  "data": [
    {
      "type": "entity_relationships",
      "id": "rel_789",
      "attributes": {
        "source_entity": "ent_456",
        "target_entity": "ent_123",
        "relationship_type": "TREATS",
        "evidence_strength": 0.95,
        "evidence_count": 25,
        "properties": {
          "mechanism": "Inhibits hepatic gluconeogenesis",
          "efficacy": "High",
          "safety": "Good"
        }
      }
    }
  ]
}
```

#### Ingestion

##### Ingest Clinical Trials

```bash
POST /v1/ingest/clinicaltrials
```

**Request Body:**

```json
{
  "data": {
    "type": "ingestion",
    "attributes": {
      "nct_ids": ["NCT04267848", "NCT04345678"],
      "include_full_text": true,
      "extraction_types": ["pico", "effects", "adverse_events"],
      "priority": "normal"
    }
  }
}
```

**Response:**

```json
{
  "data": {
    "type": "ingestion_jobs",
    "id": "job_123",
    "attributes": {
      "status": "queued",
      "total_items": 2,
      "processed_items": 0,
      "failed_items": 0,
      "created_at": "2024-01-15T10:30:00Z",
      "estimated_completion": "2024-01-15T11:00:00Z"
    }
  }
}
```

##### Get Ingestion Job Status

```bash
GET /v1/ingest/jobs/{id}
```

**Response:**

```json
{
  "data": {
    "type": "ingestion_jobs",
    "id": "job_123",
    "attributes": {
      "status": "processing",
      "total_items": 2,
      "processed_items": 1,
      "failed_items": 0,
      "progress_percentage": 50,
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:45:00Z",
      "estimated_completion": "2024-01-15T11:00:00Z",
      "errors": []
    }
  }
}
```

## GraphQL API

### Schema Overview

The GraphQL API provides a flexible interface for querying the knowledge graph:

```graphql
type Query {
  documents(
    first: Int
    after: String
    filter: DocumentFilter
    sort: [DocumentSort!]
  ): DocumentConnection

  entities(
    first: Int
    after: String
    filter: EntityFilter
    sort: [EntitySort!]
  ): EntityConnection

  searchDocuments(
    query: String!
    strategy: SearchStrategy
    first: Int
    after: String
  ): SearchResultConnection

  searchEntities(
    query: String!
    type: EntityType
    first: Int
    after: String
  ): SearchResultConnection

  knowledgeGraph(
    query: String!
    parameters: JSON
  ): KnowledgeGraphResult
}

type Document {
  id: ID!
  title: String!
  content: String
  source: String!
  type: DocumentType!
  status: ProcessingStatus!
  createdAt: DateTime!
  updatedAt: DateTime!
  metadata: JSON
  extractions: [Extraction!]!
  entities: [Entity!]!
  claims: [Claim!]!
}

type Entity {
  id: ID!
  name: String!
  type: EntityType!
  synonyms: [String!]!
  description: String
  ontologyMappings: JSON
  properties: JSON
  createdAt: DateTime!
  updatedAt: DateTime!
  documents: [Document!]!
  relationships: [EntityRelationship!]!
}

type EntityRelationship {
  id: ID!
  sourceEntity: Entity!
  targetEntity: Entity!
  relationshipType: String!
  evidenceStrength: Float!
  evidenceCount: Int!
  properties: JSON
}

type Extraction {
  id: ID!
  type: ExtractionType!
  data: JSON!
  confidence: Float
  sourceSpan: TextSpan
}

type Claim {
  id: ID!
  text: String!
  confidence: Float!
  sourceDocument: Document!
  sourceSpan: TextSpan
  entities: [Entity!]!
}

enum SearchStrategy {
  BM25
  DENSE
  SPARSE
  HYBRID
}

enum EntityType {
  DRUG
  DISEASE
  GENE
  PROTEIN
  ORGAN
  SYMPTOM
  TREATMENT
}

enum DocumentType {
  RESEARCH_PAPER
  CLINICAL_TRIAL
  CASE_REPORT
  CLINICAL_GUIDELINE
  DRUG_LABEL
  SAFETY_REPORT
}

enum ProcessingStatus {
  QUEUED
  PROCESSING
  PROCESSED
  FAILED
}

enum ExtractionType {
  PICO
  EFFECTS
  ADVERSE_EVENTS
  DOSE_REGIMENS
  ELIGIBILITY_CRITERIA
}
```

### Example Queries

#### Search Documents

```graphql
query SearchDocuments($query: String!, $strategy: SearchStrategy!) {
  searchDocuments(query: $query, strategy: $strategy, first: 10) {
    edges {
      node {
        id
        title
        snippet
        score
        document {
          id
          title
          source
          type
          createdAt
        }
      }
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      startCursor
      endCursor
    }
  }
}
```

**Variables:**

```json
{
  "query": "metformin diabetes",
  "strategy": "HYBRID"
}
```

#### Get Document with Entities

```graphql
query GetDocument($id: ID!) {
  document(id: $id) {
    id
    title
    content
    source
    type
    status
    createdAt
    metadata
    extractions {
      id
      type
      data
      confidence
    }
    entities {
      id
      name
      type
      synonyms
      description
    }
    claims {
      id
      text
      confidence
      entities {
        id
        name
        type
      }
    }
  }
}
```

#### Get Entity Relationships

```graphql
query GetEntityRelationships($id: ID!) {
  entity(id: $id) {
    id
    name
    type
    description
    relationships {
      id
      targetEntity {
        id
        name
        type
      }
      relationshipType
      evidenceStrength
      evidenceCount
      properties
    }
  }
}
```

#### Knowledge Graph Query

```graphql
query KnowledgeGraphQuery($query: String!) {
  knowledgeGraph(query: $query) {
    results
    queryTime
    resultCount
  }
}
```

**Variables:**

```json
{
  "query": "MATCH (d:Drug)-[r:TREATS]->(diabetes:Disease) RETURN d.name, r.evidence_strength ORDER BY r.evidence_strength DESC LIMIT 10"
}
```

### Mutations

#### Create Document

```graphql
mutation CreateDocument($input: CreateDocumentInput!) {
  createDocument(input: $input) {
    document {
      id
      title
      status
      createdAt
    }
    errors {
      field
      message
    }
  }
}
```

**Variables:**

```json
{
  "input": {
    "title": "New Research Paper",
    "content": "Document content...",
    "source": "manual_upload",
    "type": "RESEARCH_PAPER",
    "metadata": {
      "authors": ["Author Name"],
      "journal": "Journal Name",
      "year": 2024
    }
  }
}
```

#### Ingest Clinical Trials

```graphql
mutation IngestClinicalTrials($input: IngestClinicalTrialsInput!) {
  ingestClinicalTrials(input: $input) {
    job {
      id
      status
      totalItems
      createdAt
    }
    errors {
      field
      message
    }
  }
}
```

**Variables:**

```json
{
  "input": {
    "nctIds": ["NCT04267848", "NCT04345678"],
    "includeFullText": true,
    "extractionTypes": ["PICO", "EFFECTS", "ADVERSE_EVENTS"],
    "priority": "NORMAL"
  }
}
```

### Subscriptions

#### Real-time Updates

```graphql
subscription DocumentUpdates($documentId: ID!) {
  documentUpdated(documentId: $documentId) {
    id
    status
    progress
    updatedAt
  }
}
```

#### Ingestion Progress

```graphql
subscription IngestionProgress($jobId: ID!) {
  ingestionProgress(jobId: $jobId) {
    id
    status
    progressPercentage
    processedItems
    failedItems
    updatedAt
  }
}
```

## gRPC API

### Service Definitions

#### Document Service

```protobuf
syntax = "proto3";

package medical_kg_rev.v1;

import "google/protobuf/timestamp.proto";
import "google/protobuf/struct.proto";

service DocumentService {
  rpc GetDocument(GetDocumentRequest) returns (GetDocumentResponse);
  rpc ListDocuments(ListDocumentsRequest) returns (stream Document);
  rpc CreateDocument(CreateDocumentRequest) returns (CreateDocumentResponse);
  rpc UpdateDocument(UpdateDocumentRequest) returns (UpdateDocumentResponse);
  rpc DeleteDocument(DeleteDocumentRequest) returns (DeleteDocumentResponse);
}

message GetDocumentRequest {
  string id = 1;
}

message GetDocumentResponse {
  Document document = 1;
}

message ListDocumentsRequest {
  int32 page_size = 1;
  string page_token = 2;
  string filter = 3;
  string order_by = 4;
}

message CreateDocumentRequest {
  Document document = 1;
}

message CreateDocumentResponse {
  Document document = 1;
}

message Document {
  string id = 1;
  string title = 2;
  string content = 3;
  string source = 4;
  string type = 5;
  string status = 6;
  google.protobuf.Timestamp created_at = 7;
  google.protobuf.Timestamp updated_at = 8;
  google.protobuf.Struct metadata = 9;
  repeated Extraction extractions = 10;
  repeated Entity entities = 11;
  repeated Claim claims = 12;
}

message Extraction {
  string id = 1;
  string type = 2;
  google.protobuf.Struct data = 3;
  double confidence = 4;
  TextSpan source_span = 5;
}

message Entity {
  string id = 1;
  string name = 2;
  string type = 3;
  repeated string synonyms = 4;
  string description = 5;
  google.protobuf.Struct ontology_mappings = 6;
  google.protobuf.Struct properties = 7;
}

message Claim {
  string id = 1;
  string text = 2;
  double confidence = 3;
  string source_document_id = 4;
  TextSpan source_span = 5;
  repeated string entity_ids = 6;
}

message TextSpan {
  int32 start = 1;
  int32 end = 2;
  string text = 3;
}
```

#### Search Service

```protobuf
service SearchService {
  rpc SearchDocuments(SearchDocumentsRequest) returns (SearchDocumentsResponse);
  rpc SearchEntities(SearchEntitiesRequest) returns (SearchEntitiesResponse);
  rpc SearchHybrid(SearchHybridRequest) returns (SearchHybridResponse);
}

message SearchDocumentsRequest {
  string query = 1;
  string strategy = 2;
  int32 limit = 3;
  int32 offset = 4;
  google.protobuf.Struct filters = 5;
}

message SearchDocumentsResponse {
  repeated SearchResult results = 1;
  int32 total_results = 2;
  int64 search_time_ms = 3;
}

message SearchResult {
  string id = 1;
  double score = 2;
  Document document = 3;
  string snippet = 4;
  repeated TextSpan highlighted_spans = 5;
}
```

#### Knowledge Graph Service

```protobuf
service KnowledgeGraphService {
  rpc Query(QueryRequest) returns (QueryResponse);
  rpc GetEntityRelationships(GetEntityRelationshipsRequest) returns (GetEntityRelationshipsResponse);
  rpc GetEntityNeighbors(GetEntityNeighborsRequest) returns (GetEntityNeighborsResponse);
}

message QueryRequest {
  string query = 1;
  google.protobuf.Struct parameters = 2;
}

message QueryResponse {
  repeated google.protobuf.Struct results = 1;
  int64 query_time_ms = 2;
  int32 result_count = 3;
}

message GetEntityRelationshipsRequest {
  string entity_id = 1;
  string relationship_type = 2;
  string direction = 3;
  int32 limit = 4;
}

message GetEntityRelationshipsResponse {
  repeated EntityRelationship relationships = 1;
}

message EntityRelationship {
  string id = 1;
  string source_entity_id = 2;
  string target_entity_id = 3;
  string relationship_type = 4;
  double evidence_strength = 5;
  int32 evidence_count = 6;
  google.protobuf.Struct properties = 7;
}
```

### Usage Examples

#### Python Client

```python
import grpc
from medical_kg_rev.v1 import document_service_pb2
from medical_kg_rev.v1 import document_service_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:50051')

# Create stub
stub = document_service_pb2_grpc.DocumentServiceStub(channel)

# Get document
request = document_service_pb2.GetDocumentRequest(id='doc_123')
response = stub.GetDocument(request)
print(f"Document: {response.document.title}")

# List documents
request = document_service_pb2.ListDocumentsRequest(
    page_size=10,
    filter="source='pubmed'",
    order_by="created_at desc"
)
for document in stub.ListDocuments(request):
    print(f"Document: {document.title}")
```

#### Go Client

```go
package main

import (
    "context"
    "log"

    "google.golang.org/grpc"
    pb "medical_kg_rev/v1"
)

func main() {
    // Create connection
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()

    // Create client
    client := pb.NewDocumentServiceClient(conn)

    // Get document
    req := &pb.GetDocumentRequest{Id: "doc_123"}
    resp, err := client.GetDocument(context.Background(), req)
    if err != nil {
        log.Fatalf("Failed to get document: %v", err)
    }

    log.Printf("Document: %s", resp.Document.Title)
}
```

## WebSocket API

### Connection

Connect to the WebSocket endpoint:

```javascript
const ws = new WebSocket('wss://api.medical-kg-rev.com/ws/v1');
```

### Authentication

Send authentication message:

```javascript
ws.send(JSON.stringify({
  type: 'auth',
  token: 'YOUR_JWT_TOKEN'
}));
```

### Message Format

All messages follow this format:

```json
{
  "type": "message_type",
  "id": "message_id",
  "data": {
    // Message-specific data
  }
}
```

### Message Types

#### Search Request

```json
{
  "type": "search",
  "id": "search_123",
  "data": {
    "query": "metformin diabetes",
    "strategy": "hybrid",
    "limit": 10
  }
}
```

#### Search Response

```json
{
  "type": "search_result",
  "id": "search_123",
  "data": {
    "results": [
      {
        "id": "result_1",
        "score": 0.95,
        "document": {
          "id": "doc_123",
          "title": "Efficacy of Metformin",
          "snippet": "Metformin significantly reduced..."
        }
      }
    ],
    "total_results": 150,
    "search_time_ms": 245
  }
}
```

#### Real-time Updates

```json
{
  "type": "document_update",
  "id": "update_456",
  "data": {
    "document_id": "doc_123",
    "status": "processing",
    "progress": 75,
    "updated_at": "2024-01-15T10:45:00Z"
  }
}
```

### JavaScript Client

```javascript
class MedicalKGClient {
  constructor(url, token) {
    this.url = url;
    this.token = token;
    this.ws = null;
    this.messageHandlers = new Map();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      // Authenticate
      this.send({
        type: 'auth',
        token: this.token
      });
    };

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };

    this.ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  send(message) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  handleMessage(message) {
    const handler = this.messageHandlers.get(message.type);
    if (handler) {
      handler(message.data);
    }
  }

  onMessageType(type, handler) {
    this.messageHandlers.set(type, handler);
  }

  search(query, strategy = 'hybrid', limit = 10) {
    const messageId = `search_${Date.now()}`;
    this.send({
      type: 'search',
      id: messageId,
      data: {
        query,
        strategy,
        limit
      }
    });
    return messageId;
  }

  subscribeToDocument(documentId) {
    this.send({
      type: 'subscribe',
      data: {
        resource: 'document',
        id: documentId
      }
    });
  }
}

// Usage
const client = new MedicalKGClient('wss://api.medical-kg-rev.com/ws/v1', 'YOUR_JWT_TOKEN');

client.onMessageType('search_result', (data) => {
  console.log('Search results:', data.results);
});

client.onMessageType('document_update', (data) => {
  console.log('Document updated:', data);
});

client.connect();
client.search('metformin diabetes');
```

## SDKs and Libraries

### Python SDK

#### Installation

```bash
pip install medical-kg-rev-sdk
```

#### Usage

```python
from medical_kg_rev import Client

# Initialize client
client = Client(
    api_key='YOUR_API_KEY',
    base_url='https://api.medical-kg-rev.com'
)

# Search documents
results = client.search_documents(
    query='metformin diabetes',
    strategy='hybrid',
    limit=10
)

# Get document
document = client.get_document('doc_123')

# Get entity
entity = client.get_entity('ent_456')

# Query knowledge graph
kg_results = client.query_knowledge_graph(
    query='MATCH (d:Drug)-[r:TREATS]->(diabetes:Disease) RETURN d.name, r.evidence_strength'
)

# Ingest clinical trials
job = client.ingest_clinical_trials(
    nct_ids=['NCT04267848', 'NCT04345678'],
    include_full_text=True,
    extraction_types=['pico', 'effects', 'adverse_events']
)
```

### JavaScript SDK

#### Installation

```bash
npm install @medical-kg-rev/sdk
```

#### Usage

```javascript
import { Client } from '@medical-kg-rev/sdk';

// Initialize client
const client = new Client({
  apiKey: 'YOUR_API_KEY',
  baseUrl: 'https://api.medical-kg-rev.com'
});

// Search documents
const results = await client.searchDocuments({
  query: 'metformin diabetes',
  strategy: 'hybrid',
  limit: 10
});

// Get document
const document = await client.getDocument('doc_123');

// Get entity
const entity = await client.getEntity('ent_456');

// Query knowledge graph
const kgResults = await client.queryKnowledgeGraph({
  query: 'MATCH (d:Drug)-[r:TREATS]->(diabetes:Disease) RETURN d.name, r.evidence_strength'
});

// Ingest clinical trials
const job = await client.ingestClinicalTrials({
  nctIds: ['NCT04267848', 'NCT04345678'],
  includeFullText: true,
  extractionTypes: ['pico', 'effects', 'adverse_events']
});
```

### R SDK

#### Installation

```r
install.packages("medicalKGR")
```

#### Usage

```r
library(medicalKGR)

# Initialize client
client <- MedicalKGClient$new(
  api_key = "YOUR_API_KEY",
  base_url = "https://api.medical-kg-rev.com"
)

# Search documents
results <- client$search_documents(
  query = "metformin diabetes",
  strategy = "hybrid",
  limit = 10
)

# Get document
document <- client$get_document("doc_123")

# Get entity
entity <- client$get_entity("ent_456")

# Query knowledge graph
kg_results <- client$query_knowledge_graph(
  query = "MATCH (d:Drug)-[r:TREATS]->(diabetes:Disease) RETURN d.name, r.evidence_strength"
)
```

## Examples

### Complete Workflow Example

#### 1. Search for Documents

```bash
curl -X GET "https://api.medical-kg-rev.com/v1/search/documents?q=metformin+diabetes&strategy=hybrid&limit=5" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### 2. Get Document Details

```bash
curl -X GET "https://api.medical-kg-rev.com/v1/documents/doc_123" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### 3. Extract Entities from Document

```bash
curl -X GET "https://api.medical-kg-rev.com/v1/documents/doc_123/entities" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### 4. Query Knowledge Graph

```bash
curl -X POST "https://api.medical-kg-rev.com/v1/kg/query" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MATCH (d:Drug {name: \"metformin\"})-[r:TREATS]->(diabetes:Disease) RETURN d.name, r.evidence_strength, diabetes.name"
  }'
```

#### 5. Ingest New Data

```bash
curl -X POST "https://api.medical-kg-rev.com/v1/ingest/clinicaltrials" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "type": "ingestion",
      "attributes": {
        "nct_ids": ["NCT04267848"],
        "include_full_text": true,
        "extraction_types": ["pico", "effects"]
      }
    }
  }'
```

### Python Complete Example

```python
import requests
import json
import time

class MedicalKGClient:
    def __init__(self, api_key, base_url='https://api.medical-kg-rev.com'):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })

    def search_documents(self, query, strategy='hybrid', limit=10):
        response = self.session.get(
            f'{self.base_url}/v1/search/documents',
            params={
                'q': query,
                'strategy': strategy,
                'limit': limit
            }
        )
        response.raise_for_status()
        return response.json()

    def get_document(self, document_id):
        response = self.session.get(f'{self.base_url}/v1/documents/{document_id}')
        response.raise_for_status()
        return response.json()

    def query_knowledge_graph(self, query):
        response = self.session.post(
            f'{self.base_url}/v1/kg/query',
            json={'query': query}
        )
        response.raise_for_status()
        return response.json()

    def ingest_clinical_trials(self, nct_ids, include_full_text=True, extraction_types=None):
        if extraction_types is None:
            extraction_types = ['pico', 'effects', 'adverse_events']

        payload = {
            'data': {
                'type': 'ingestion',
                'attributes': {
                    'nct_ids': nct_ids,
                    'include_full_text': include_full_text,
                    'extraction_types': extraction_types
                }
            }
        }

        response = self.session.post(
            f'{self.base_url}/v1/ingest/clinicaltrials',
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def get_ingestion_job_status(self, job_id):
        response = self.session.get(f'{self.base_url}/v1/ingest/jobs/{job_id}')
        response.raise_for_status()
        return response.json()

# Usage example
def main():
    # Initialize client
    client = MedicalKGClient('YOUR_API_KEY')

    # Search for documents
    print("Searching for documents...")
    search_results = client.search_documents('metformin diabetes', limit=3)

    for result in search_results['data']:
        document_id = result['document']['id']
        print(f"Found document: {result['document']['title']}")

        # Get document details
        document = client.get_document(document_id)
        print(f"Document source: {document['data']['attributes']['source']}")
        print(f"Document type: {document['data']['attributes']['type']}")

        # Extract entities
        entities = document['data']['relationships']['entities']['data']
        print(f"Entities found: {len(entities)}")

        # Query knowledge graph for relationships
        kg_query = f"MATCH (d:Drug)-[r:TREATS]->(diabetes:Disease) WHERE d.name CONTAINS 'metformin' RETURN d.name, r.evidence_strength LIMIT 5"
        kg_results = client.query_knowledge_graph(kg_query)
        print(f"Knowledge graph results: {len(kg_results['data'])} relationships found")

    # Ingest new clinical trial data
    print("\nIngesting clinical trial data...")
    ingestion_job = client.ingest_clinical_trials(['NCT04267848'])
    job_id = ingestion_job['data']['id']
    print(f"Ingestion job started: {job_id}")

    # Monitor ingestion progress
    while True:
        job_status = client.get_ingestion_job_status(job_id)
        status = job_status['data']['attributes']['status']
        progress = job_status['data']['attributes']['progress_percentage']

        print(f"Job status: {status} ({progress}% complete)")

        if status in ['completed', 'failed']:
            break

        time.sleep(10)

    print("Ingestion completed!")

if __name__ == '__main__':
    main()
```

### GraphQL Complete Example

```javascript
import { gql, GraphQLClient } from 'graphql-request';

const client = new GraphQLClient('https://api.medical-kg-rev.com/graphql', {
  headers: {
    authorization: 'Bearer YOUR_API_KEY',
  },
});

// Search documents query
const SEARCH_DOCUMENTS = gql`
  query SearchDocuments($query: String!, $strategy: SearchStrategy!) {
    searchDocuments(query: $query, strategy: $strategy, first: 5) {
      edges {
        node {
          id
          title
          snippet
          score
          document {
            id
            title
            source
            type
            createdAt
            entities {
              id
              name
              type
            }
          }
        }
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
      }
    }
  }
`;

// Get document with relationships
const GET_DOCUMENT = gql`
  query GetDocument($id: ID!) {
    document(id: $id) {
      id
      title
      content
      source
      type
      status
      createdAt
      extractions {
        id
        type
        data
        confidence
      }
      entities {
        id
        name
        type
        synonyms
        description
        relationships {
          id
          targetEntity {
            id
            name
            type
          }
          relationshipType
          evidenceStrength
        }
      }
    }
  }
`;

// Knowledge graph query
const KNOWLEDGE_GRAPH_QUERY = gql`
  query KnowledgeGraphQuery($query: String!) {
    knowledgeGraph(query: $query) {
      results
      queryTime
      resultCount
    }
  }
`;

async function main() {
  try {
    // Search for documents
    console.log('Searching for documents...');
    const searchResults = await client.request(SEARCH_DOCUMENTS, {
      query: 'metformin diabetes',
      strategy: 'HYBRID'
    });

    console.log(`Found ${searchResults.searchDocuments.edges.length} documents`);

    // Get details for first document
    if (searchResults.searchDocuments.edges.length > 0) {
      const firstDocument = searchResults.searchDocuments.edges[0].node.document;
      console.log(`\nGetting details for: ${firstDocument.title}`);

      const documentDetails = await client.request(GET_DOCUMENT, {
        id: firstDocument.id
      });

      console.log(`Document source: ${documentDetails.document.source}`);
      console.log(`Document type: ${documentDetails.document.type}`);
      console.log(`Entities found: ${documentDetails.document.entities.length}`);

      // Query knowledge graph
      console.log('\nQuerying knowledge graph...');
      const kgResults = await client.request(KNOWLEDGE_GRAPH_QUERY, {
        query: 'MATCH (d:Drug)-[r:TREATS]->(diabetes:Disease) WHERE d.name CONTAINS "metformin" RETURN d.name, r.evidence_strength LIMIT 5'
      });

      console.log(`Knowledge graph results: ${kgResults.knowledgeGraph.resultCount} relationships found`);
      console.log(`Query time: ${kgResults.knowledgeGraph.queryTime}ms`);
    }

  } catch (error) {
    console.error('Error:', error);
  }
}

main();
```

This comprehensive API reference provides all the information needed to effectively use the Medical_KG_rev system's APIs. The examples demonstrate common workflows and best practices for integration.
