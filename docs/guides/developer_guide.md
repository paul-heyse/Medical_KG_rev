# Developer Guide

This comprehensive developer guide provides technical documentation for developers working with the Medical_KG_rev system, including architecture overview, development setup, coding standards, and integration patterns.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Development Environment Setup](#development-environment-setup)
3. [Codebase Structure](#codebase-structure)
4. [Development Guidelines](#development-guidelines)
5. [API Development](#api-development)
6. [Database Design](#database-design)
7. [Testing Strategy](#testing-strategy)
8. [Deployment and DevOps](#deployment-and-devops)
9. [Performance Optimization](#performance-optimization)
10. [Security Considerations](#security-considerations)
11. [Troubleshooting](#troubleshooting)

## System Architecture

### Overview

The Medical_KG_rev system is built on a microservices architecture with a multi-protocol API gateway:

```mermaid
graph TD
    A[Client Applications] --> B[API Gateway]
    B --> C[Authentication Service]
    B --> D[Document Service]
    B --> E[Search Service]
    B --> F[Knowledge Graph Service]
    B --> G[Analytics Service]

    D --> H[Document Processing Pipeline]
    E --> I[Search Engine]
    F --> J[Graph Database]
    G --> K[Analytics Engine]

    H --> L[External Data Sources]
    I --> M[Vector Database]
    J --> N[Neo4j Database]
    K --> O[Analytics Database]
```

### Core Components

#### API Gateway

- **Multi-protocol support**: REST, GraphQL, gRPC, WebSocket
- **Authentication and authorization**: JWT-based security
- **Rate limiting**: Per-user and per-IP limits
- **Request routing**: Intelligent routing to microservices

#### Document Processing Pipeline

- **Ingestion**: Data from various biomedical sources
- **Processing**: Automated extraction and structuring
- **Storage**: Multi-database storage strategy
- **Indexing**: Search index creation and maintenance

#### Knowledge Graph Engine

- **Graph database**: Neo4j for relationship storage
- **Entity recognition**: Automated entity extraction
- **Relationship discovery**: AI-powered relationship identification
- **Graph analytics**: Centrality, clustering, and path analysis

#### Search Engine

- **Multi-strategy search**: BM25, dense vectors, sparse vectors
- **Hybrid ranking**: Reciprocal Rank Fusion (RRF)
- **Real-time indexing**: Near real-time search updates
- **Performance optimization**: Sub-500ms P95 latency

### Technology Stack

#### Backend Services

- **Python 3.11+**: Primary development language
- **FastAPI**: REST API framework
- **Strawberry GraphQL**: GraphQL implementation
- **gRPC**: High-performance inter-service communication
- **Pydantic v2**: Data validation and serialization

#### Databases

- **PostgreSQL**: Relational data storage
- **Neo4j**: Graph database for relationships
- **Redis**: Caching and session storage
- **Qdrant**: Vector database for embeddings

#### Infrastructure

- **Kubernetes**: Container orchestration
- **Docker**: Containerization
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards
- **Jaeger**: Distributed tracing

## Development Environment Setup

### Prerequisites

- **Python 3.11+**: Latest Python version
- **Node.js 18+**: For frontend development
- **Docker**: Container runtime
- **Docker Compose**: Local development environment
- **Git**: Version control
- **VS Code**: Recommended IDE

### Local Development Setup

#### 1. Clone Repository

```bash
git clone https://github.com/your-org/Medical_KG_rev.git
cd Medical_KG_rev
```

#### 2. Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

#### 4. Database Setup

```bash
# Start local databases
docker-compose up -d postgresql redis neo4j qdrant

# Run migrations
python -m Medical_KG_rev.storage.migrate
```

#### 5. Start Development Services

```bash
# Start all services
docker-compose up -d

# Start API gateway
python -m Medical_KG_rev.gateway.main

# Start document processing
python -m Medical_KG_rev.services.document.main

# Start search service
python -m Medical_KG_rev.services.search.main
```

### IDE Configuration

#### VS Code Settings

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  }
}
```

#### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Codebase Structure

### Directory Organization

```
Medical_KG_rev/
├── src/Medical_KG_rev/          # Main source code
│   ├── gateway/                 # API gateway
│   ├── services/                # Microservices
│   ├── adapters/                # Data source adapters
│   ├── orchestration/          # Workflow orchestration
│   ├── kg/                      # Knowledge graph
│   ├── models/                  # Data models
│   ├── storage/                 # Database layer
│   ├── validation/              # Data validation
│   ├── auth/                    # Authentication
│   ├── observability/          # Monitoring and logging
│   └── utils/                   # Utilities
├── tests/                       # Test suite
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
├── docker/                      # Docker configurations
├── kubernetes/                  # Kubernetes manifests
└── openspec/                    # OpenSpec specifications
```

### Key Modules

#### Gateway Module

```python
# src/Medical_KG_rev/gateway/
├── main.py                      # FastAPI application
├── middleware/                   # Custom middleware
├── rest/                        # REST API handlers
├── graphql/                     # GraphQL resolvers
├── grpc/                        # gRPC services
└── websocket/                   # WebSocket handlers
```

#### Services Module

```python
# src/Medical_KG_rev/services/
├── document/                    # Document processing
├── search/                      # Search functionality
├── extraction/                  # Information extraction
├── embedding/                   # Vector embeddings
├── analytics/                   # Analytics engine
└── notification/                # Notification service
```

#### Models Module

```python
# src/Medical_KG_rev/models/
├── base.py                      # Base model classes
├── document.py                  # Document models
├── entity.py                    # Entity models
├── claim.py                     # Claim models
├── extraction.py                # Extraction models
└── user.py                      # User models
```

## Development Guidelines

### Coding Standards

#### Python Style Guide

- **PEP 8**: Follow Python style guidelines
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

#### Code Organization

```python
# File structure example
"""
Module docstring describing purpose and usage.
"""

# Standard library imports
import os
import sys
from typing import List, Dict, Optional

# Third-party imports
import pandas as pd
from fastapi import FastAPI, HTTPException

# Local imports
from Medical_KG_rev.models import Document
from Medical_KG_rev.utils import logger

# Constants
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Classes
class DocumentService:
    """Service for document operations."""

    def __init__(self, db_client):
        self.db_client = db_client

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        # Implementation
        pass

# Functions
async def process_document(doc: Document) -> bool:
    """Process a document."""
    # Implementation
    pass

# Main execution
if __name__ == "__main__":
    # Main execution code
    pass
```

#### Type Hints

```python
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel

# Use modern type hints
def process_documents(
    documents: List[Document],
    options: Dict[str, Any],
    timeout: Optional[int] = None
) -> Union[bool, str]:
    """Process documents with type hints."""
    pass

# Use Pydantic models for data validation
class DocumentCreate(BaseModel):
    title: str
    content: str
    source: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

#### Error Handling

```python
import logging
from fastapi import HTTPException
from Medical_KG_rev.exceptions import (
    DocumentNotFoundError,
    ValidationError,
    ProcessingError
)

logger = logging.getLogger(__name__)

async def get_document(doc_id: str) -> Document:
    """Get document with proper error handling."""
    try:
        document = await document_service.get_by_id(doc_id)
        if not document:
            raise DocumentNotFoundError(f"Document {doc_id} not found")
        return document

    except ValidationError as e:
        logger.error(f"Validation error for document {doc_id}: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except ProcessingError as e:
        logger.error(f"Processing error for document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

    except Exception as e:
        logger.error(f"Unexpected error for document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Documentation Standards

#### Docstring Format

```python
def extract_entities(
    text: str,
    model_name: str = "bert-base",
    confidence_threshold: float = 0.8
) -> List[Entity]:
    """
    Extract entities from text using specified model.

    Args:
        text: Input text for entity extraction
        model_name: Name of the NER model to use
        confidence_threshold: Minimum confidence score for entities

    Returns:
        List of extracted entities with confidence scores

    Raises:
        ModelNotFoundError: If specified model is not available
        ValidationError: If input text is invalid

    Example:
        >>> entities = extract_entities(
        ...     "Metformin is used to treat diabetes",
        ...     model_name="bert-base",
        ...     confidence_threshold=0.9
        ... )
        >>> print(entities[0].name)
        metformin
    """
    pass
```

#### API Documentation

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

app = FastAPI(
    title="Medical KG Rev API",
    description="Biomedical knowledge integration system",
    version="1.0.0"
)

class DocumentResponse(BaseModel):
    """Document response model."""

    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Document source")
    created_at: datetime = Field(..., description="Creation timestamp")

@app.get(
    "/documents/{doc_id}",
    response_model=DocumentResponse,
    summary="Get document by ID",
    description="Retrieve a specific document by its unique identifier",
    responses={
        200: {"description": "Document retrieved successfully"},
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_document(
    doc_id: str = Field(..., description="Document identifier"),
    current_user: User = Depends(get_current_user)
) -> DocumentResponse:
    """
    Get document by ID.

    Retrieve a specific document with all its metadata and content.
    Requires authentication and appropriate permissions.
    """
    pass
```

## API Development

### REST API Development

#### FastAPI Best Practices

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Medical KG Rev API",
    description="Biomedical knowledge integration system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.medical-kg-rev.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.medical-kg-rev.com", "*.medical-kg-rev.com"]
)

# Request/Response models
class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, max_length=100)
    metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    source: str
    created_at: datetime
    updated_at: datetime

# Dependency injection
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current authenticated user."""
    # Implementation
    pass

# API endpoints
@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create document",
    description="Create a new document in the system"
)
async def create_document(
    document: DocumentCreate,
    current_user: User = Depends(get_current_user)
) -> DocumentResponse:
    """Create a new document."""
    try:
        # Validate input
        if not document.title.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Title cannot be empty"
            )

        # Create document
        new_document = await document_service.create(
            title=document.title,
            content=document.content,
            source=document.source,
            metadata=document.metadata,
            user_id=current_user.id
        )

        logger.info(f"Document created: {new_document.id}")
        return DocumentResponse.from_orm(new_document)

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get(
    "/documents/{doc_id}",
    response_model=DocumentResponse,
    summary="Get document",
    description="Retrieve a document by ID"
)
async def get_document(
    doc_id: str,
    current_user: User = Depends(get_current_user)
) -> DocumentResponse:
    """Get document by ID."""
    document = await document_service.get_by_id(doc_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return DocumentResponse.from_orm(document)
```

#### Error Handling

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors()
            }
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "http_error",
                "message": exc.detail
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Internal server error"
            }
        }
    )
```

### GraphQL API Development

#### Strawberry GraphQL Setup

```python
import strawberry
from strawberry.fastapi import GraphQLRouter
from typing import List, Optional
from datetime import datetime

@strawberry.type
class Document:
    id: str
    title: str
    content: str
    source: str
    created_at: datetime
    updated_at: datetime

@strawberry.type
class Entity:
    id: str
    name: str
    type: str
    synonyms: List[str]
    description: Optional[str]

@strawberry.input
class DocumentCreateInput:
    title: str
    content: str
    source: str
    metadata: Optional[strawberry.scalars.JSON] = None

@strawberry.type
class Query:
    @strawberry.field
    async def document(self, id: str) -> Optional[Document]:
        """Get document by ID."""
        return await document_service.get_by_id(id)

    @strawberry.field
    async def documents(
        self,
        first: int = 20,
        after: Optional[str] = None,
        filter: Optional[str] = None
    ) -> List[Document]:
        """Get list of documents."""
        return await document_service.list(
            limit=first,
            cursor=after,
            filter=filter
        )

    @strawberry.field
    async def search_documents(
        self,
        query: str,
        strategy: str = "hybrid",
        limit: int = 20
    ) -> List[Document]:
        """Search documents."""
        return await search_service.search_documents(
            query=query,
            strategy=strategy,
            limit=limit
        )

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_document(
        self,
        input: DocumentCreateInput
    ) -> Document:
        """Create a new document."""
        return await document_service.create(
            title=input.title,
            content=input.content,
            source=input.source,
            metadata=input.metadata
        )

# Create schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Create GraphQL router
graphql_app = GraphQLRouter(schema)

# Add to FastAPI app
app.include_router(graphql_app, prefix="/graphql")
```

### gRPC API Development

#### Protocol Buffer Definition

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
}
```

#### gRPC Server Implementation

```python
import grpc
from concurrent import futures
from medical_kg_rev.v1 import document_service_pb2
from medical_kg_rev.v1 import document_service_pb2_grpc
from Medical_KG_rev.services.document import DocumentService
import logging

logger = logging.getLogger(__name__)

class DocumentServicer(document_service_pb2_grpc.DocumentServiceServicer):
    """gRPC service implementation for document operations."""

    def __init__(self):
        self.document_service = DocumentService()

    async def GetDocument(
        self,
        request: document_service_pb2.GetDocumentRequest,
        context: grpc.aio.ServicerContext
    ) -> document_service_pb2.GetDocumentResponse:
        """Get document by ID."""
        try:
            document = await self.document_service.get_by_id(request.id)
            if not document:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Document not found")
                return document_service_pb2.GetDocumentResponse()

            # Convert to protobuf message
            pb_document = document_service_pb2.Document(
                id=document.id,
                title=document.title,
                content=document.content,
                source=document.source,
                type=document.type,
                status=document.status,
                created_at=document.created_at,
                updated_at=document.updated_at,
                metadata=document.metadata
            )

            return document_service_pb2.GetDocumentResponse(document=pb_document)

        except Exception as e:
            logger.error(f"Error getting document {request.id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return document_service_pb2.GetDocumentResponse()

    async def ListDocuments(
        self,
        request: document_service_pb2.ListDocumentsRequest,
        context: grpc.aio.ServicerContext
    ):
        """List documents with streaming response."""
        try:
            documents = await self.document_service.list(
                page_size=request.page_size,
                page_token=request.page_token,
                filter=request.filter,
                order_by=request.order_by
            )

            for document in documents:
                pb_document = document_service_pb2.Document(
                    id=document.id,
                    title=document.title,
                    content=document.content,
                    source=document.source,
                    type=document.type,
                    status=document.status,
                    created_at=document.created_at,
                    updated_at=document.updated_at,
                    metadata=document.metadata
                )
                yield pb_document

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")

async def serve():
    """Start gRPC server."""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

    # Add servicer
    document_service_pb2_grpc.add_DocumentServiceServicer_to_server(
        DocumentServicer(), server
    )

    # Add port
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    import asyncio
    asyncio.run(serve())
```

## Database Design

### Database Architecture

#### Multi-Database Strategy

```python
# Database configuration
DATABASE_CONFIG = {
    "postgresql": {
        "host": "postgresql-primary",
        "port": 5432,
        "database": "medical_kg_rev",
        "user": "postgres",
        "password": "password",
        "pool_size": 20,
        "max_overflow": 30
    },
    "neo4j": {
        "uri": "bolt://neo4j-primary:7687",
        "user": "neo4j",
        "password": "password",
        "max_connections": 50
    },
    "redis": {
        "host": "redis-primary",
        "port": 6379,
        "password": "password",
        "db": 0,
        "max_connections": 100
    },
    "qdrant": {
        "host": "qdrant-primary",
        "port": 6333,
        "api_key": "api_key",
        "timeout": 30
    }
}
```

#### Database Models

```python
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Document(Base):
    """Document model for PostgreSQL."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)
    status = Column(String(20), default="processing")
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    extractions = relationship("Extraction", back_populates="document")
    entities = relationship("DocumentEntity", back_populates="document")

class Entity(Base):
    """Entity model for PostgreSQL."""

    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    type = Column(String(50), nullable=False)
    synonyms = Column(JSON)
    description = Column(Text)
    ontology_mappings = Column(JSON)
    properties = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Extraction(Base):
    """Extraction model for PostgreSQL."""

    __tablename__ = "extractions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    type = Column(String(50), nullable=False)
    data = Column(JSON, nullable=False)
    confidence = Column(Float)
    source_span_start = Column(Integer)
    source_span_end = Column(Integer)
    source_span_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="extractions")
```

#### Database Migrations

```python
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from Medical_KG_rev.storage.database import get_database_url

def run_migrations():
    """Run database migrations."""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", get_database_url())
    command.upgrade(alembic_cfg, "head")

def create_migration(message: str):
    """Create a new migration."""
    alembic_cfg = Config("alembic.ini")
    command.revision(alembic_cfg, message=message, autogenerate=True)
```

### Graph Database Design

#### Neo4j Schema

```cypher
// Node labels
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE;

// Relationship types
// Document relationships
CREATE (d:Document)-[:MENTIONS]->(e:Entity);
CREATE (d:Document)-[:CONTAINS]->(c:Claim);

// Entity relationships
CREATE (e1:Entity)-[:TREATS]->(e2:Entity);
CREATE (e1:Entity)-[:CAUSES]->(e2:Entity);
CREATE (e1:Entity)-[:PREVENTS]->(e2:Entity);
CREATE (e1:Entity)-[:INTERACTS_WITH]->(e2:Entity);

// Claim relationships
CREATE (c:Claim)-[:INVOLVES]->(e:Entity);
CREATE (c:Claim)-[:SUPPORTS]->(c2:Claim);
CREATE (c:Claim)-[:CONTRADICTS]->(c2:Claim);
```

#### Graph Operations

```python
from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class GraphDatabase:
    """Neo4j graph database operations."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close database connection."""
        self.driver.close()

    async def create_document(self, document: Dict[str, Any]) -> str:
        """Create document node in graph."""
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (d:Document {
                    id: $id,
                    title: $title,
                    source: $source,
                    type: $type,
                    created_at: $created_at
                })
                RETURN d.id as id
                """,
                **document
            )
            return result.single()["id"]

    async def create_entity(self, entity: Dict[str, Any]) -> str:
        """Create entity node in graph."""
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (e:Entity {
                    id: $id,
                    name: $name,
                    type: $type,
                    synonyms: $synonyms,
                    description: $description
                })
                RETURN e.id as id
                """,
                **entity
            )
            return result.single()["id"]

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> str:
        """Create relationship between entities."""
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (source), (target)
                WHERE source.id = $source_id AND target.id = $target_id
                CREATE (source)-[r:{relationship_type} $properties]->(target)
                RETURN r
                """,
                source_id=source_id,
                target_id=target_id,
                properties=properties or {}
            )
            return result.single()["r"]

    async def query_relationships(
        self,
        entity_id: str,
        relationship_type: str = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Query entity relationships."""
        with self.driver.session() as session:
            if direction == "incoming":
                query = """
                MATCH (target)-[r]->(source)
                WHERE source.id = $entity_id
                """
            elif direction == "outgoing":
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.id = $entity_id
                """
            else:
                query = """
                MATCH (source)-[r]-(target)
                WHERE source.id = $entity_id
                """

            if relationship_type:
                query += f" AND type(r) = '{relationship_type}'"

            query += " RETURN r, target"

            result = session.run(query, entity_id=entity_id)
            return [record.data() for record in result]

    async def execute_cypher(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute custom Cypher query."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
```

## Testing Strategy

### Test Structure

#### Test Organization

```
tests/
├── unit/                          # Unit tests
│   ├── test_models.py
│   ├── test_services.py
│   └── test_utils.py
├── integration/                   # Integration tests
│   ├── test_api.py
│   ├── test_database.py
│   └── test_external_apis.py
├── contract/                      # Contract tests
│   ├── test_rest_api.py
│   ├── test_graphql_api.py
│   └── test_grpc_api.py
├── performance/                   # Performance tests
│   ├── test_search_performance.py
│   └── test_ingestion_performance.py
├── fixtures/                      # Test fixtures
│   ├── sample_documents.json
│   ├── sample_entities.json
│   └── mock_responses.json
└── conftest.py                    # Pytest configuration
```

#### Unit Testing

```python
import pytest
from unittest.mock import Mock, patch
from Medical_KG_rev.models import Document, Entity
from Medical_KG_rev.services.document import DocumentService
from Medical_KG_rev.exceptions import DocumentNotFoundError

class TestDocumentService:
    """Test cases for DocumentService."""

    @pytest.fixture
    def document_service(self):
        """Create DocumentService instance with mocked dependencies."""
        mock_db = Mock()
        mock_cache = Mock()
        return DocumentService(db_client=mock_db, cache_client=mock_cache)

    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        return Document(
            id="doc_123",
            title="Test Document",
            content="Test content",
            source="test",
            type="research_paper"
        )

    async def test_get_document_success(self, document_service, sample_document):
        """Test successful document retrieval."""
        # Arrange
        document_service.db_client.get_by_id.return_value = sample_document

        # Act
        result = await document_service.get_by_id("doc_123")

        # Assert
        assert result == sample_document
        document_service.db_client.get_by_id.assert_called_once_with("doc_123")

    async def test_get_document_not_found(self, document_service):
        """Test document not found scenario."""
        # Arrange
        document_service.db_client.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(DocumentNotFoundError):
            await document_service.get_by_id("nonexistent")

    async def test_create_document(self, document_service):
        """Test document creation."""
        # Arrange
        document_data = {
            "title": "New Document",
            "content": "New content",
            "source": "test",
            "type": "research_paper"
        }
        expected_document = Document(**document_data, id="doc_new")
        document_service.db_client.create.return_value = expected_document

        # Act
        result = await document_service.create(**document_data)

        # Assert
        assert result == expected_document
        document_service.db_client.create.assert_called_once_with(**document_data)

    @patch('Medical_KG_rev.services.document.logger')
    async def test_create_document_with_logging(self, mock_logger, document_service):
        """Test document creation with logging."""
        # Arrange
        document_data = {
            "title": "New Document",
            "content": "New content",
            "source": "test",
            "type": "research_paper"
        }
        expected_document = Document(**document_data, id="doc_new")
        document_service.db_client.create.return_value = expected_document

        # Act
        result = await document_service.create(**document_data)

        # Assert
        mock_logger.info.assert_called_once_with(f"Document created: {expected_document.id}")
```

#### Integration Testing

```python
import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from Medical_KG_rev.gateway.main import app
from Medical_KG_rev.storage.database import get_db
from Medical_KG_rev.models import Document

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "title": "Test Document",
        "content": "Test content",
        "source": "test",
        "type": "research_paper",
        "metadata": {"author": "Test Author"}
    }

class TestDocumentAPI:
    """Integration tests for Document API."""

    def test_create_document_success(self, client, sample_document_data):
        """Test successful document creation via API."""
        # Act
        response = client.post(
            "/documents",
            json={"data": {"type": "documents", "attributes": sample_document_data}}
        )

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["data"]["attributes"]["title"] == sample_document_data["title"]
        assert "id" in data["data"]

    def test_get_document_success(self, client, sample_document_data):
        """Test successful document retrieval via API."""
        # Arrange - Create document first
        create_response = client.post(
            "/documents",
            json={"data": {"type": "documents", "attributes": sample_document_data}}
        )
        document_id = create_response.json()["data"]["id"]

        # Act
        response = client.get(f"/documents/{document_id}")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["attributes"]["title"] == sample_document_data["title"]

    def test_get_document_not_found(self, client):
        """Test document not found via API."""
        # Act
        response = client.get("/documents/nonexistent")

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["code"] == "not_found"

    async def test_document_processing_pipeline(self, async_client, sample_document_data):
        """Test complete document processing pipeline."""
        # Create document
        create_response = await async_client.post(
            "/documents",
            json={"data": {"type": "documents", "attributes": sample_document_data}}
        )
        assert create_response.status_code == 201
        document_id = create_response.json()["data"]["id"]

        # Wait for processing
        max_attempts = 30
        for attempt in range(max_attempts):
            response = await async_client.get(f"/documents/{document_id}")
            status = response.json()["data"]["attributes"]["status"]

            if status == "processed":
                break
            elif status == "failed":
                pytest.fail("Document processing failed")

            await asyncio.sleep(1)
        else:
            pytest.fail("Document processing timeout")

        # Verify processing results
        response = await async_client.get(f"/documents/{document_id}")
        document = response.json()["data"]

        assert document["attributes"]["status"] == "processed"
        assert "extractions" in document["attributes"]
        assert "entities" in document["relationships"]
```

#### Contract Testing

```python
import pytest
import schemathesis
from Medical_KG_rev.gateway.main import app

# Create schemathesis test client
schema = schemathesis.from_asgi(app)

@schema.parametrize()
def test_api_contract(request, case):
    """Test API contract compliance."""
    response = case.call()
    case.validate_response(response)

@pytest.mark.parametrize("endpoint", [
    "/documents",
    "/entities",
    "/search/documents",
    "/search/entities"
])
def test_api_endpoints_exist(client, endpoint):
    """Test that all required endpoints exist."""
    response = client.get(endpoint)
    assert response.status_code in [200, 405]  # 405 = Method Not Allowed

def test_api_response_format(client):
    """Test API response format compliance."""
    response = client.get("/documents")
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    assert "meta" in data

    # Check pagination metadata
    if "pagination" in data["meta"]:
        pagination = data["meta"]["pagination"]
        assert "page" in pagination
        assert "per_page" in pagination
        assert "total" in pagination
```

#### Performance Testing

```python
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from Medical_KG_rev.services.search import SearchService

class TestSearchPerformance:
    """Performance tests for search functionality."""

    @pytest.fixture
    def search_service(self):
        """Create SearchService instance."""
        return SearchService()

    async def test_search_latency_p95(self, search_service):
        """Test P95 latency requirement (< 500ms)."""
        query = "metformin diabetes treatment"
        latencies = []

        # Run 100 searches
        for _ in range(100):
            start_time = time.time()
            await search_service.search_documents(query, limit=10)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)

        # Calculate P95
        latencies.sort()
        p95_index = int(0.95 * len(latencies))
        p95_latency = latencies[p95_index]

        assert p95_latency < 500, f"P95 latency {p95_latency}ms exceeds 500ms requirement"

    async def test_concurrent_search(self, search_service):
        """Test concurrent search performance."""
        query = "diabetes treatment"
        concurrent_requests = 50

        async def single_search():
            return await search_service.search_documents(query, limit=5)

        # Run concurrent searches
        start_time = time.time()
        tasks = [single_search() for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Verify all searches completed
        assert len(results) == concurrent_requests

        # Verify performance (should complete within 10 seconds)
        assert total_time < 10, f"Concurrent searches took {total_time}s, exceeding 10s limit"

    def test_search_throughput(self, search_service):
        """Test search throughput requirements."""
        query = "clinical trial"
        duration = 60  # 1 minute
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < duration:
            asyncio.run(search_service.search_documents(query, limit=1))
            request_count += 1

        # Calculate requests per second
        rps = request_count / duration

        # Should handle at least 100 RPS
        assert rps >= 100, f"Throughput {rps} RPS below 100 RPS requirement"
```

## Deployment and DevOps

### Containerization

#### Dockerfile

```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "Medical_KG_rev.gateway.main"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  api-gateway:
    build:
      context: .
      dockerfile: docker/Dockerfile.gateway
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgresql:5432/medical_kg_rev
      - REDIS_URL=redis://redis:6379/0
      - NEO4J_URI=bolt://neo4j:7687
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - postgresql
      - redis
      - neo4j
      - qdrant
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  document-service:
    build:
      context: .
      dockerfile: docker/Dockerfile.document
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgresql:5432/medical_kg_rev
      - REDIS_URL=redis://redis:6379/0
      - NEO4J_URI=bolt://neo4j:7687
    depends_on:
      - postgresql
      - redis
      - neo4j
    restart: unless-stopped

  search-service:
    build:
      context: .
      dockerfile: docker/Dockerfile.search
    environment:
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - qdrant
      - redis
    restart: unless-stopped

  postgresql:
    image: postgres:14
    environment:
      - POSTGRES_DB=medical_kg_rev
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  neo4j:
    image: neo4j:5
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    ports:
      - "7474:7474"
      - "7687:7687"
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  neo4j_data:
  neo4j_logs:
  qdrant_data:
```

### Kubernetes Deployment

#### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: medical-kg-rev
  labels:
    app: api-gateway
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
        version: v1.0.0
    spec:
      containers:
      - name: api-gateway
        image: medical-kg-rev/api-gateway:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        - name: NEO4J_URI
          valueFrom:
            secretKeyRef:
              name: neo4j-secret
              key: uri
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
      nodeSelector:
        kubernetes.io/os: linux
      tolerations:
      - key: "node.kubernetes.io/not-ready"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 300
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
  namespace: medical-kg-rev
spec:
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-gateway-ingress
  namespace: medical-kg-rev
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.medical-kg-rev.com
    secretName: api-tls
  rules:
  - host: api.medical-kg-rev.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway-service
            port:
              number: 80
```

### CI/CD Pipeline

#### GitHub Actions Workflow

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: medical-kg-rev

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgresql:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

      neo4j:
        image: neo4j:5
        env:
          NEO4J_AUTH: neo4j/password
        options: >-
          --health-cmd "cypher-shell -u neo4j -p password 'RETURN 1'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 7474:7474
          - 7687:7687

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run linting
      run: |
        flake8 src/
        black --check src/
        isort --check-only src/
        mypy src/

    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/0
        NEO4J_URI: bolt://localhost:7687
        NEO4J_USER: neo4j
        NEO4J_PASSWORD: password
      run: |
        pytest tests/ -v --cov=src/ --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: docker/Dockerfile.gateway
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}

    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f kubernetes/
        kubectl rollout status deployment/api-gateway -n medical-kg-rev
        kubectl rollout status deployment/document-service -n medical-kg-rev
        kubectl rollout status deployment/search-service -n medical-kg-rev
```

## Performance Optimization

### Database Optimization

#### Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection manager with pooling."""

    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def get_session(self):
        """Get database session."""
        return self.SessionLocal()

    def close(self):
        """Close database connections."""
        self.engine.dispose()

# Usage
db_manager = DatabaseManager(DATABASE_URL)

async def get_documents(page: int = 1, per_page: int = 20):
    """Get documents with pagination."""
    session = db_manager.get_session()
    try:
        offset = (page - 1) * per_page
        documents = session.query(Document).offset(offset).limit(per_page).all()
        return documents
    finally:
        session.close()
```

#### Query Optimization

```python
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy import func, desc

async def get_documents_with_entities(document_ids: List[str]):
    """Optimized query to get documents with entities."""
    session = db_manager.get_session()
    try:
        # Use joinedload to avoid N+1 queries
        documents = session.query(Document)\
            .options(joinedload(Document.entities))\
            .filter(Document.id.in_(document_ids))\
            .all()

        return documents
    finally:
        session.close()

async def get_document_statistics():
    """Get document statistics with optimized query."""
    session = db_manager.get_session()
    try:
        stats = session.query(
            Document.source,
            func.count(Document.id).label('count'),
            func.max(Document.created_at).label('latest')
        ).group_by(Document.source).all()

        return stats
    finally:
        session.close()
```

### Caching Strategy

#### Redis Caching

```python
import redis
import json
import pickle
from typing import Any, Optional
from datetime import timedelta

class CacheManager:
    """Redis cache manager."""

    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600
    ) -> bool:
        """Set value in cache."""
        try:
            serialized_value = pickle.dumps(value)
            return self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            return self.redis_client.delete(key) > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def get_or_set(
        self,
        key: str,
        func,
        ttl: int = 3600
    ) -> Any:
        """Get from cache or set using function."""
        value = await self.get(key)
        if value is not None:
            return value

        value = await func()
        await self.set(key, value, ttl)
        return value

# Usage
cache_manager = CacheManager(REDIS_URL)

async def get_document_cached(document_id: str) -> Optional[Document]:
    """Get document with caching."""
    cache_key = f"document:{document_id}"

    return await cache_manager.get_or_set(
        cache_key,
        lambda: document_service.get_by_id(document_id),
        ttl=1800  # 30 minutes
    )
```

### API Performance

#### Response Compression

```python
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/documents")
async def get_documents():
    """Get documents with compression."""
    # Large response will be automatically compressed
    return await document_service.list_all()
```

#### Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

class AsyncDocumentProcessor:
    """Async document processor for better performance."""

    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_documents_batch(
        self,
        documents: List[Document]
    ) -> List[ProcessedDocument]:
        """Process documents in parallel."""
        tasks = [
            self.process_single_document(doc)
            for doc in documents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        processed_docs = [
            result for result in results
            if not isinstance(result, Exception)
        ]

        return processed_docs

    async def process_single_document(
        self,
        document: Document
    ) -> ProcessedDocument:
        """Process single document asynchronously."""
        loop = asyncio.get_event_loop()

        # Run CPU-intensive tasks in thread pool
        processed_content = await loop.run_in_executor(
            self.executor,
            self._extract_entities,
            document.content
        )

        return ProcessedDocument(
            id=document.id,
            content=processed_content,
            entities=processed_content.entities
        )

    def _extract_entities(self, content: str) -> ExtractedContent:
        """CPU-intensive entity extraction."""
        # Implementation
        pass
```

## Security Considerations

### Authentication and Authorization

#### JWT Token Management

```python
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import secrets

class TokenManager:
    """JWT token management."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create access token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None

    def create_refresh_token(self, data: dict) -> str:
        """Create refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
```

#### Role-Based Access Control

```python
from enum import Enum
from typing import List, Set
from functools import wraps

class Permission(Enum):
    READ_DOCUMENTS = "read:documents"
    WRITE_DOCUMENTS = "write:documents"
    DELETE_DOCUMENTS = "delete:documents"
    READ_ENTITIES = "read:entities"
    WRITE_ENTITIES = "write:entities"
    ADMIN_ACCESS = "admin:access"

class Role(Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    CLINICIAN = "clinician"
    ANALYST = "analyst"
    VIEWER = "viewer"

# Role-permission mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: {
        Permission.READ_DOCUMENTS,
        Permission.WRITE_DOCUMENTS,
        Permission.DELETE_DOCUMENTS,
        Permission.READ_ENTITIES,
        Permission.WRITE_ENTITIES,
        Permission.ADMIN_ACCESS
    },
    Role.RESEARCHER: {
        Permission.READ_DOCUMENTS,
        Permission.WRITE_DOCUMENTS,
        Permission.READ_ENTITIES
    },
    Role.CLINICIAN: {
        Permission.READ_DOCUMENTS,
        Permission.READ_ENTITIES
    },
    Role.ANALYST: {
        Permission.READ_DOCUMENTS,
        Permission.READ_ENTITIES
    },
    Role.VIEWER: {
        Permission.READ_DOCUMENTS
    }
}

def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")

            user_permissions = get_user_permissions(current_user)
            if permission not in user_permissions:
                raise HTTPException(status_code=403, detail="Insufficient permissions")

            return await func(*args, **kwargs)
        return wrapper
    return decorator

def get_user_permissions(user: User) -> Set[Permission]:
    """Get user permissions based on roles."""
    permissions = set()
    for role in user.roles:
        permissions.update(ROLE_PERMISSIONS.get(role, set()))
    return permissions
```

### Input Validation

#### Pydantic Models

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import re

class DocumentCreate(BaseModel):
    """Document creation model with validation."""

    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., regex="^(research_paper|clinical_trial|case_report)$")
    metadata: Optional[dict] = None

    @validator('title')
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

    @validator('content')
    def validate_content(cls, v):
        if len(v) > 1000000:  # 1MB limit
            raise ValueError('Content too large')
        return v

    @validator('source')
    def validate_source(cls, v):
        allowed_sources = ['pubmed', 'clinicaltrials', 'manual_upload']
        if v not in allowed_sources:
            raise ValueError(f'Invalid source. Must be one of: {allowed_sources}')
        return v

class SearchRequest(BaseModel):
    """Search request model with validation."""

    query: str = Field(..., min_length=1, max_length=500)
    strategy: str = Field(default="hybrid", regex="^(bm25|dense|sparse|hybrid)$")
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    filters: Optional[dict] = None

    @validator('query')
    def validate_query(cls, v):
        # Remove potentially dangerous characters
        v = re.sub(r'[<>"\']', '', v)
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

    @validator('filters')
    def validate_filters(cls, v):
        if v:
            allowed_keys = ['source', 'type', 'date_from', 'date_to']
            for key in v.keys():
                if key not in allowed_keys:
                    raise ValueError(f'Invalid filter key: {key}')
        return v
```

### Data Protection

#### Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    """Data encryption utilities."""

    def __init__(self, password: str = None):
        if password is None:
            password = os.getenv("ENCRYPTION_PASSWORD", "default-password")

        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'medical_kg_rev_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.cipher_suite = Fernet(key)

    def encrypt(self, data: str) -> str:
        """Encrypt data."""
        encrypted = self.cipher_suite.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data."""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted.decode()

# Usage
encryption = DataEncryption()

def encrypt_sensitive_data(data: str) -> str:
    """Encrypt sensitive data before storage."""
    return encryption.encrypt(data)

def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive data after retrieval."""
    return encryption.decrypt(encrypted_data)
```

## Troubleshooting

### Common Issues

#### Database Connection Issues

```python
import logging
from sqlalchemy.exc import OperationalError, DisconnectionError

logger = logging.getLogger(__name__)

async def handle_database_errors(func):
    """Decorator to handle database errors."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except OperationalError as e:
            logger.error(f"Database operational error: {e}")
            raise HTTPException(status_code=503, detail="Database temporarily unavailable")
        except DisconnectionError as e:
            logger.error(f"Database connection error: {e}")
            raise HTTPException(status_code=503, detail="Database connection lost")
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
            raise HTTPException(status_code=500, detail="Internal database error")
    return wrapper

# Usage
@handle_database_errors
async def get_document(doc_id: str) -> Document:
    """Get document with error handling."""
    return await document_service.get_by_id(doc_id)
```

#### Performance Issues

```python
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def monitor_performance(threshold_ms: int = 1000):
    """Decorator to monitor performance."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            if duration_ms > threshold_ms:
                logger.warning(
                    f"Slow operation: {func.__name__} took {duration_ms:.2f}ms "
                    f"(threshold: {threshold_ms}ms)"
                )

            return result
        return wrapper
    return decorator

# Usage
@monitor_performance(threshold_ms=500)
async def search_documents(query: str, limit: int = 20):
    """Search documents with performance monitoring."""
    return await search_service.search(query, limit)
```

### Debugging Tools

#### Logging Configuration

```python
import logging
import sys
from datetime import datetime

def setup_logging(level: str = "INFO"):
    """Setup application logging."""

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Specific loggers
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

# Usage
setup_logging("DEBUG")
```

#### Health Checks

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import asyncio

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }

    # Check database
    try:
        await database_service.ping()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    # Check Redis
    try:
        await cache_service.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    # Check Neo4j
    try:
        await graph_service.ping()
        health_status["checks"]["neo4j"] = "healthy"
    except Exception as e:
        health_status["checks"]["neo4j"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    # Check if all required services are available
    try:
        await asyncio.gather(
            database_service.ping(),
            cache_service.ping(),
            graph_service.ping()
        )
        return JSONResponse(content={"status": "ready"}, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"status": "not_ready", "error": str(e)},
            status_code=503
        )
```

This comprehensive developer guide provides all the technical information needed for developers to work effectively with the Medical_KG_rev system. It covers architecture, development setup, coding standards, API development, database design, testing, deployment, performance optimization, security, and troubleshooting.
