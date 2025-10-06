"""GraphQL schema definition leveraging the shared gateway service."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, List, Optional, Sequence

import strawberry
from strawberry import ID
from strawberry.fastapi import GraphQLRouter
from strawberry.scalars import JSON
from strawberry.types import Info

from ..models import (
    ChunkRequest,
    DocumentChunk,
    DocumentSummary,
    EmbedRequest,
    EmbeddingVector,
    EntityLinkRequest,
    ExtractionRequest,
    IngestionRequest,
    KnowledgeGraphWriteRequest,
    OperationStatus,
    Pagination,
    RetrievalResult,
    RetrieveRequest,
    SearchArguments,
    build_batch_result,
)
from ..services import GatewayService, get_gateway_service
from .context import GraphQLContext, build_context


def _operation_status_to_type(status: OperationStatus) -> "OperationStatusType":
    return OperationStatusType(
        job_id=status.job_id,
        status=status.status,
        message=status.message,
        metadata=status.metadata,
    )


def _document_to_type(document: DocumentSummary) -> "DocumentType":
    return DocumentType(
        id=document.id,
        title=document.title,
        score=document.score,
        summary=document.summary,
        source=document.source,
        metadata=document.metadata,
    )


def _retrieval_to_type(result: RetrievalResult) -> "RetrievalResultType":
    return RetrievalResultType(
        query=result.query,
        total=result.total,
        documents=[_document_to_type(document) for document in result.documents],
    )


def _embedding_to_type(vector: EmbeddingVector) -> "EmbeddingVectorType":
    return EmbeddingVectorType(
        id=vector.id,
        vector=vector.vector,
        model=vector.model,
        metadata=vector.metadata,
    )


def _chunk_to_type(chunk: DocumentChunk) -> "DocumentChunkType":
    return DocumentChunkType(
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        metadata=chunk.metadata,
    )


@strawberry.type
class OperationStatusType:
    job_id: str
    status: str
    message: Optional[str]
    metadata: JSON


@strawberry.type
class BatchOperationType:
    total: int
    operations: List[OperationStatusType]


@strawberry.type
class OrganizationType:
    id: str
    name: str
    country: str


@strawberry.type
class ClaimType:
    id: str
    predicate: str
    object_id: str


@strawberry.type
class DocumentType:
    id: str
    title: str
    score: float
    summary: Optional[str]
    source: str
    metadata: JSON

    @strawberry.field
    async def organization(self, info: Info[GraphQLContext, None], id: Optional[str] = None) -> OrganizationType:
        identifier = id or f"org-{self.id}"  # synthetic identifier
        data = await info.context.loaders.organization_loader.load(identifier)
        return OrganizationType(id=data["id"], name=data["name"], country=data["country"])

    @strawberry.field
    async def claims(self, info: Info[GraphQLContext, None]) -> List[ClaimType]:
        # Return synthetic claims derived from document id
        return [
            ClaimType(id=f"claim-{self.id}", predicate="supports", object_id="obj-1"),
        ]


@strawberry.type
class RetrievalResultType:
    query: str
    total: int
    documents: List[DocumentType]


@strawberry.type
class EmbeddingVectorType:
    id: str
    vector: List[float]
    model: str
    metadata: JSON


@strawberry.type
class DocumentChunkType:
    document_id: str
    chunk_index: int
    content: str
    metadata: JSON


@strawberry.type
class ExtractionResultType:
    kind: str
    document_id: str
    results: JSON


@strawberry.type
class KnowledgeGraphWriteResultType:
    nodes_written: int
    edges_written: int
    metadata: JSON


@strawberry.input
class PaginationInput:
    after: Optional[str] = None
    first: int = 10


@strawberry.input
class SearchInput:
    query: str
    filters: JSON = strawberry.field(default_factory=dict)
    pagination: PaginationInput = strawberry.field(default_factory=PaginationInput)


@strawberry.input
class IngestionInput:
    tenant_id: str
    items: JSON
    priority: str = "normal"
    metadata: JSON = strawberry.field(default_factory=dict)


@strawberry.input
class ChunkInput:
    tenant_id: str
    document_id: str
    strategy: str = "semantic"
    chunk_size: int = 1024


@strawberry.input
class EmbedInput:
    tenant_id: str
    inputs: List[str]
    model: str
    normalize: bool = True


@strawberry.input
class RetrieveInput:
    tenant_id: str
    query: str
    top_k: int = 5
    filters: JSON = strawberry.field(default_factory=dict)


@strawberry.input
class EntityLinkInput:
    tenant_id: str
    mentions: List[str]
    context: Optional[str] = None


@strawberry.input
class ExtractionInput:
    tenant_id: str
    document_id: str
    options: JSON = strawberry.field(default_factory=dict)


@strawberry.input
class KnowledgeGraphWriteInput:
    tenant_id: str
    nodes: JSON = strawberry.field(default=None)
    edges: JSON = strawberry.field(default=None)
    transactional: bool = True


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence):
        return value
    return [value]


async def _get_service(info: Info[GraphQLContext, None]) -> GatewayService:
    return info.context.service


@strawberry.type
class Query:
    @strawberry.field
    async def document(self, info: Info[GraphQLContext, None], id: ID) -> DocumentType:
        service = await _get_service(info)
        doc = await info.context.loaders.document_loader.load(str(id))
        return _document_to_type(doc)

    @strawberry.field
    async def organization(self, info: Info[GraphQLContext, None], id: str) -> OrganizationType:
        data = await info.context.loaders.organization_loader.load(id)
        return OrganizationType(id=data["id"], name=data["name"], country=data["country"])

    @strawberry.field
    async def search(self, info: Info[GraphQLContext, None], arguments: SearchInput) -> RetrievalResultType:
        service = await _get_service(info)
        args = SearchArguments(
            query=arguments.query,
            filters=dict(arguments.filters or {}),
            pagination=Pagination(**asdict(arguments.pagination)),
        )
        result = service.search(args)
        return _retrieval_to_type(result)


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def ingest(self, info: Info[GraphQLContext, None], dataset: str, input: IngestionInput) -> BatchOperationType:
        service = await _get_service(info)
        request = IngestionRequest(
            tenant_id=input.tenant_id,
            items=_ensure_sequence(input.items),
            priority=input.priority,  # type: ignore[arg-type]
            metadata=dict(input.metadata or {}),
        )
        result = service.ingest(dataset, request)
        return BatchOperationType(
            total=result.total,
            operations=[_operation_status_to_type(status) for status in result.operations],
        )

    @strawberry.mutation
    async def chunk(self, info: Info[GraphQLContext, None], input: ChunkInput) -> List[DocumentChunkType]:
        service = await _get_service(info)
        request = ChunkRequest(**asdict(input))
        return [_chunk_to_type(chunk) for chunk in service.chunk_document(request)]

    @strawberry.mutation
    async def embed(self, info: Info[GraphQLContext, None], input: EmbedInput) -> List[EmbeddingVectorType]:
        service = await _get_service(info)
        request = EmbedRequest(**asdict(input))
        return [_embedding_to_type(vector) for vector in service.embed(request)]

    @strawberry.mutation
    async def retrieve(self, info: Info[GraphQLContext, None], input: RetrieveInput) -> RetrievalResultType:
        service = await _get_service(info)
        request = RetrieveRequest(**asdict(input))
        result = service.retrieve(request)
        return _retrieval_to_type(result)

    @strawberry.mutation
    async def entity_link(self, info: Info[GraphQLContext, None], input: EntityLinkInput) -> BatchOperationType:
        service = await _get_service(info)
        request = EntityLinkRequest(**asdict(input))
        result = service.entity_link(request)
        batch = build_batch_result(
            [
                OperationStatus(job_id=item.entity_id, status="completed", metadata={"mention": item.mention})
                for item in result
            ]
        )
        return BatchOperationType(
            total=batch.total,
            operations=[_operation_status_to_type(status) for status in batch.operations],
        )

    @strawberry.mutation
    async def extract(self, info: Info[GraphQLContext, None], kind: str, input: ExtractionInput) -> ExtractionResultType:
        service = await _get_service(info)
        request = ExtractionRequest(**asdict(input))
        result = service.extract(kind, request)
        return ExtractionResultType(kind=result.kind, document_id=result.document_id, results=result.results)

    @strawberry.mutation
    async def write_kg(
        self,
        info: Info[GraphQLContext, None],
        input: KnowledgeGraphWriteInput,
    ) -> KnowledgeGraphWriteResultType:
        service = await _get_service(info)
        request = KnowledgeGraphWriteRequest(**asdict(input))
        if isinstance(request.nodes, dict):
            request = request.model_copy(update={"nodes": list(request.nodes.values())})
        if isinstance(request.edges, dict):
            request = request.model_copy(update={"edges": list(request.edges.values())})
        if request.nodes is None:
            request = request.model_copy(update={"nodes": []})
        if request.edges is None:
            request = request.model_copy(update={"edges": []})
        result = service.write_kg(request)
        return KnowledgeGraphWriteResultType(
            nodes_written=result.nodes_written,
            edges_written=result.edges_written,
            metadata=result.metadata,
        )


schema = strawberry.Schema(query=Query, mutation=Mutation)


async def get_context() -> GraphQLContext:
    service = get_gateway_service()
    return await build_context(service)


graphql_router = GraphQLRouter(schema, context_getter=get_context)
