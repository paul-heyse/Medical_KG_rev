"""GraphQL schema definition leveraging the shared gateway service."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from fastapi import Request
from strawberry import ID
from strawberry.fastapi import GraphQLRouter
from strawberry.scalars import JSON
from strawberry.types import Info
import strawberry

from Medical_KG_rev.auth.scopes import Scopes

from ..models import HttpClient
from ..services import GatewayService, get_gateway_service
from .context import GraphQLContext, build_context



def _operation_status_to_type(status: OperationStatus) -> OperationStatusType:
    return OperationStatusType(
        job_id=status.job_id,
        status=status.status,
        message=status.message,
        metadata=status.metadata,
    )


def _document_to_type(document: DocumentSummary) -> DocumentType:
    return DocumentType(
        id=document.id,
        title=document.title,
        score=document.score,
        summary=document.summary,
        source=document.source,
        metadata=document.metadata,
        explain=document.explain,
    )


def _retrieval_to_type(result: RetrievalResult) -> RetrievalResultType:
    return RetrievalResultType(
        query=result.query,
        total=result.total,
        documents=[_document_to_type(document) for document in result.documents],
        pipeline_version=result.pipeline_version,
        partial=result.partial,
        degraded=result.degraded,
        rerank_metrics=result.rerank_metrics,
        stage_timings=result.stage_timings,
        intent=result.intent,
        errors=[_problem_to_type(problem) for problem in result.errors],
    )


def _problem_to_type(problem) -> ProblemDetailType:
    return ProblemDetailType(
        type=problem.type,
        title=problem.title,
        status=problem.status,
        detail=problem.detail,
        extensions=problem.extensions,
    )


def _embedding_to_type(vector: EmbeddingVector) -> EmbeddingVectorType:
    return EmbeddingVectorType(
        id=vector.id,
        model=vector.model,
        namespace=vector.namespace,
        kind=vector.kind,
        dimension=vector.dimension,
        vector=list(vector.vector or []),
        terms=vector.terms,
        metadata=vector.metadata,
    )


def _embedding_result_to_type(result: EmbeddingResponse) -> EmbeddingResultType:
    return EmbeddingResultType(
        namespace=result.namespace,
        embeddings=[_embedding_to_type(vector) for vector in result.embeddings],
        metadata=EmbeddingMetadataType(
            provider=result.metadata.provider,
            dimension=result.metadata.dimension,
            duration_ms=result.metadata.duration_ms,
            model=result.metadata.model,
        ),
    )


def _namespace_to_type(info: NamespaceInfo) -> NamespaceInfoType:
    return NamespaceInfoType(
        id=info.id,
        provider=info.provider,
        kind=info.kind,
        dimension=info.dimension,
        max_tokens=info.max_tokens,
        enabled=info.enabled,
        allowed_tenants=info.allowed_tenants,
        allowed_scopes=info.allowed_scopes,
    )


def _chunk_to_type(chunk: DocumentChunk) -> DocumentChunkType:
    return DocumentChunkType(
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        metadata=chunk.metadata,
        token_count=chunk.token_count,
    )


def _adapter_metadata_to_type(metadata: AdapterMetadataView) -> AdapterMetadataType:
    return AdapterMetadataType(
        name=metadata.name,
        version=metadata.version,
        domain=metadata.domain.value,
        summary=metadata.summary,
        capabilities=metadata.capabilities,
        maintainer=metadata.maintainer,
        dataset=metadata.dataset,
        config_schema=metadata.config_schema,
    )


def _adapter_health_to_type(health: AdapterHealthView) -> AdapterHealthType:
    return AdapterHealthType(name=health.name, healthy=health.healthy)


@strawberry.type
class OperationStatusType:
    job_id: str
    status: str
    message: str | None
    metadata: JSON


@strawberry.type
class BatchOperationType:
    total: int
    operations: list[OperationStatusType]


@strawberry.type
class AdapterMetadataType:
    name: str
    version: str
    domain: str
    summary: str
    capabilities: list[str]
    maintainer: str | None
    dataset: str | None
    config_schema: JSON


@strawberry.type
class AdapterHealthType:
    name: str
    healthy: bool


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
    summary: str | None
    source: str
    metadata: JSON
    explain: JSON | None = None

    @strawberry.field
    async def organization(
        self, info: Info[GraphQLContext, None], id: str | None = None
    ) -> OrganizationType:
        identifier = id or f"org-{self.id}"  # synthetic identifier
        data = await info.context.loaders.organization_loader.load(identifier)
        return OrganizationType(id=data["id"], name=data["name"], country=data["country"])

    @strawberry.field
    async def claims(self, info: Info[GraphQLContext, None]) -> list[ClaimType]:
        # Return synthetic claims derived from document id
        return [
            ClaimType(id=f"claim-{self.id}", predicate="supports", object_id="obj-1"),
        ]


@strawberry.type
class RetrievalResultType:
    query: str
    total: int
    documents: list[DocumentType]
    pipeline_version: str | None
    partial: bool
    degraded: bool
    rerank_metrics: JSON
    stage_timings: JSON
    intent: JSON
    errors: list[ProblemDetailType]


@strawberry.type
class ProblemDetailType:
    type: str
    title: str
    status: int
    detail: str | None
    extensions: JSON


@strawberry.type
class EmbeddingVectorType:
    id: str
    model: str
    namespace: str
    kind: str
    dimension: int | None
    vector: list[float] | None
    terms: JSON | None
    metadata: JSON


@strawberry.type
class EmbeddingMetadataType:
    provider: str
    dimension: int | None
    duration_ms: float | None
    model: str | None


@strawberry.type
class EmbeddingResultType:
    namespace: str
    embeddings: list[EmbeddingVectorType]
    metadata: EmbeddingMetadataType


@strawberry.type
class DocumentChunkType:
    document_id: str
    chunk_index: int
    content: str
    metadata: JSON
    token_count: int


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


@strawberry.type
class NamespaceInfoType:
    id: str
    provider: str
    kind: str
    dimension: int | None
    max_tokens: int | None
    enabled: bool
    allowed_tenants: list[str]
    allowed_scopes: list[str]


@strawberry.input
class PaginationInput:
    after: str | None = None
    first: int = 10


@strawberry.input
class SearchInput:
    query: str
    filters: JSON = strawberry.field(default_factory=dict)
    pagination: PaginationInput = strawberry.field(default_factory=PaginationInput)
    query_intent: str | None = None
    table_only: bool = False


@strawberry.input
class IngestionInput:
    tenant_id: str
    items: JSON
    priority: str = "normal"
    metadata: JSON = strawberry.field(default_factory=dict)
    profile: str | None = None


@strawberry.input
class ChunkInput:
    tenant_id: str
    document_id: str
    strategy: str = "semantic"
    chunk_size: int = 1024


@strawberry.input
class EmbedInput:
    tenant_id: str | None = None
    namespace: str
    texts: list[str]
    options: EmbeddingOptionsInput | None = None


@strawberry.input
class EmbeddingOptionsInput:
    normalize: bool = True
    model: str | None = None


@strawberry.input
class RetrieveInput:
    tenant_id: str
    query: str
    top_k: int = 5
    filters: JSON = strawberry.field(default_factory=dict)
    rerank: bool = True
    rerank_model: str | None = None
    rerank_top_k: int = 10
    rerank_overflow: bool = False
    profile: str | None = None
    metadata: JSON = strawberry.field(default_factory=dict)
    explain: bool = False
    query_intent: str | None = None
    table_only: bool = False


@strawberry.input
class EntityLinkInput:
    tenant_id: str
    mentions: list[str]
    context: str | None = None


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
    async def adapters(
        self, info: Info[GraphQLContext, None], domain: str | None = None
    ) -> list[AdapterMetadataType]:
        service = await _get_service(info)
        try:
            metadata = service.list_adapters(domain)
        except ValueError as exc:
            raise ValueError(str(exc))
        return [_adapter_metadata_to_type(item) for item in metadata]

    @strawberry.field
    async def adapter(
        self, info: Info[GraphQLContext, None], name: str
    ) -> AdapterMetadataType | None:
        service = await _get_service(info)
        metadata = service.get_adapter_metadata(name)
        if metadata is None:
            return None
        return _adapter_metadata_to_type(metadata)

    @strawberry.field
    async def namespaces(
        self, info: Info[GraphQLContext, None], tenant_id: str | None = None
    ) -> list[NamespaceInfoType]:
        service = await _get_service(info)
        resolved_tenant = tenant_id or info.context.tenant_id
        namespaces = service.list_namespaces(tenant_id=resolved_tenant, scope=Scopes.EMBED_READ)
        return [_namespace_to_type(item) for item in namespaces]

    @strawberry.field
    async def namespace(
        self, info: Info[GraphQLContext, None], id: str, tenant_id: str | None = None
    ) -> NamespaceInfoType | None:
        service = await _get_service(info)
        resolved_tenant = tenant_id or info.context.tenant_id
        namespaces = service.list_namespaces(tenant_id=resolved_tenant, scope=Scopes.EMBED_READ)
        for item in namespaces:
            if item.id == id:
                return _namespace_to_type(item)
        return None

    @strawberry.field
    async def adapter_health(
        self, info: Info[GraphQLContext, None], name: str
    ) -> AdapterHealthType | None:
        service = await _get_service(info)
        health = service.get_adapter_health(name)
        if health is None:
            return None
        return _adapter_health_to_type(health)

    @strawberry.field
    async def document(self, info: Info[GraphQLContext, None], id: ID) -> DocumentType:
        doc = await info.context.loaders.document_loader.load(str(id))
        return _document_to_type(doc)

    @strawberry.field
    async def organization(self, info: Info[GraphQLContext, None], id: str) -> OrganizationType:
        data = await info.context.loaders.organization_loader.load(id)
        return OrganizationType(id=data["id"], name=data["name"], country=data["country"])

    @strawberry.field
    async def search(
        self, info: Info[GraphQLContext, None], arguments: SearchInput
    ) -> RetrievalResultType:
        service = await _get_service(info)
        args = SearchArguments(
            query=arguments.query,
            filters=dict(arguments.filters or {}),
            pagination=Pagination(**asdict(arguments.pagination)),
            query_intent=arguments.query_intent,
            table_only=arguments.table_only,
        )
        result = service.search(args)
        return _retrieval_to_type(result)


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def ingest(
        self, info: Info[GraphQLContext, None], dataset: str, input: IngestionInput
    ) -> BatchOperationType:
        service = await _get_service(info)
        request = IngestionRequest(
            tenant_id=input.tenant_id,
            items=_ensure_sequence(input.items),
            priority=input.priority,  # type: ignore[arg-type]
            metadata=dict(input.metadata or {}),
            profile=input.profile,
        )
        result = service.ingest(dataset, request)
        return BatchOperationType(
            total=result.total,
            operations=[_operation_status_to_type(status) for status in result.operations],
        )

    @strawberry.mutation
    async def chunk(
        self, info: Info[GraphQLContext, None], input: ChunkInput
    ) -> list[DocumentChunkType]:
        service = await _get_service(info)
        request = ChunkRequest(**asdict(input))
        return [_chunk_to_type(chunk) for chunk in service.chunk_document(request)]

    @strawberry.mutation
    async def embed(
        self, info: Info[GraphQLContext, None], input: EmbedInput
    ) -> EmbeddingResultType:
        service = await _get_service(info)
        options = None
        if input.options is not None:
            options = EmbeddingOptions(**asdict(input.options))
        tenant_id = input.tenant_id or info.context.tenant_id
        if not tenant_id:
            raise ValueError("Tenant ID is required for embedding operations")
        request = EmbedRequest(
            tenant_id=tenant_id,
            texts=list(input.texts),
            namespace=input.namespace,
            options=options,
        )
        response = service.embed(request)
        return _embedding_result_to_type(response)

    @strawberry.mutation
    async def retrieve(
        self, info: Info[GraphQLContext, None], input: RetrieveInput
    ) -> RetrievalResultType:
        service = await _get_service(info)
        request = RetrieveRequest(**asdict(input))
        result = service.retrieve(request)
        return _retrieval_to_type(result)

    @strawberry.mutation
    async def entity_link(
        self, info: Info[GraphQLContext, None], input: EntityLinkInput
    ) -> BatchOperationType:
        service = await _get_service(info)
        request = EntityLinkRequest(**asdict(input))
        result = service.entity_link(request)
        batch = build_batch_result(
            [
                OperationStatus(
                    job_id=item.entity_id, status="completed", metadata={"mention": item.mention}
                )
                for item in result
            ]
        )
        return BatchOperationType(
            total=batch.total,
            operations=[_operation_status_to_type(status) for status in batch.operations],
        )

    @strawberry.mutation
    async def extract(
        self, info: Info[GraphQLContext, None], kind: str, input: ExtractionInput
    ) -> ExtractionResultType:
        service = await _get_service(info)
        request = ExtractionRequest(**asdict(input))
        result = service.extract(kind, request)
        return ExtractionResultType(
            kind=result.kind, document_id=result.document_id, results=result.results
        )

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


async def get_context(request: Request) -> GraphQLContext:
    service = get_gateway_service()
    return await build_context(service, request)


graphql_router = GraphQLRouter(schema, context_getter=get_context)
