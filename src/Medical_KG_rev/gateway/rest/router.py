"""REST API router exposing gateway operations."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...auth import Scopes, SecurityContext, secure_endpoint
from ...auth.audit import get_audit_trail
from ...services.health import HealthService
from ..models import (
    AdapterConfigSchemaView,
    AdapterHealthView,
    AdapterMetadataView,
    BatchOperationResult,
    ChunkRequest,
    EmbeddingResponse,
    EmbedRequest,
    EntityLinkRequest,
    EvaluationRequest,
    EvaluationResponse,
    ExtractionRequest,
    IngestionRequest,
    JobStatus,
    NamespaceInfo,
    NamespaceValidationRequest,
    NamespaceValidationResponse,
    KnowledgeGraphWriteRequest,
    PipelineIngestionRequest,
    PipelineQueryRequest,
    RetrievalResult,
    RetrieveRequest,
)
from ..presentation.dependencies import get_response_presenter
from ..presentation.errors import ErrorDetail
from ..presentation.interface import ResponsePresenter
from ..presentation.odata import ODataParams
from ..presentation.requests import apply_tenant_context
from ..services import GatewayService, get_gateway_service
from ..services.retrieval.routing import QueryIntent

router = APIRouter(prefix="/v1", tags=["gateway"])
health_router = APIRouter(tags=["system"])


@router.get("/adapters", response_model=None)
async def list_adapters(
    domain: str | None = Query(default=None),
    service: GatewayService = Depends(get_gateway_service),
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.ADAPTERS_READ], endpoint="GET /v1/adapters")
    ),
    presenter: PresenterDep,
) -> JSONResponse:
    try:
        adapters = service.list_adapters(domain)
    except ValueError as exc:
        return presenter.error(
            ErrorDetail(
                status=400,
                code="invalid-adapter-domain",
                title="Invalid adapter domain",
                detail=str(exc),
            )
        )
    return presenter.success(adapters, meta={"total": len(adapters)})


@router.get("/adapters/{name}/metadata", response_model=None)
async def get_adapter_metadata(
    name: str = Path(..., description="Adapter name"),
    service: GatewayService = Depends(get_gateway_service),
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.ADAPTERS_READ], endpoint="GET /v1/adapters/{name}/metadata")
    ),
    presenter: PresenterDep,
) -> JSONResponse:
    metadata = service.get_adapter_metadata(name)
    if metadata is None:
        return presenter.error(
            ErrorDetail(
                status=404,
                code="adapter-not-found",
                title="Adapter not found",
                detail=f"Adapter '{name}' was not registered",
            ),
            status_code=404,
        )
    return presenter.success(metadata)


@router.get("/adapters/{name}/health", response_model=None)
async def get_adapter_health(
    name: str,
    service: GatewayService = Depends(get_gateway_service),
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.ADAPTERS_READ], endpoint="GET /v1/adapters/{name}/health")
    ),
    presenter: PresenterDep,
) -> JSONResponse:
    health = service.get_adapter_health(name)
    if health is None:
        return presenter.error(
            ErrorDetail(
                status=404,
                code="adapter-not-found",
                title="Adapter not found",
                detail=f"Adapter '{name}' was not registered",
            ),
            status_code=404,
        )
    return presenter.success(health)


@router.get("/adapters/{name}/config-schema", response_model=None)
async def get_adapter_config_schema(
    name: str,
    service: GatewayService = Depends(get_gateway_service),
    security: SecurityContext = Depends(
        secure_endpoint(
            scopes=[Scopes.ADAPTERS_READ], endpoint="GET /v1/adapters/{name}/config-schema"
        )
    ),
    presenter: PresenterDep,
) -> JSONResponse:
    schema = service.get_adapter_config_schema(name)
    if schema is None:
        return presenter.error(
            ErrorDetail(
                status=404,
                code="adapter-not-found",
                title="Adapter not found",
                detail=f"Adapter '{name}' was not registered",
            ),
            status_code=404,
        )
    return presenter.success(schema)


@health_router.get("/health", include_in_schema=True)
async def health_check(request: Request) -> JSONResponse:
    service: HealthService = request.app.state.health  # type: ignore[attr-defined]
    return JSONResponse(service.liveness())


@health_router.get("/ready", include_in_schema=True)
async def readiness_check(request: Request) -> JSONResponse:
    service: HealthService = request.app.state.health  # type: ignore[attr-defined]
    return JSONResponse(service.readiness())


TModel = TypeVar("TModel", bound=BaseModel)
PresenterDep = Annotated[ResponsePresenter, Depends(get_response_presenter)]


def _apply_tenant(
    request_model: TModel,
    security: SecurityContext,
    http_request: Request | None = None,
) -> TModel:
    try:
        return apply_tenant_context(request_model, security, http_request)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/ingest/{dataset}", status_code=207, response_model=None)
async def ingest_dataset(
    dataset: str,
    request: IngestionRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/ingest")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    result: BatchOperationResult = service.ingest(dataset, request)
    meta = {"total": result.total, "dataset": dataset}
    get_audit_trail().record(
        context=security,
        action="ingest",
        resource=f"dataset:{dataset}",
        metadata={"items": len(request.items)},
    )
    return presenter.success(result.operations, status_code=207, meta=meta)


@router.post("/pipelines/ingest", status_code=207, response_model=None)
async def ingest_pipeline(
    request: PipelineIngestionRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/pipelines/ingest")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    ingest_request = IngestionRequest.model_validate(
        request.model_dump(exclude={"dataset"})
    )
    result: BatchOperationResult = service.ingest(request.dataset, ingest_request)
    meta = {
        "total": result.total,
        "dataset": request.dataset,
        "profile": request.profile,
    }
    get_audit_trail().record(
        context=security,
        action="ingest_pipeline",
        resource=f"dataset:{request.dataset}",
        metadata={
            "items": len(request.items),
            "profile": request.profile,
        },
    )
    return presenter.success(result.operations, status_code=207, meta=meta)


@router.get("/jobs/{job_id}", status_code=200, response_model=JobStatus)
async def get_job(
    job_id: str,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_READ], endpoint="GET /v1/jobs/{job_id}")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    job = service.get_job(job_id, tenant_id=security.tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return presenter.success(job)


@router.get("/jobs/{job_id}/events", status_code=200)
async def list_job_events(
    job_id: str,
    *,
    since: datetime | None = Query(
        None,
        description="Return events emitted after this timestamp",
        alias="since",
    ),
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_READ], endpoint="GET /v1/jobs/{job_id}/events")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    job = service.get_job(job_id, tenant_id=security.tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    events = service.events.history(job_id, since=since)
    data = [
        {
            "job_id": event.job_id,
            "type": event.type,
            "payload": event.payload,
            "emitted_at": event.emitted_at.isoformat(),
        }
        for event in events
    ]
    meta = {"job_id": job_id, "count": len(data)}
    if since is not None:
        meta["since"] = since.isoformat()
    return presenter.success(data, meta=meta)


@router.get("/jobs", status_code=200, response_model=list[JobStatus])
async def list_jobs(
    status: str | None = None,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_READ], endpoint="GET /v1/jobs")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    jobs = service.list_jobs(status=status, tenant_id=security.tenant_id)
    meta = {"total": len(jobs), "status": status}
    return presenter.success(jobs, meta=meta)


@router.post("/jobs/{job_id}/cancel", status_code=202, response_model=JobStatus)
async def cancel_job(
    job_id: str,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_WRITE], endpoint="POST /v1/jobs/{job_id}/cancel")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    job = service.cancel_job(job_id, tenant_id=security.tenant_id, reason="client-request")
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    get_audit_trail().record(
        context=security,
        action="cancel_job",
        resource=f"job:{job_id}",
        metadata={"reason": "client-request"},
    )
    return presenter.success(job, status_code=202)


@router.post("/ingest/clinicaltrials", status_code=207, include_in_schema=False)
async def ingest_clinicaltrials(
    request: IngestionRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/ingest/clinicaltrials")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    return await ingest_dataset(
        "clinicaltrials",
        request,
        http_request,
        security,
        service,
        presenter,
    )


@router.post("/ingest/dailymed", status_code=207, include_in_schema=False)
async def ingest_dailymed(
    request: IngestionRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/ingest/dailymed")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    return await ingest_dataset("dailymed", request, http_request, security, service, presenter)


@router.post("/ingest/pmc", status_code=207, include_in_schema=False)
async def ingest_pmc(
    request: IngestionRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/ingest/pmc")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    return await ingest_dataset("pmc", request, http_request, security, service, presenter)


@router.post("/chunk", status_code=200)
async def chunk_document(
    request: ChunkRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/chunk")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    chunks = service.chunk_document(request)
    meta = {"total": len(chunks), "document_id": request.document_id}
    get_audit_trail().record(
        context=security,
        action="chunk",
        resource=f"document:{request.document_id}",
        metadata={"chunks": len(chunks)},
    )
    return presenter.success(chunks, meta=meta)


@router.post("/embed", status_code=200)
async def embed_text(
    request: EmbedRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EMBED_WRITE], endpoint="POST /v1/embed")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    response = service.embed(request)
    meta = {
        "total": len(response.embeddings),
        "namespace": response.namespace,
        "provider": response.metadata.provider,
        "model": response.metadata.model,
    }
    get_audit_trail().record(
        context=security,
        action="embed",
        resource="embedding",
        metadata={
            "inputs": len(request.texts),
            "namespace": response.namespace,
            "provider": response.metadata.provider,
        },
    )
    return presenter.success(response, meta=meta)


@router.post("/retrieve", status_code=200)
async def retrieve(
    request: RetrieveRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.RETRIEVE_READ], endpoint="POST /v1/retrieve")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    odata = ODataParams.from_request(http_request)
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    result: RetrievalResult = service.retrieve(request)
    meta = {
        "total": result.total,
        "select": odata.select,
        "expand": odata.expand,
        "rerank": result.rerank_metrics,
        "pipeline_version": result.pipeline_version,
        "partial": result.partial,
        "degraded": result.degraded,
        "stage_timings": result.stage_timings,
        "errors": [error.model_dump(mode="json") for error in result.errors],
    }
    return presenter.success(result, meta=meta)


@router.get("/namespaces", status_code=200)
async def list_namespaces(
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EMBED_READ], endpoint="GET /v1/namespaces")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    http_request.state.requested_tenant_id = security.tenant_id
    namespaces = service.list_namespaces(tenant_id=security.tenant_id, scope=Scopes.EMBED_READ)
    return presenter.success(namespaces, meta={"total": len(namespaces)})


@router.post("/namespaces/{namespace}/validate", status_code=200)
async def validate_namespace(
    namespace: str,
    request: NamespaceValidationRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EMBED_READ], endpoint="POST /v1/namespaces/{namespace}/validate")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    result = service.validate_namespace_texts(
        tenant_id=request.tenant_id,
        namespace=namespace,
        texts=request.texts,
    )
    return presenter.success(result)


@router.post("/evaluate", status_code=200)
async def evaluate(
    request: EvaluationRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EVALUATE_WRITE], endpoint="POST /v1/evaluate")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    try:
        result = service.evaluate_retrieval(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = EvaluationResponse.from_result(result)
    meta = {"cache": response.cache, "test_set_version": response.test_set_version}
    return presenter.success(response, meta=meta)


@router.post("/pipelines/query", status_code=200)
async def query_pipeline(
    request: PipelineQueryRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.RETRIEVE_READ], endpoint="POST /v1/pipelines/query")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    odata = ODataParams.from_request(http_request)
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    result: RetrievalResult = service.retrieve(request)
    meta = {
        "total": result.total,
        "select": odata.select,
        "expand": odata.expand,
        "rerank": result.rerank_metrics,
        "pipeline_version": result.pipeline_version,
        "partial": result.partial,
        "degraded": result.degraded,
        "stage_timings": result.stage_timings,
        "errors": [error.model_dump(mode="json") for error in result.errors],
        "profile": request.profile,
    }
    return presenter.success(result, meta=meta)


@router.get("/search", status_code=200)
async def search(
    http_request: Request,
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    rerank: bool = Query(True),
    rerank_model: str | None = Query(default=None, min_length=1, max_length=128),
    query_intent: QueryIntent | None = Query(default=None),
    table_only: bool = Query(False),
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.RETRIEVE_READ], endpoint="GET /v1/search")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    request_model = RetrieveRequest(
        tenant_id=security.tenant_id,
        query=query,
        top_k=top_k,
        filters={},
        rerank=rerank,
        rerank_model=rerank_model,
        query_intent=query_intent,
        table_only=table_only,
    )
    result: RetrievalResult = service.retrieve(request_model)
    odata = ODataParams.from_request(http_request)
    meta = {
        "total": result.total,
        "select": odata.select,
        "expand": odata.expand,
        "rerank": result.rerank_metrics,
        "pipeline_version": result.pipeline_version,
        "partial": result.partial,
        "degraded": result.degraded,
        "stage_timings": result.stage_timings,
        "intent": result.intent,
        "errors": [error.model_dump(mode="json") for error in result.errors],
    }
    return presenter.success(result, meta=meta)


@router.post("/map/el", status_code=207)
async def entity_link(
    request: EntityLinkRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.PROCESS_WRITE], endpoint="POST /v1/map/el")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    results = service.entity_link(request)
    meta = {"total": len(results)}
    get_audit_trail().record(
        context=security,
        action="entity_link",
        resource="entity_link",
        metadata={"mentions": len(request.mentions)},
    )
    return presenter.success(results, status_code=207, meta=meta)


@router.post("/extract/{kind}", status_code=200)
async def extract(
    *,
    kind: str = Path(pattern="^(pico|effects|ae|dose|eligibility)$"),
    request: ExtractionRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.PROCESS_WRITE], endpoint="POST /v1/extract")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    extraction = service.extract(kind, request)
    get_audit_trail().record(
        context=security,
        action="extract",
        resource=f"extract:{kind}",
        metadata={"document_id": request.document_id},
    )
    return presenter.success(extraction)


@router.post("/kg/write", status_code=200)
async def kg_write(
    request: KnowledgeGraphWriteRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.KG_WRITE], endpoint="POST /v1/kg/write")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: PresenterDep,
) -> JSONResponse:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    result = service.write_kg(request)
    get_audit_trail().record(
        context=security,
        action="kg_write",
        resource="knowledge_graph",
        metadata={"nodes": len(request.nodes), "edges": len(request.edges)},
    )
    return presenter.success(result)


@router.get("/audit/logs", status_code=200)
async def list_audit_logs(
    limit: int = 50,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.AUDIT_READ], endpoint="GET /v1/audit/logs")
    ),
    presenter: PresenterDep,
) -> JSONResponse:
    logs = get_audit_trail().list(tenant_id=security.tenant_id, limit=limit)
    data = [
        {
            "timestamp": entry.timestamp.isoformat(),
            "tenant_id": entry.tenant_id,
            "subject": entry.subject,
            "action": entry.action,
            "resource": entry.resource,
            "metadata": entry.metadata,
        }
        for entry in logs
    ]
    return presenter.success(data, meta={"total": len(data)})
