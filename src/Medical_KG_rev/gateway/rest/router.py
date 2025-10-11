"""REST API router exposing gateway operations.

This module provides the REST API endpoints for the gateway application,
implementing HTTP-based access to all gateway operations including ingestion,
retrieval, embedding, evaluation, and adapter management.

Key Responsibilities:
    - REST endpoint definitions for all gateway operations
    - Request/response serialization and validation
    - Authentication and authorization enforcement
    - Error handling and status code management
    - OpenAPI documentation generation
    - Content negotiation and response formatting

Collaborators:
    - Upstream: HTTP clients, API documentation tools
    - Downstream: Gateway services, coordinators, domain services

Side Effects:
    - Processes HTTP requests and responses
    - Validates authentication tokens
    - Logs API access and audit trails
    - Emits metrics for API usage

Thread Safety:
    - Thread-safe: FastAPI handles concurrent requests
    - Stateless: No shared mutable state between requests

Performance Characteristics:
    - O(1) request routing overhead
    - O(n) serialization where n is response size
    - Rate limited by middleware

Example:
    >>> from fastapi import FastAPI
    >>> from Medical_KG_rev.gateway.rest.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

from datetime import datetime
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================
from Medical_KG_rev.services.retrieval.routing import QueryIntent

from ...auth import Scopes, SecurityContext, secure_endpoint
from ...auth.audit import get_audit_trail
from ...services.health import HealthService
from ..models import (
    BatchOperationResult,
    ChunkRequest,
    EmbedRequest,
    EntityLinkRequest,
    EvaluationRequest,
    EvaluationResponse,
    ExtractionRequest,
    IngestionRequest,
    JobStatus,
    KnowledgeGraphWriteRequest,
    NamespacePolicyInvalidateRequest,
    NamespacePolicyUpdateRequest,
    NamespaceValidationRequest,
    PipelineIngestionRequest,
    PipelineQueryRequest,
    RetrievalResult,
    RetrieveRequest,
)
from ..presentation.dependencies import get_request_lifecycle, get_response_presenter
from ..presentation.errors import ErrorDetail
from ..presentation.interface import ResponsePresenter
from ..presentation.lifecycle import RequestLifecycle
from ..presentation.odata import ODataParams
from ..presentation.requests import apply_tenant_context
from ..services import GatewayService, get_gateway_service

# Type aliases for dependency injection
PresenterDep = Annotated[ResponsePresenter, Depends(get_response_presenter)]
LifecycleDep = Annotated[RequestLifecycle, Depends(get_request_lifecycle)]

# Router instances
router = APIRouter(prefix="/v1", tags=["gateway"])
health_router = APIRouter(tags=["system"])


# ==============================================================================
# CONSTANTS
# ==============================================================================

JSONAPI_CONTENT_TYPE = "application/vnd.api+json"


# ==============================================================================
# ADAPTER ENDPOINTS
# ==============================================================================


@router.get("/adapters", response_model=None)
async def list_adapters(
    domain: str | None = Query(default=None),
    service: GatewayService = Depends(get_gateway_service),
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.ADAPTERS_READ], endpoint="GET /v1/adapters")
    ),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
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
    return presenter.success(adapters, meta=lifecycle.meta({"total": len(adapters)}))


@router.get("/adapters/{name}/metadata", response_model=None)
async def get_adapter_metadata(
    name: str = Path(..., description="Adapter name"),
    service: GatewayService = Depends(get_gateway_service),
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.ADAPTERS_READ], endpoint="GET /v1/adapters/{name}/metadata")
    ),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
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
    return presenter.success(metadata, meta=lifecycle.meta())


@router.get("/adapters/{name}/health", response_model=None)
async def get_adapter_health(
    name: str,
    service: GatewayService = Depends(get_gateway_service),
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.ADAPTERS_READ], endpoint="GET /v1/adapters/{name}/health")
    ),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
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
    return presenter.success(health, meta=lifecycle.meta())


@router.get("/adapters/{name}/config-schema", response_model=None)
async def get_adapter_config_schema(
    name: str,
    service: GatewayService = Depends(get_gateway_service),
    security: SecurityContext = Depends(
        secure_endpoint(
            scopes=[Scopes.ADAPTERS_READ], endpoint="GET /v1/adapters/{name}/config-schema"
        )
    ),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
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
    return presenter.success(schema, meta=lifecycle.meta())


# ==============================================================================
# HEALTH ENDPOINTS
# ==============================================================================


@health_router.get("/health", include_in_schema=True)
async def health_check(request: Request) -> JSONResponse:
    service: HealthService = request.app.state.health  # type: ignore[attr-defined]
    return JSONResponse(service.liveness())


@health_router.get("/ready", include_in_schema=True)
async def readiness_check(request: Request) -> JSONResponse:
    service: HealthService = request.app.state.health  # type: ignore[attr-defined]
    return JSONResponse(service.readiness())


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

TModel = TypeVar("TModel", bound=BaseModel)


def _apply_tenant(
    request_model: TModel,
    security: SecurityContext,
    http_request: Request | None = None,
) -> TModel:
    try:
        return apply_tenant_context(request_model, security, http_request)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


# ==============================================================================
# INGESTION ENDPOINTS
# ==============================================================================


@router.post("/ingest/{dataset}", status_code=207, response_model=None)
async def ingest_dataset(
    dataset: str,
    request: IngestionRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/ingest")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    result: BatchOperationResult = service.ingest(dataset, request)
    meta = lifecycle.meta({"total": result.total, "dataset": dataset})
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
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    ingest_request = IngestionRequest.model_validate(request.model_dump(exclude={"dataset"}))
    result: BatchOperationResult = service.ingest(request.dataset, ingest_request)
    meta = lifecycle.meta(
        {
            "total": result.total,
            "dataset": request.dataset,
            "profile": request.profile,
        }
    )
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


# ==============================================================================
# JOB MANAGEMENT ENDPOINTS
# ==============================================================================


@router.get("/jobs/{job_id}", status_code=200, response_model=JobStatus)
async def get_job(
    job_id: str,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_READ], endpoint="GET /v1/jobs/{job_id}")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    job = service.get_job(job_id, tenant_id=security.tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return presenter.success(job, meta=lifecycle.meta())


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
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
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
    return presenter.success(data, meta=lifecycle.meta(meta))


@router.get("/jobs", status_code=200, response_model=list[JobStatus])
async def list_jobs(
    status: str | None = None,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_READ], endpoint="GET /v1/jobs")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    jobs = service.list_jobs(status=status, tenant_id=security.tenant_id)
    meta = lifecycle.meta({"total": len(jobs), "status": status})
    return presenter.success(jobs, meta=meta)


@router.post("/jobs/{job_id}/cancel", status_code=202, response_model=JobStatus)
async def cancel_job(
    job_id: str,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_WRITE], endpoint="POST /v1/jobs/{job_id}/cancel")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    job = service.cancel_job(job_id, tenant_id=security.tenant_id, reason="client-request")
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    get_audit_trail().record(
        context=security,
        action="cancel_job",
        resource=f"job:{job_id}",
        metadata={"reason": "client-request"},
    )
    return presenter.success(job, status_code=202, meta=lifecycle.meta())


@router.post("/ingest/clinicaltrials", status_code=207, include_in_schema=False)
async def ingest_clinicaltrials(
    request: IngestionRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/ingest/clinicaltrials")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    return await ingest_dataset(
        "clinicaltrials",
        request,
        http_request,
        security,
        service,
        presenter,
        lifecycle,
    )


@router.post("/ingest/dailymed", status_code=207, include_in_schema=False)
async def ingest_dailymed(
    request: IngestionRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/ingest/dailymed")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    return await ingest_dataset(
        "dailymed",
        request,
        http_request,
        security,
        service,
        presenter,
        lifecycle,
    )


@router.post("/ingest/pmc", status_code=207, include_in_schema=False)
async def ingest_pmc(
    request: IngestionRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/ingest/pmc")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    return await ingest_dataset(
        "pmc",
        request,
        http_request,
        security,
        service,
        presenter,
        lifecycle,
    )


# ==============================================================================
# PROCESSING ENDPOINTS
# ==============================================================================


@router.post("/chunk", status_code=200)
async def chunk_document(
    request: ChunkRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/chunk")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    chunks = service.chunk_document(request)
    meta = lifecycle.meta({"total": len(chunks), "document_id": request.document_id})
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
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    response = service.embed(request)
    meta = lifecycle.meta(
        {
            "total": len(response.embeddings),
            "namespace": response.namespace,
            "provider": response.metadata.provider,
            "model": response.metadata.model,
        }
    )
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
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    odata = ODataParams.from_request(http_request)
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    result: RetrievalResult = service.retrieve(request)
    meta = lifecycle.meta(
        {
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
    )
    return presenter.success(result, meta=meta)


# ==============================================================================
# NAMESPACE ENDPOINTS
# ==============================================================================


@router.get("/namespaces", status_code=200)
async def list_namespaces(
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EMBED_READ], endpoint="GET /v1/namespaces")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    http_request.state.requested_tenant_id = security.tenant_id
    namespaces = service.list_namespaces(tenant_id=security.tenant_id, scope=Scopes.EMBED_READ)
    return presenter.success(namespaces, meta=lifecycle.meta({"total": len(namespaces)}))


@router.get("/namespaces/policy", status_code=200)
async def get_namespace_policy(
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EMBED_ADMIN], endpoint="GET /v1/namespaces/policy")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    status = service.namespace_policy_status()
    return presenter.success(status, meta=lifecycle.meta())


@router.patch("/namespaces/policy", status_code=200)
async def update_namespace_policy(
    request: NamespacePolicyUpdateRequest,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EMBED_ADMIN], endpoint="PATCH /v1/namespaces/policy")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    try:
        status = service.update_namespace_policy(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return presenter.success(status, meta=lifecycle.meta())


@router.get("/namespaces/policy/diagnostics", status_code=200)
async def get_namespace_policy_diagnostics(
    security: SecurityContext = Depends(
        secure_endpoint(
            scopes=[Scopes.EMBED_ADMIN], endpoint="GET /v1/namespaces/policy/diagnostics"
        )
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    diagnostics = service.namespace_policy_diagnostics()
    return presenter.success(diagnostics, meta=lifecycle.meta())


@router.get("/namespaces/policy/health", status_code=200)
async def get_namespace_policy_health(
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EMBED_ADMIN], endpoint="GET /v1/namespaces/policy/health")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    health = service.namespace_policy_health()
    return presenter.success(health, meta=lifecycle.meta())


@router.get("/namespaces/policy/metrics", status_code=200)
async def get_namespace_policy_metrics(
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EMBED_ADMIN], endpoint="GET /v1/namespaces/policy/metrics")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    metrics = service.namespace_policy_metrics()
    return presenter.success(metrics, meta=lifecycle.meta())


@router.post("/namespaces/policy/cache/invalidate", status_code=202)
async def invalidate_namespace_policy_cache(
    request: NamespacePolicyInvalidateRequest,
    security: SecurityContext = Depends(
        secure_endpoint(
            scopes=[Scopes.EMBED_ADMIN], endpoint="POST /v1/namespaces/policy/cache/invalidate"
        )
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    service.invalidate_namespace_policy_cache(request.namespace)
    payload = {"status": "cache-invalidated", "namespace": request.namespace}
    return presenter.success(payload, status_code=202, meta=lifecycle.meta())


@router.post("/namespaces/{namespace}/validate", status_code=200)
async def validate_namespace(
    namespace: str,
    request: NamespaceValidationRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(
            scopes=[Scopes.EMBED_READ], endpoint="POST /v1/namespaces/{namespace}/validate"
        )
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    result = service.validate_namespace_texts(
        tenant_id=request.tenant_id,
        namespace=namespace,
        texts=request.texts,
    )
    return presenter.success(result, meta=lifecycle.meta())


@router.post("/evaluate", status_code=200)
async def evaluate(
    request: EvaluationRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EVALUATE_WRITE], endpoint="POST /v1/evaluate")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    try:
        result = service.evaluate_retrieval(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = EvaluationResponse.from_result(result)
    meta = lifecycle.meta({"cache": response.cache, "test_set_version": response.test_set_version})
    return presenter.success(response, meta=meta)


@router.post("/pipelines/query", status_code=200)
async def query_pipeline(
    request: PipelineQueryRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.RETRIEVE_READ], endpoint="POST /v1/pipelines/query")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    odata = ODataParams.from_request(http_request)
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    result: RetrievalResult = service.retrieve(request)
    meta = lifecycle.meta(
        {
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
    )
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
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
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
    meta = lifecycle.meta(
        {
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
    )
    return presenter.success(result, meta=meta)


@router.post("/map/el", status_code=207)
async def entity_link(
    request: EntityLinkRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.PROCESS_WRITE], endpoint="POST /v1/map/el")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    results = service.entity_link(request)
    meta = lifecycle.meta({"total": len(results)})
    get_audit_trail().record(
        context=security,
        action="entity_link",
        resource="entity_link",
        metadata={"mentions": len(request.mentions)},
    )
    return presenter.success(results, status_code=207, meta=meta)


# ==============================================================================
# EXTRACTION ENDPOINTS
# ==============================================================================


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
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    extraction = service.extract(kind, request)
    get_audit_trail().record(
        context=security,
        action="extract",
        resource=f"extract:{kind}",
        metadata={"document_id": request.document_id},
    )
    return presenter.success(extraction, meta=lifecycle.meta())


@router.post("/kg/write", status_code=200)
async def kg_write(
    request: KnowledgeGraphWriteRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.KG_WRITE], endpoint="POST /v1/kg/write")
    ),
    service: GatewayService = Depends(get_gateway_service),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
    request = _apply_tenant(request, security, http_request)  # type: ignore[assignment]
    result = service.write_kg(request)
    get_audit_trail().record(
        context=security,
        action="kg_write",
        resource="knowledge_graph",
        metadata={"nodes": len(request.nodes), "edges": len(request.edges)},
    )
    return presenter.success(result, meta=lifecycle.meta())


# ==============================================================================
# AUDIT ENDPOINTS
# ==============================================================================


@router.get("/audit/logs", status_code=200)
async def list_audit_logs(
    limit: int = 50,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.AUDIT_READ], endpoint="GET /v1/audit/logs")
    ),
    presenter: ResponsePresenter = Depends(get_response_presenter),
    lifecycle: RequestLifecycle = Depends(get_request_lifecycle),
) -> Response:
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
    return presenter.success(data, meta=lifecycle.meta({"total": len(data)}))


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "JSONAPI_CONTENT_TYPE",
    "LifecycleDep",
    "PresenterDep",
    "health_router",
    "router",
]
