"""REST API router exposing gateway operations."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, TypeVar, cast

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...auth import Scopes, SecurityContext, secure_endpoint
from ...auth.audit import get_audit_trail
from ...services.health import HealthService
from ..models import (
    BatchOperationResult,
    ChunkRequest,
    EmbedRequest,
    EntityLinkRequest,
    ExtractionRequest,
    IngestionRequest,
    JobStatus,
    KnowledgeGraphWriteRequest,
    RetrievalResult,
    RetrieveRequest,
)
from ..services import GatewayService, get_gateway_service

router = APIRouter(prefix="/v1", tags=["gateway"])
health_router = APIRouter(tags=["system"])


class ODataParams(BaseModel):
    select: list[str] | None = None
    expand: list[str] | None = None
    filter: str | None = Field(default=None, alias="$filter")
    top: int | None = Field(default=None, alias="$top")
    skip: int | None = Field(default=None, alias="$skip")

    @classmethod
    def from_request(cls, request: Request) -> ODataParams:
        params: dict[str, Any] = {}
        qp = request.query_params
        if "$select" in qp:
            params["select"] = [
                value.strip() for value in qp["$select"].split(",") if value.strip()
            ]
        if "$expand" in qp:
            params["expand"] = [
                value.strip() for value in qp["$expand"].split(",") if value.strip()
            ]
        if "$filter" in qp:
            params["$filter"] = qp["$filter"]
        if "$top" in qp:
            params["$top"] = int(qp["$top"])
        if "$skip" in qp:
            params["$skip"] = int(qp["$skip"])
        return cls.model_validate(params)


JSONAPI_CONTENT_TYPE = "application/vnd.api+json"


def _normalise_payload(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, dict)):
        return [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in data
        ]
    return data


def json_api_payload(data: Any, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"data": _normalise_payload(data), "meta": meta or {}}


def json_api_response(
    data: Any,
    *,
    status_code: int = 200,
    meta: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        json_api_payload(data, meta=meta),
        status_code=status_code,
        media_type=JSONAPI_CONTENT_TYPE,
    )


@health_router.get("/health", include_in_schema=True)
async def health_check(request: Request) -> JSONResponse:
    service: HealthService = request.app.state.health  # type: ignore[attr-defined]
    return JSONResponse(service.liveness())


@health_router.get("/ready", include_in_schema=True)
async def readiness_check(request: Request) -> JSONResponse:
    service: HealthService = request.app.state.health  # type: ignore[attr-defined]
    return JSONResponse(service.readiness())


TModel = TypeVar("TModel", bound=BaseModel)


def _ensure_tenant(request_model: TModel, security: SecurityContext) -> TModel:
    tenant_id = getattr(request_model, "tenant_id", None)
    if tenant_id and tenant_id != security.tenant_id:
        raise HTTPException(status_code=403, detail="Tenant mismatch")
    return cast(TModel, request_model.model_copy(update={"tenant_id": security.tenant_id}))


@router.post("/ingest/{dataset}", status_code=207, response_model=None)
async def ingest_dataset(
    dataset: str,
    request: IngestionRequest,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/ingest")
    ),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    request = _ensure_tenant(request, security)  # type: ignore[assignment]
    result: BatchOperationResult = service.ingest(dataset, request)
    meta = {"total": result.total, "dataset": dataset}
    get_audit_trail().record(
        context=security,
        action="ingest",
        resource=f"dataset:{dataset}",
        metadata={"items": len(request.items)},
    )
    return json_api_response(result.operations, status_code=207, meta=meta)


@router.get("/jobs/{job_id}", status_code=200, response_model=JobStatus)
async def get_job(
    job_id: str,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_READ], endpoint="GET /v1/jobs/{job_id}")
    ),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    job = service.get_job(job_id, tenant_id=security.tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return json_api_response(job)


@router.get("/jobs/{job_id}/events", status_code=200)
async def list_job_events(
    job_id: str,
    *,
    since: datetime = Query(
        ..., description="Return events emitted after this timestamp", alias="since"
    ),
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_READ], endpoint="GET /v1/jobs/{job_id}/events")
    ),
) -> JSONResponse:
    meta = {"job_id": job_id, "since": since.isoformat()}
    return json_api_response([], meta=meta)


@router.get("/jobs", status_code=200, response_model=list[JobStatus])
async def list_jobs(
    status: str | None = None,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_READ], endpoint="GET /v1/jobs")
    ),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    jobs = service.list_jobs(status=status, tenant_id=security.tenant_id)
    meta = {"total": len(jobs), "status": status}
    return json_api_response(jobs, meta=meta)


@router.post("/jobs/{job_id}/cancel", status_code=202, response_model=JobStatus)
async def cancel_job(
    job_id: str,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.JOBS_WRITE], endpoint="POST /v1/jobs/{job_id}/cancel")
    ),
    service: GatewayService = Depends(get_gateway_service),
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
    return json_api_response(job, status_code=202)


@router.post("/ingest/clinicaltrials", status_code=207, include_in_schema=False)
async def ingest_clinicaltrials(
    request: IngestionRequest,
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    return await ingest_dataset("clinicaltrials", request, service)


@router.post("/ingest/dailymed", status_code=207, include_in_schema=False)
async def ingest_dailymed(
    request: IngestionRequest,
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    return await ingest_dataset("dailymed", request, service)


@router.post("/ingest/pmc", status_code=207, include_in_schema=False)
async def ingest_pmc(
    request: IngestionRequest,
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    return await ingest_dataset("pmc", request, service)


@router.post("/chunk", status_code=200)
async def chunk_document(
    request: ChunkRequest,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.INGEST_WRITE], endpoint="POST /v1/chunk")
    ),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    request = _ensure_tenant(request, security)  # type: ignore[assignment]
    chunks = service.chunk_document(request)
    meta = {"total": len(chunks), "document_id": request.document_id}
    get_audit_trail().record(
        context=security,
        action="chunk",
        resource=f"document:{request.document_id}",
        metadata={"chunks": len(chunks)},
    )
    return json_api_response(chunks, meta=meta)


@router.post("/embed", status_code=200)
async def embed_text(
    request: EmbedRequest,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.EMBED_WRITE], endpoint="POST /v1/embed")
    ),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    request = _ensure_tenant(request, security)  # type: ignore[assignment]
    embeddings = service.embed(request)
    meta = {"total": len(embeddings), "model": request.model}
    get_audit_trail().record(
        context=security,
        action="embed",
        resource="embedding",
        metadata={"inputs": len(request.inputs), "model": request.model},
    )
    return json_api_response(embeddings, meta=meta)


@router.post("/retrieve", status_code=200)
async def retrieve(
    request: RetrieveRequest,
    http_request: Request,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.RETRIEVE_READ], endpoint="POST /v1/retrieve")
    ),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    odata = ODataParams.from_request(http_request)
    request = _ensure_tenant(request, security)  # type: ignore[assignment]
    result: RetrievalResult = service.retrieve(request)
    meta = {
        "total": result.total,
        "select": odata.select,
        "expand": odata.expand,
        "rerank": result.rerank_metrics,
    }
    return json_api_response(result, meta=meta)


@router.get("/search", status_code=200)
async def search(
    http_request: Request,
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    rerank: bool = Query(True),
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.RETRIEVE_READ], endpoint="GET /v1/search")
    ),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    request_model = RetrieveRequest(
        tenant_id=security.tenant_id,
        query=query,
        top_k=top_k,
        filters={},
        rerank=rerank,
    )
    result: RetrievalResult = service.retrieve(request_model)
    odata = ODataParams.from_request(http_request)
    meta = {
        "total": result.total,
        "select": odata.select,
        "expand": odata.expand,
        "rerank": result.rerank_metrics,
    }
    return json_api_response(result, meta=meta)


@router.post("/map/el", status_code=207)
async def entity_link(
    request: EntityLinkRequest,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.PROCESS_WRITE], endpoint="POST /v1/map/el")
    ),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    request = _ensure_tenant(request, security)  # type: ignore[assignment]
    results = service.entity_link(request)
    meta = {"total": len(results)}
    get_audit_trail().record(
        context=security,
        action="entity_link",
        resource="entity_link",
        metadata={"mentions": len(request.mentions)},
    )
    return json_api_response(results, status_code=207, meta=meta)


@router.post("/extract/{kind}", status_code=200)
async def extract(
    *,
    kind: str = Path(pattern="^(pico|effects|ae|dose|eligibility)$"),
    request: ExtractionRequest,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.PROCESS_WRITE], endpoint="POST /v1/extract")
    ),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    request = _ensure_tenant(request, security)  # type: ignore[assignment]
    extraction = service.extract(kind, request)
    get_audit_trail().record(
        context=security,
        action="extract",
        resource=f"extract:{kind}",
        metadata={"document_id": request.document_id},
    )
    return json_api_response(extraction)


@router.post("/kg/write", status_code=200)
async def kg_write(
    request: KnowledgeGraphWriteRequest,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.KG_WRITE], endpoint="POST /v1/kg/write")
    ),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    request = _ensure_tenant(request, security)  # type: ignore[assignment]
    result = service.write_kg(request)
    get_audit_trail().record(
        context=security,
        action="kg_write",
        resource="knowledge_graph",
        metadata={"nodes": len(request.nodes), "edges": len(request.edges)},
    )
    return json_api_response(result)


@router.get("/audit/logs", status_code=200)
async def list_audit_logs(
    limit: int = 50,
    security: SecurityContext = Depends(
        secure_endpoint(scopes=[Scopes.AUDIT_READ], endpoint="GET /v1/audit/logs")
    ),
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
    return json_api_response(data, meta={"total": len(data)})
