"""REST API router exposing gateway operations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, Depends, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..models import (
    BatchOperationResult,
    ChunkRequest,
    EmbedRequest,
    EntityLinkRequest,
    ExtractionRequest,
    IngestionRequest,
    KnowledgeGraphWriteRequest,
    RetrievalResult,
    RetrieveRequest,
)
from ..services import GatewayService, get_gateway_service

router = APIRouter(prefix="/v1", tags=["gateway"])


class ODataParams(BaseModel):
    select: Optional[List[str]] = None
    expand: Optional[List[str]] = None
    filter: Optional[str] = Field(default=None, alias="$filter")
    top: Optional[int] = Field(default=None, alias="$top")
    skip: Optional[int] = Field(default=None, alias="$skip")

    @classmethod
    def from_request(cls, request: Request) -> "ODataParams":
        params: Dict[str, Any] = {}
        qp = request.query_params
        if "$select" in qp:
            params["select"] = [value.strip() for value in qp["$select"].split(",") if value.strip()]
        if "$expand" in qp:
            params["expand"] = [value.strip() for value in qp["$expand"].split(",") if value.strip()]
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
        return [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in data]
    return data


def json_api_payload(data: Any, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"data": _normalise_payload(data), "meta": meta or {}}


def json_api_response(
    data: Any,
    *,
    status_code: int = 200,
    meta: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    return JSONResponse(
        json_api_payload(data, meta=meta),
        status_code=status_code,
        media_type=JSONAPI_CONTENT_TYPE,
    )


@router.post("/ingest/{dataset}", status_code=207, response_model=None)
async def ingest_dataset(
    dataset: str,
    request: IngestionRequest,
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    result: BatchOperationResult = service.ingest(dataset, request)
    meta = {"total": result.total, "dataset": dataset}
    return json_api_response(result.operations, status_code=207, meta=meta)


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
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    chunks = service.chunk_document(request)
    meta = {"total": len(chunks), "document_id": request.document_id}
    return json_api_response(chunks, meta=meta)


@router.post("/embed", status_code=200)
async def embed_text(
    request: EmbedRequest,
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    embeddings = service.embed(request)
    meta = {"total": len(embeddings), "model": request.model}
    return json_api_response(embeddings, meta=meta)


@router.post("/retrieve", status_code=200)
async def retrieve(
    request: RetrieveRequest,
    http_request: Request,
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    odata = ODataParams.from_request(http_request)
    result: RetrievalResult = service.retrieve(request)
    meta = {"total": result.total, "select": odata.select, "expand": odata.expand}
    return json_api_response(result, meta=meta)


@router.post("/map/el", status_code=207)
async def entity_link(
    request: EntityLinkRequest,
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    results = service.entity_link(request)
    meta = {"total": len(results)}
    return json_api_response(results, status_code=207, meta=meta)


@router.post("/extract/{kind}", status_code=200)
async def extract(
    *,
    kind: str = Path(pattern="^(pico|effects|ae|dose|eligibility)$"),
    request: ExtractionRequest,
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    extraction = service.extract(kind, request)
    return json_api_response(extraction)


@router.post("/kg/write", status_code=200)
async def kg_write(
    request: KnowledgeGraphWriteRequest,
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    result = service.write_kg(request)
    return json_api_response(result)
