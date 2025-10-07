"""Server-Sent Event endpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ...auth import Scopes, SecurityContext, secure_endpoint
from ..models import JobEvent
from ..services import GatewayService, get_gateway_service

router = APIRouter(prefix="/v1", tags=["sse"])


def _format_event(event: JobEvent) -> str:
    payload = event.model_dump(mode="json")
    payload.setdefault("payload", event.payload)
    payload["emitted_at"] = event.emitted_at.isoformat()
    return "".join(
        [
            f"id: {event.job_id}\n",
            f"event: {event.type}\n",
            f"data: {json.dumps(payload)}\n\n",
        ]
    )


async def _event_iterator(service: GatewayService, job_id: str) -> AsyncIterator[bytes]:
    async for event in service.events.subscribe(job_id):
        yield _format_event(event).encode("utf-8")


@router.get("/jobs/{job_id}/events", response_class=StreamingResponse)
async def stream_job_events(
    job_id: str,
    service: GatewayService = Depends(get_gateway_service),
    security: SecurityContext = Depends(
        secure_endpoint(
            scopes=[Scopes.JOBS_READ], endpoint="GET /v1/jobs/{job_id}/events/stream"
        )
    ),
) -> StreamingResponse:
    job = service.get_job(job_id, tenant_id=security.tenant_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return StreamingResponse(
        _event_iterator(service, job_id),
        status_code=status.HTTP_200_OK,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
