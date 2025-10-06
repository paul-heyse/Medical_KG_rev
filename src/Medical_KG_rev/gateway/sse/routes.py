"""Server-Sent Event endpoints."""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from ..models import JobEvent
from ..services import GatewayService, get_gateway_service

router = APIRouter(prefix="/v1", tags=["sse"])


async def _authenticate(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    if x_api_key != "public-demo-key":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _format_event(event: JobEvent) -> str:
    payload = event.model_dump(mode="json")
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
    _: None = Depends(_authenticate),
) -> StreamingResponse:
    return StreamingResponse(
        _event_iterator(service, job_id),
        status_code=status.HTTP_200_OK,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
