from __future__ import annotations

import asyncio

import pytest

from Medical_KG_rev.gateway.models import JobEvent
from Medical_KG_rev.gateway.services import get_gateway_service


@pytest.mark.anyio("asyncio")
async def test_event_stream_manager() -> None:
    service = get_gateway_service()
    job_id = "job-test"
    stream = service.events.subscribe(job_id)
    task = asyncio.create_task(stream.__anext__())
    service.events.publish(JobEvent(job_id=job_id, type="jobs.started", payload={}))
    event = await asyncio.wait_for(task, timeout=1)
    assert event.job_id == job_id
    assert event.type == "jobs.started"
    pending = asyncio.create_task(stream.__anext__())
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
