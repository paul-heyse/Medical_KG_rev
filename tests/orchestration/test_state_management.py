from __future__ import annotations

import time
from typing import Any

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.orchestration.state import (
    PipelineStateCache,
    PipelineStatePersister,
    StatePersistenceError,
    dumps_json,
    encode_base64,
    serialise_payload,
)


class _StubMetadataStore:
    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.payloads: dict[str, dict[str, Any]] = {}

    def update_metadata(self, job_id: str, payload: dict[str, Any]) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("transient failure")
        self.payloads.setdefault(job_id, {}).update(payload)


def test_cache_expiration() -> None:
    cache = PipelineStateCache(ttl_seconds=0.1)
    cache.set("key", b"value")
    assert cache.get("key") == b"value"
    time.sleep(0.2)
    assert cache.get("key") is None


def test_serialisation_helpers_validate_payload() -> None:
    payload = {
        "version": "v1",
        "job_id": None,
        "context": {
            "tenant_id": "tenant",
            "metadata": {},
        },
        "adapter_request": {},
        "payload": {},
        "payload_count": 0,
        "chunk_count": 0,
        "embedding_count": 0,
        "entity_count": 0,
        "claim_count": 0,
        "metadata": {},
    }
    model = serialise_payload(payload)
    assert model.version == "v1"
    assert dumps_json(payload)
    blob = encode_base64(b"{}")
    assert isinstance(blob, str)


def test_persister_retries_and_persists() -> None:
    store = _StubMetadataStore(fail_times=1)
    persister = PipelineStatePersister(metadata_store=store, max_attempts=3)
    snapshot = persister.persist("job", stage="ingest", payload={"version": "v1"})
    assert store.payloads["job"].get("state.ingest.snapshot") == snapshot


def test_persister_raises_after_exhausting_retries() -> None:
    store = _StubMetadataStore(fail_times=5)
    persister = PipelineStatePersister(metadata_store=store, max_attempts=2)
    with pytest.raises(StatePersistenceError):
        persister.persist("job", stage="chunk", payload={"version": "v1"})
