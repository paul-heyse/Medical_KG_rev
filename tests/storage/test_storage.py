import asyncio

import pytest

from Medical_KG_rev.storage.cache import InMemoryCache
from Medical_KG_rev.storage.ledger import InMemoryLedger
from Medical_KG_rev.storage.object_store import InMemoryObjectStore


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeS3Client:
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes, Metadata: dict | None = None) -> None:  # noqa: N803
        self._store[Key] = Body

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        return {"Body": _FakeBody(self._store[Key])}

    def delete_object(self, Bucket: str, Key: str) -> None:  # noqa: N803
        self._store.pop(Key, None)


@pytest.mark.anyio("asyncio")
async def test_in_memory_object_store_roundtrip():
    store = InMemoryObjectStore()
    await store.put("key", b"value")
    data = await store.get("key")
    assert data == b"value"
    await store.delete("key")
    with pytest.raises(Exception):
        await store.get("key")


@pytest.mark.anyio("asyncio")
async def test_in_memory_cache():
    cache = InMemoryCache()
    await cache.set("key", b"value", ttl=1)
    assert await cache.get("key") == b"value"
    await asyncio.sleep(1.1)
    assert await cache.get("key") is None
    await cache.delete("key")
    assert await cache.get("key") is None


@pytest.mark.anyio("asyncio")
async def test_in_memory_ledger():
    ledger = InMemoryLedger()
    await ledger.record_state("job1", {"status": "running"})
    state = await ledger.get_state("job1")
    assert state["status"] == "running"


@pytest.mark.anyio("asyncio")
async def test_s3_object_store_with_fake_client(monkeypatch):
    from Medical_KG_rev.storage import object_store as obj_module

    fake_client = _FakeS3Client()
    monkeypatch.setattr(obj_module, "boto3", type("Boto", (), {"client": staticmethod(lambda service: fake_client)}))

    store = obj_module.S3ObjectStore(bucket="bucket")
    await store.put("key", b"value")
    data = await store.get("key")
    assert data == b"value"
    await store.delete("key")
