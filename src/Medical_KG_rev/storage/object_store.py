"""Object store implementations."""

from __future__ import annotations

import asyncio

try:  # Optional dependency
    import boto3
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore

from .base import ObjectStore, StorageError


class InMemoryObjectStore(ObjectStore):
    """Simple in-memory object store used for testing."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._lock = asyncio.Lock()

    async def put(self, key: str, data: bytes, *, metadata: dict[str, str] | None = None) -> None:
        async with self._lock:
            self._data[key] = data

    async def get(self, key: str) -> bytes:
        async with self._lock:
            try:
                return self._data[key]
            except KeyError as exc:
                raise StorageError(f"Object '{key}' not found") from exc

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)


class S3ObjectStore(ObjectStore):
    """S3/MinIO backed object store."""

    def __init__(self, bucket: str, *, client: boto3.client | None = None) -> None:
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3ObjectStore")
        self._bucket = bucket
        self._client = client or boto3.client("s3")

    async def put(self, key: str, data: bytes, *, metadata: dict[str, str] | None = None) -> None:
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
            Metadata=metadata or {},
        )

    async def get(self, key: str) -> bytes:
        response = await asyncio.to_thread(self._client.get_object, Bucket=self._bucket, Key=key)
        stream = response["Body"]
        data = stream.read()
        return data

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._client.delete_object, Bucket=self._bucket, Key=key)
