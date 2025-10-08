"""Object store implementations and MinerU figure helpers."""

from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Iterable

try:  # Optional dependency
    import boto3
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore

from .base import ObjectStore, StorageError


class InMemoryObjectStore(ObjectStore):
    """Simple in-memory object store used for testing."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._metadata: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()

    async def put(self, key: str, data: bytes, *, metadata: dict[str, str] | None = None) -> None:
        async with self._lock:
            self._data[key] = data
            if metadata:
                self._metadata[key] = dict(metadata)
            elif key in self._metadata:
                del self._metadata[key]

    async def get(self, key: str) -> bytes:
        async with self._lock:
            try:
                return self._data[key]
            except KeyError as exc:
                raise StorageError(f"Object '{key}' not found") from exc

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)
            self._metadata.pop(key, None)

    async def list_prefix(self, prefix: str) -> list[str]:
        async with self._lock:
            return [key for key in self._data if key.startswith(prefix)]

    async def get_metadata(self, key: str) -> dict[str, str] | None:
        async with self._lock:
            return dict(self._metadata.get(key, {}))


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

    async def list_prefix(self, prefix: str) -> list[str]:
        def _list() -> list[str]:
            paginator = self._client.get_paginator("list_objects_v2")
            keys: list[str] = []
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = item.get("Key")
                    if key:
                        keys.append(str(key))
            return keys

        return await asyncio.to_thread(_list)

    def generate_presigned_url(self, key: str, *, expires_in: int = 3600) -> str:
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception:
            return f"https://{self._bucket}.s3.amazonaws.com/{key}"

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def client(self):  # pragma: no cover - convenience accessor
        return self._client


class FileSystemObjectStore(ObjectStore):
    """Persist objects on the local filesystem.

    This backend is primarily intended for development and integration testing
    environments where an S3-compatible service is not available. Objects are
    written relative to a base directory that mirrors the logical object key
    hierarchy.
    """

    def __init__(self, base_path: str | Path) -> None:
        path = Path(base_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        self._base_path = path
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._metadata_suffix = ".meta.json"

    async def put(self, key: str, data: bytes, *, metadata: dict[str, str] | None = None) -> None:
        target = self._base_path / key
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)
        if metadata:
            meta_path = target.with_suffix(target.suffix + self._metadata_suffix)
            payload = json.dumps({str(k): str(v) for k, v in metadata.items()}, sort_keys=True)
            await asyncio.to_thread(meta_path.write_text, payload, encoding="utf-8")
        else:
            meta_path = target.with_suffix(target.suffix + self._metadata_suffix)
            if meta_path.exists():
                await asyncio.to_thread(meta_path.unlink)

    async def get(self, key: str) -> bytes:
        target = self._base_path / key
        if not target.exists():
            raise StorageError(f"Object '{key}' not found")
        return await asyncio.to_thread(target.read_bytes)

    async def delete(self, key: str) -> None:
        target = self._base_path / key
        await asyncio.to_thread(target.unlink, missing_ok=True)
        meta_path = target.with_suffix(target.suffix + self._metadata_suffix)
        if meta_path.exists():
            await asyncio.to_thread(meta_path.unlink, missing_ok=True)

    async def list_prefix(self, prefix: str) -> list[str]:
        base = self._base_path / prefix
        if not base.exists():
            return []
        keys: list[str] = []
        for file_path in base.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith(self._metadata_suffix):
                keys.append(str(file_path.relative_to(self._base_path)))
        return keys


class FigureStorageClient:
    """Synchronous helper that stores figures using the configured backend."""

    def __init__(
        self,
        backend: ObjectStore,
        *,
        base_prefix: str = "mineru",
        signed_url_ttl: int = 3600,
        default_content_type: str = "image/png",
    ) -> None:
        self._backend = backend
        self._base_prefix = base_prefix.strip("/") or "mineru"
        self._signed_url_ttl = signed_url_ttl
        self._default_content_type = default_content_type

    def store_figure(
        self,
        tenant_id: str,
        document_id: str,
        figure_id: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        key = self._figure_key(tenant_id, document_id, figure_id, content_type)
        meta = {
            "content-type": content_type or self._default_content_type,
            "tenant-id": tenant_id,
            "document-id": document_id,
            "figure-id": figure_id,
        }
        if metadata:
            meta.update({str(k): str(v) for k, v in metadata.items()})
        self._run(self._backend.put(key, data, metadata=meta))
        return key

    def generate_figure_url(
        self,
        tenant_id: str,
        document_id: str,
        figure_id: str,
        *,
        key: str | None = None,
        expires_in: int | None = None,
    ) -> str:
        resolved_key = key or self._figure_key(tenant_id, document_id, figure_id, None)
        expiry = expires_in or self._signed_url_ttl
        if isinstance(self._backend, S3ObjectStore):
            return self._backend.generate_presigned_url(resolved_key, expires_in=expiry)
        return f"inmemory://{resolved_key}?ttl={expiry}"

    def cleanup_document(self, tenant_id: str, document_id: str) -> None:
        prefix = self._document_prefix(tenant_id, document_id)
        for key in self._run(self._list_keys(prefix)):
            self._run(self._backend.delete(key))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _document_prefix(self, tenant_id: str, document_id: str) -> str:
        tenant_segment = self._safe_segment(tenant_id)
        doc_segment = self._safe_segment(document_id)
        return f"{self._base_prefix}/{tenant_segment}/documents/{doc_segment}/figures/"

    def _figure_key(
        self,
        tenant_id: str,
        document_id: str,
        figure_id: str,
        content_type: str | None,
    ) -> str:
        prefix = self._document_prefix(tenant_id, document_id)
        extension = self._extension_for_content(content_type)
        figure_segment = self._safe_segment(figure_id)
        return f"{prefix}{figure_segment}{extension}"

    def _extension_for_content(self, content_type: str | None) -> str:
        if not content_type:
            return ""
        ext = mimetypes.guess_extension(content_type) or ""
        if ext == ".jpe":
            return ".jpg"
        return ext

    async def _list_keys(self, prefix: str) -> Iterable[str]:
        if hasattr(self._backend, "list_prefix"):
            return await getattr(self._backend, "list_prefix")(prefix)
        return []

    def _safe_segment(self, value: str) -> str:
        cleaned = [ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value]
        segment = "".join(cleaned).strip("-")
        return segment or "unknown"

    def _run(self, awaitable):
        try:
            return asyncio.run(awaitable)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(awaitable)
            finally:
                asyncio.set_event_loop(None)
                loop.close()


__all__ = [
    "FileSystemObjectStore",
    "FigureStorageClient",
    "InMemoryObjectStore",
    "S3ObjectStore",
]
