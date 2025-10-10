from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from Medical_KG_rev.services.vector_store import (
    CompressionPolicy,
    IndexParams,
    NamespaceNotFoundError,
    VectorQuery,
    VectorRecord,
)
from Medical_KG_rev.services.vector_store.stores.qdrant import QdrantVectorStore


@dataclass(slots=True)
class _CollectionParams:
    vectors: Any


@dataclass(slots=True)
class _CollectionConfig:
    params: _CollectionParams


@dataclass(slots=True)
class _CollectionInfo:
    config: _CollectionConfig
    points_count: int = 0


@dataclass(slots=True)
class _UpdateResult:
    deleted: int


@dataclass(slots=True)
class _ScoredPoint:
    id: str
    score: float
    payload: Mapping[str, object]


@dataclass(slots=True)
class _Snapshot:
    name: str
    size: int
    location: str


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.upserts: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.snapshots: dict[str, _Snapshot] = {}
        self.recoveries: list[dict[str, Any]] = []
        self.collection_sizes: dict[str, int] = {}

    def get_collection(self, *, collection_name: str, **_: Any) -> _CollectionInfo:
        if collection_name not in self.collections:
            raise UnexpectedResponse(
                status_code=404, reason_phrase="Not Found", content=b"", headers={}
            )
        record = self.collections[collection_name]
        params = _CollectionParams(vectors=record["vectors"])
        size = self.collection_sizes.get(collection_name, 0)
        return _CollectionInfo(config=_CollectionConfig(params=params), points_count=size)

    def recreate_collection(
        self, *, collection_name: str, vectors_config: Any, **kwargs: Any
    ) -> bool:
        self.collections[collection_name] = {
            "vectors": vectors_config,
            "kwargs": kwargs,
        }
        return True

    def update_collection(self, *, collection_name: str, **kwargs: Any) -> bool:
        self.collections.setdefault(collection_name, {}).setdefault("updates", []).append(kwargs)
        return True

    def upsert(self, *, collection_name: str, points: Sequence[qm.PointStruct], **_: Any) -> None:
        self.upserts.append({"collection": collection_name, "points": list(points)})
        self.collection_sizes[collection_name] = len(points)

    def search(
        self,
        *,
        collection_name: str,
        query_vector: Any,
        limit: int,
        query_filter: qm.Filter | None,
        search_params: qm.SearchParams | None,
        **_: Any,
    ) -> list[_ScoredPoint]:
        self.search_calls.append(
            {
                "collection": collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "filter": query_filter,
                "params": search_params,
            }
        )
        return [
            _ScoredPoint(id="vec-1", score=0.42, payload={"label": "alpha"}),
            _ScoredPoint(id="vec-2", score=0.33, payload={}),
        ]

    def delete(
        self, *, collection_name: str, points_selector: qm.PointIdsList, **_: Any
    ) -> _UpdateResult:
        self.delete_calls.append({"collection": collection_name, "selector": points_selector})
        deleted = len(points_selector.points)
        self.collection_sizes[collection_name] = max(
            0, self.collection_sizes.get(collection_name, 0) - deleted
        )
        return _UpdateResult(deleted=deleted)

    def create_snapshot(self, *, collection_name: str, **_: Any) -> _Snapshot:
        snapshot = _Snapshot(
            name=f"{collection_name}-snapshot",
            size=1024,
            location=f"/snapshots/{collection_name}.snap",
        )
        self.snapshots[collection_name] = snapshot
        return snapshot

    def recover_snapshot(
        self, *, collection_name: str, location: str | None, snapshot_name: str | None, **_: Any
    ) -> None:
        self.recoveries.append(
            {
                "collection": collection_name,
                "location": location,
                "snapshot_name": snapshot_name,
            }
        )


@pytest.fixture()
def client() -> FakeQdrantClient:
    return FakeQdrantClient()


def test_create_collection_hnsw_with_quantization(client: FakeQdrantClient) -> None:
    store = QdrantVectorStore(client=client)
    params = IndexParams(dimension=128, metric="cosine", kind="hnsw", m=16, ef_construct=64)
    compression = CompressionPolicy(kind="scalar_int8")

    store.create_or_update_collection(
        tenant_id="tenant-a",
        namespace="dense.documents",
        params=params,
        compression=compression,
        metadata={"gpu": {"enabled": True, "indexing_threshold": 10_000}},
    )

    vectors = client.collections["dense.documents"]["vectors"]
    assert isinstance(vectors, qm.VectorParams)
    assert vectors.size == 128
    assert isinstance(vectors.hnsw_config, qm.HnswConfigDiff)

    update_kwargs = client.collections["dense.documents"].get("updates", [])
    assert not update_kwargs


def test_upsert_and_query_with_filters(client: FakeQdrantClient) -> None:
    store = QdrantVectorStore(client=client)
    params = IndexParams(dimension=32, metric="dot", kind="hnsw")
    compression = CompressionPolicy(kind="binary")
    store.create_or_update_collection(
        tenant_id="tenant-a",
        namespace="dense.entities",
        params=params,
        compression=compression,
        metadata={"search": {"reorder_final": True}},
    )

    store.upsert(
        tenant_id="tenant-a",
        namespace="dense.entities",
        records=[
            VectorRecord(
                vector_id="vec-1",
                values=[0.1] * 32,
                metadata={"category": "trial"},
            )
        ],
    )

    assert client.upserts[0]["points"][0].payload["category"] == "trial"

    matches = store.query(
        tenant_id="tenant-a",
        namespace="dense.entities",
        query=VectorQuery(
            values=[0.2] * 32,
            top_k=2,
            filters={"category": "trial"},
        ),
    )

    assert [match.vector_id for match in matches] == ["vec-1", "vec-2"]
    call = client.search_calls[0]
    assert isinstance(call["filter"], qm.Filter)
    assert isinstance(call["params"].quantization, qm.QuantizationSearchParams)


def test_named_vectors_support(client: FakeQdrantClient) -> None:
    store = QdrantVectorStore(client=client)
    params = IndexParams(dimension=64, metric="cosine", kind="hnsw")
    compression = CompressionPolicy(kind="none")
    named = {"title": IndexParams(dimension=32, metric="cosine", kind="hnsw")}

    store.create_or_update_collection(
        tenant_id="tenant-a",
        namespace="multivector.documents",
        params=params,
        compression=compression,
        named_vectors=named,
    )

    store.upsert(
        tenant_id="tenant-a",
        namespace="multivector.documents",
        records=[
            VectorRecord(
                vector_id="vec-42",
                values=[0.0] * 64,
                named_vectors={"title": [0.5] * 32},
                metadata={"text": "alpha"},
            )
        ],
    )

    point = client.upserts[0]["points"][0]
    assert isinstance(point.vector, dict)
    assert len(point.vector["title"]) == 32

    store.query(
        tenant_id="tenant-a",
        namespace="multivector.documents",
        query=VectorQuery(values=[0.1] * 32, vector_name="title"),
    )

    call = client.search_calls[0]
    assert call["query_vector"] == ("title", [0.1] * 32)


def test_delete_wraps_qdrant_response(client: FakeQdrantClient) -> None:
    store = QdrantVectorStore(client=client)
    params = IndexParams(dimension=16, metric="cosine", kind="hnsw")
    compression = CompressionPolicy(kind="none")
    store.create_or_update_collection(
        tenant_id="tenant-a",
        namespace="dense.delete",
        params=params,
        compression=compression,
    )

    removed = store.delete(
        tenant_id="tenant-a",
        namespace="dense.delete",
        vector_ids=["vec-1", "vec-2"],
    )

    assert removed == 2
    assert isinstance(client.delete_calls[0]["selector"], qm.PointIdsList)


def test_snapshot_restore_and_health(client: FakeQdrantClient, tmp_path: Path) -> None:
    store = QdrantVectorStore(client=client)
    params = IndexParams(dimension=16, metric="cosine", kind="hnsw")
    compression = CompressionPolicy(kind="none")
    store.create_or_update_collection(
        tenant_id="tenant-a",
        namespace="dense.snapshot",
        params=params,
        compression=compression,
        metadata={},
    )
    store.upsert(
        tenant_id="tenant-a",
        namespace="dense.snapshot",
        records=[VectorRecord(vector_id="vec-1", values=[0.2] * 16, metadata={})],
    )
    path = tmp_path / "snapshot.json"
    info = store.create_snapshot(
        tenant_id="tenant-a",
        namespace="dense.snapshot",
        destination=str(path),
    )
    assert info.metadata and info.metadata["name"].endswith("snapshot")
    report = store.restore_snapshot(
        tenant_id="tenant-a",
        namespace="dense.snapshot",
        source=str(path),
    )
    assert report.rebuilt
    health = store.check_health(tenant_id="tenant-a")
    assert "dense.snapshot" in health and health["dense.snapshot"].healthy


def test_rebuild_index_triggers_update(client: FakeQdrantClient) -> None:
    store = QdrantVectorStore(client=client)
    params = IndexParams(dimension=16, metric="cosine", kind="hnsw")
    compression = CompressionPolicy(kind="none")
    store.create_or_update_collection(
        tenant_id="tenant-a",
        namespace="dense.rebuild",
        params=params,
        compression=compression,
        metadata={},
    )
    report = store.rebuild_index(
        tenant_id="tenant-a",
        namespace="dense.rebuild",
        force=True,
    )
    assert report.rebuilt


def test_missing_collection_raises(client: FakeQdrantClient) -> None:
    store = QdrantVectorStore(client=client)
    with pytest.raises(NamespaceNotFoundError):
        store.upsert(
            tenant_id="tenant-a",
            namespace="missing.namespace",
            records=[VectorRecord(vector_id="vec-1", values=[0.1], metadata={})],
        )
