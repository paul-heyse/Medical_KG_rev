from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
import pytest

from Medical_KG_rev.services.vector_store.errors import (
    DimensionMismatchError,
    InvalidNamespaceConfigError,
)
from Medical_KG_rev.services.vector_store.models import (
    CompressionPolicy,
    IndexParams,
    VectorMatch,
    VectorQuery,
    VectorRecord,
)
from Medical_KG_rev.services.vector_store.stores.faiss import FaissVectorStore

TENANT = "tenant-a"
NAMESPACE = "dense.test"


def _make_vectors(count: int, dimension: int) -> list[VectorRecord]:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((count, dimension)).astype(np.float32)
    return [
        VectorRecord(
            vector_id=f"vec-{idx}",
            values=vector.tolist(),
            metadata={"order": idx},
        )
        for idx, vector in enumerate(base)
    ]


def _top_ids(matches: Sequence[VectorMatch]) -> list[str]:
    return [match.vector_id for match in matches]


def test_flat_index_roundtrip(tmp_path: Path) -> None:
    store = FaissVectorStore(base_path=tmp_path)
    params = IndexParams(dimension=8, metric="cosine", kind="flat")
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=CompressionPolicy(),
    )
    records = _make_vectors(6, params.dimension)
    store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=records)

    query_vector = records[0].values
    matches = store.query(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        query=VectorQuery(values=query_vector, top_k=3),
    )

    assert _top_ids(matches)[0] == records[0].vector_id
    assert matches[0].metadata == {"order": 0}


def test_ivf_flat_trains_and_queries(tmp_path: Path) -> None:
    store = FaissVectorStore(base_path=tmp_path)
    params = IndexParams(
        dimension=16,
        metric="l2",
        kind="ivf_flat",
        nlist=4,
        nprobe=2,
        train_size=12,
    )
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=CompressionPolicy(),
    )
    records = _make_vectors(24, params.dimension)
    store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=records)

    matches = store.query(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        query=VectorQuery(values=records[1].values, top_k=5),
    )

    assert records[1].vector_id in _top_ids(matches)


def test_ivf_pq_reorders_results(tmp_path: Path) -> None:
    store = FaissVectorStore(base_path=tmp_path)
    params = IndexParams(
        dimension=12,
        metric="cosine",
        kind="ivf_pq",
        nlist=2,
        nprobe=2,
        reorder_k=8,
        train_size=16,
    )
    compression = CompressionPolicy(kind="pq", pq_m=3, pq_nbits=4)
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=compression,
    )
    records = _make_vectors(128, params.dimension)
    store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=records)

    query_vector = records[2].values
    matches = store.query(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        query=VectorQuery(values=query_vector, top_k=5, reorder=True),
    )

    assert _top_ids(matches)[0] == records[2].vector_id
    assert len(matches) == 5


def test_opq_ivf_pq_supports_reorder_flag(tmp_path: Path) -> None:
    store = FaissVectorStore(base_path=tmp_path)
    params = IndexParams(
        dimension=12,
        metric="cosine",
        kind="opq_ivf_pq",
        nlist=2,
        nprobe=1,
        train_size=256,
    )
    compression = CompressionPolicy(kind="opq_pq", pq_m=3, pq_nbits=8, opq_m=3)
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=compression,
    )
    records = _make_vectors(512, params.dimension)
    store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=records)

    query_vector = records[4].values
    matches = store.query(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        query=VectorQuery(values=query_vector, top_k=3),
    )

    assert _top_ids(matches)[0] == records[4].vector_id


def test_scalar_quantization_requires_training(tmp_path: Path) -> None:
    store = FaissVectorStore(base_path=tmp_path)
    params = IndexParams(dimension=10, metric="l2", kind="flat", train_size=10)
    compression = CompressionPolicy(kind="scalar_int8")
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=compression,
    )
    records = _make_vectors(12, params.dimension)
    store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=records)

    matches = store.query(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        query=VectorQuery(values=records[0].values, top_k=2),
    )

    assert _top_ids(matches)[0] == records[0].vector_id


def test_fp16_quantization(tmp_path: Path) -> None:
    store = FaissVectorStore(base_path=tmp_path)
    params = IndexParams(dimension=9, metric="cosine", kind="flat", train_size=10)
    compression = CompressionPolicy(kind="fp16")
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=compression,
    )
    records = _make_vectors(10, params.dimension)
    store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=records)

    matches = store.query(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        query=VectorQuery(values=records[3].values, top_k=3),
    )

    assert records[3].vector_id in _top_ids(matches)


def test_gpu_requested_without_device_falls_back(tmp_path: Path) -> None:
    store = FaissVectorStore(base_path=tmp_path)
    params = IndexParams(dimension=6, metric="cosine", kind="flat", use_gpu=True)
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=CompressionPolicy(),
    )
    records = _make_vectors(6, params.dimension)
    store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=records)

    matches = store.query(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        query=VectorQuery(values=records[0].values, top_k=2),
    )

    assert len(matches) == 2
    state = store._tenants[TENANT][NAMESPACE]
    if faiss.get_num_gpus() == 0:
        assert state.gpu_index is None


def test_index_persistence_round_trip(tmp_path: Path) -> None:
    params = IndexParams(dimension=7, metric="cosine", kind="flat")
    compression = CompressionPolicy()
    records = _make_vectors(5, params.dimension)

    store = FaissVectorStore(base_path=tmp_path)
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=compression,
    )
    store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=records)

    store = FaissVectorStore(base_path=tmp_path)
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=compression,
    )
    matches = store.query(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        query=VectorQuery(values=records[1].values, top_k=2),
    )

    assert records[1].vector_id in _top_ids(matches)


def test_delete_removes_vector(tmp_path: Path) -> None:
    store = FaissVectorStore(base_path=tmp_path)
    params = IndexParams(dimension=5, metric="cosine", kind="flat")
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=CompressionPolicy(),
    )
    records = _make_vectors(4, params.dimension)
    store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=records)

    removed = store.delete(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        vector_ids=[records[0].vector_id],
    )
    assert removed == 1

    matches = store.query(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        query=VectorQuery(values=records[0].values, top_k=3),
    )

    assert records[0].vector_id not in _top_ids(matches)


def test_named_vectors_not_supported(tmp_path: Path) -> None:
    store = FaissVectorStore(base_path=tmp_path)
    params = IndexParams(dimension=4, metric="cosine", kind="flat")
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=CompressionPolicy(),
    )
    record = VectorRecord(
        vector_id="vec-1",
        values=[0.1, 0.2, 0.3, 0.4],
        named_vectors={"extra": [0.1, 0.2, 0.3, 0.4]},
    )

    with pytest.raises(InvalidNamespaceConfigError):
        store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=[record])


def test_dimension_mismatch_raises(tmp_path: Path) -> None:
    store = FaissVectorStore(base_path=tmp_path)
    params = IndexParams(dimension=4, metric="cosine", kind="flat")
    store.create_or_update_collection(
        tenant_id=TENANT,
        namespace=NAMESPACE,
        params=params,
        compression=CompressionPolicy(),
    )

    record = VectorRecord(vector_id="bad", values=[0.1, 0.2])
    with pytest.raises(DimensionMismatchError):
        store.upsert(tenant_id=TENANT, namespace=NAMESPACE, records=[record])

    with pytest.raises(DimensionMismatchError):
        store.query(
            tenant_id=TENANT,
            namespace=NAMESPACE,
            query=VectorQuery(values=[0.1, 0.2], top_k=1),
        )
