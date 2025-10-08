import types

import pytest

np = pytest.importorskip("numpy")

from Medical_KG_rev.services.vector_store.errors import InvalidNamespaceConfigError
from Medical_KG_rev.services.vector_store.models import CompressionPolicy, IndexParams
from Medical_KG_rev.services.vector_store.stores import faiss as faiss_store


@pytest.fixture(autouse=True)
def stub_faiss(monkeypatch: pytest.MonkeyPatch):
    class DummyIndex:
        pass

    class DummyPreTransform(DummyIndex):
        def __init__(self) -> None:
            self.index = DummyIndex()

    class DummyIDMap(DummyIndex):
        def __init__(self) -> None:
            self.index = DummyIndex()

    class DummyHnsw(DummyIndex):
        def __init__(self) -> None:
            self.hnsw = types.SimpleNamespace(efSearch=0)

    fake = types.SimpleNamespace(
        METRIC_INNER_PRODUCT=1,
        METRIC_L2=2,
        Index=DummyIndex,
        IndexIDMap2=DummyIDMap,
        IndexPreTransform=DummyPreTransform,
        IndexHNSW=DummyHnsw,
        downcast_index=lambda index: index,
        get_num_gpus=lambda: 0,
    )
    monkeypatch.setattr(faiss_store, "faiss", fake)
    return fake


def test_flat_factory_string_variants() -> None:
    assert faiss_store._flat_factory_string(CompressionPolicy(kind="none")) == "Flat"
    assert faiss_store._flat_factory_string(CompressionPolicy(kind="scalar_int8")) == "SQ8"
    assert faiss_store._flat_factory_string(CompressionPolicy(kind="fp16")) == "SQfp16"


def test_should_normalize_cosine() -> None:
    assert faiss_store._should_normalize("cosine") is True
    assert faiss_store._should_normalize("l2") is False


def test_faiss_metric_mapping() -> None:
    assert faiss_store._faiss_metric("cosine") == 1
    assert faiss_store._faiss_metric("l2") == 2
    with pytest.raises(ValueError):
        faiss_store._faiss_metric("unknown")


def test_should_enable_reorder_based_on_params() -> None:
    params = IndexParams(dimension=128, kind="ivf_pq")
    compression = CompressionPolicy(kind="none")
    assert faiss_store.FaissVectorStore()._should_enable_reorder(params, compression) is True
    params = IndexParams(dimension=128, kind="flat")
    compression = CompressionPolicy(kind="pq")
    assert faiss_store.FaissVectorStore()._should_enable_reorder(params, compression) is True
    params = IndexParams(dimension=128, kind="flat")
    compression = CompressionPolicy(kind="none")
    assert faiss_store.FaissVectorStore()._should_enable_reorder(params, compression) is False


def test_training_threshold_uses_compression() -> None:
    params = IndexParams(dimension=128, kind="flat")
    compression = CompressionPolicy(kind="pq")
    assert faiss_store._training_threshold(params, compression) >= 128 * 4
    params = IndexParams(dimension=64, kind="flat", train_size=50)
    compression = CompressionPolicy(kind="none")
    assert faiss_store._training_threshold(params, compression) == 50


def test_distance_to_score_transforms_sign() -> None:
    assert faiss_store._distance_to_score(0.25, "l2") == -0.25
    assert faiss_store._distance_to_score(0.75, "cosine") == pytest.approx(0.75)


def test_normalize_vector_handles_zero() -> None:
    zero = np.zeros(4, dtype=np.float32)
    assert np.allclose(faiss_store._normalize(zero), zero)
    vector = np.array([3.0, 4.0], dtype=np.float32)
    normalized = faiss_store._normalize(vector)
    assert np.allclose(normalized, np.array([0.6, 0.8], dtype=np.float32))


def test_json_default_handles_complex_types() -> None:
    class Custom:
        def __str__(self) -> str:
            return "custom"

    assert faiss_store._json_default(1) == 1
    assert faiss_store._json_default([1, 2]) == [1, 2]
    assert faiss_store._json_default(Custom()) == "custom"


def test_memory_state_reorder_defaults(stub_faiss) -> None:
    store = faiss_store.FaissVectorStore()
    params = IndexParams(dimension=16)
    compression = CompressionPolicy(kind="none")
    store.create_or_update_collection(
        tenant_id="tenant",
        namespace="demo",
        params=params,
        compression=compression,
        metadata={},
        named_vectors=None,
    )
    state = store._tenants["tenant"]["demo"]  # type: ignore[attr-defined]
    assert state.reorder_enabled is False


def test_create_collection_rejects_named_vectors(stub_faiss) -> None:
    store = faiss_store.FaissVectorStore()
    params = IndexParams(dimension=16)
    compression = CompressionPolicy(kind="none")
    with pytest.raises(InvalidNamespaceConfigError):
        store.create_or_update_collection(
            tenant_id="tenant",
            namespace="demo",
            params=params,
            compression=compression,
            metadata={},
            named_vectors={"extra": IndexParams(dimension=16)},
        )
