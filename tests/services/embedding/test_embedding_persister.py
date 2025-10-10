from __future__ import annotations

import pytest

from Medical_KG_rev.embeddings.ports import EmbeddingRecord
from Medical_KG_rev.embeddings.storage import StorageRouter
from Medical_KG_rev.services.embedding.persister import (
    DatabasePersister,
    DryRunPersister,
    HybridPersister,
    MockPersister,
    PersistenceContext,
    PersisterRuntimeSettings,
    VectorStorePersister,
    build_persister,
)
from Medical_KG_rev.services.embedding.telemetry import (
    StandardEmbeddingTelemetry,
    TelemetrySettings,
)


def _sample_record(record_id: str, kind: str = "single_vector") -> EmbeddingRecord:
    return EmbeddingRecord(
        id=record_id,
        tenant_id="tenant-1",
        namespace="single_vector.test.3.v1",
        model_id="test-model",
        model_version="v1",
        kind=kind,  # type: ignore[arg-type]
        dim=3,
        vectors=[[0.1, 0.2, 0.3]] if kind != "sparse" else None,
        terms={"a": 1.0} if kind == "sparse" else None,
        metadata={"source": "unit-test"},
    )


def _context() -> PersistenceContext:
    return PersistenceContext(
        tenant_id="tenant-1",
        namespace="single_vector.test.3.v1",
        model="test-model",
        provider="test-provider",
        job_id="job-123",
        correlation_id="corr-123",
        normalize=False,
    )


def test_vector_store_persister_persists_and_caches() -> None:
    router = StorageRouter()
    telemetry = StandardEmbeddingTelemetry(
        TelemetrySettings(enable_logging=False, enable_metrics=False)
    )
    persister = VectorStorePersister(router, telemetry=telemetry)

    record = _sample_record("rec-1")
    report = persister.persist_batch([record], _context())

    assert report.persisted == 1
    buffered = router.buffered("faiss", tenant_id="tenant-1")
    assert len(buffered) == 1
    snapshot = telemetry.snapshot()
    assert snapshot.metadata["persistence"]["single_vector.test.3.v1"] == 1


def test_database_persister_stores_records() -> None:
    persister = DatabasePersister()
    record = _sample_record("rec-2")

    report = persister.persist_batch([record], _context())
    assert report.persisted == 1
    assert persister.retrieve(ids=["rec-2"]) == [record]
    assert persister.delete(ids=["rec-2"]) == 1


def test_dry_run_persister_records_operations() -> None:
    persister = DryRunPersister()
    record = _sample_record("rec-3")

    report = persister.persist_batch([record], _context())
    assert report.skipped == 1
    assert len(persister.operations) == 1


def test_mock_persister_tracks_persisted_records() -> None:
    persister = MockPersister()
    record = _sample_record("rec-4")

    report = persister.persist_batch([record], _context())
    assert report.persisted == 1
    assert persister.persisted_records == [record]


def test_hybrid_persister_routes_by_kind() -> None:
    dense_persister = MockPersister()
    sparse_persister = MockPersister()
    hybrid = HybridPersister({"single_vector": dense_persister, "sparse": sparse_persister})

    dense = _sample_record("rec-5", kind="single_vector")
    sparse = EmbeddingRecord(
        id="rec-6",
        tenant_id="tenant-1",
        namespace="single_vector.test.3.v1",
        model_id="test-model",
        model_version="v1",
        kind="sparse",  # type: ignore[arg-type]
        dim=None,
        vectors=None,
        terms={"term": 0.5},
        metadata={"source": "unit-test"},
    )

    report = hybrid.persist_batch([dense, sparse], _context())

    assert report.persisted == 2
    assert dense_persister.persisted_records == [dense]
    assert sparse_persister.persisted_records == [sparse]


def test_configure_cache_limit_controls_eviction() -> None:
    router = StorageRouter()
    persister = VectorStorePersister(router)
    first = _sample_record("rec-cache-1")
    second = _sample_record("rec-cache-2")

    persister.persist_batch([first], _context())
    persister.configure(cache_limit=1)
    persister.persist_batch([second], _context())

    assert persister.retrieve(ids=[first.id]) == []
    assert persister.retrieve(ids=[second.id]) == [second]
    with pytest.raises(ValueError):
        persister.configure(cache_limit=-1)


def test_build_persister_from_settings_vector_store() -> None:
    router = StorageRouter()
    settings = PersisterRuntimeSettings(backend="vector_store", cache_limit=16)
    persister = build_persister(router, settings=settings)

    assert isinstance(persister, VectorStorePersister)


def test_build_persister_hybrid_instantiates_children() -> None:
    router = StorageRouter()
    settings = PersisterRuntimeSettings(
        backend="hybrid",
        hybrid_backends={"single_vector": "vector_store", "sparse": "database"},
    )
    persister = build_persister(router, settings=settings)

    assert isinstance(persister, HybridPersister)


def test_build_persister_invalid_backend_raises() -> None:
    router = StorageRouter()
    settings = PersisterRuntimeSettings(backend="unknown")

    with pytest.raises(ValueError):
        build_persister(router, settings=settings)
