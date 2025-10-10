from __future__ import annotations

import json
from pathlib import Path

import faiss  # type: ignore import-not-found
import pytest

from Medical_KG_rev.services import GpuNotAvailableError
from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex


def test_gpu_enforcement_without_device():
    if faiss.get_num_gpus() > 0:  # pragma: no cover - exercised only on GPU hosts
        pytest.skip("GPU present in test environment")
    with pytest.raises(GpuNotAvailableError):
        FAISSIndex(dimension=4, use_gpu=True)


def test_add_and_search_vectors_cpu():
    index = FAISSIndex(dimension=4, use_gpu=False)
    index.add("a", [1.0, 0.0, 0.0, 0.0], {"chunk_id": "chunk-1"})
    index.add("b", [0.0, 1.0, 0.0, 0.0], {"chunk_id": "chunk-2"})

    results = index.search([1.0, 0.0, 0.0, 0.0], k=1)

    assert results[0][0] == "a"
    assert results[0][2] == {"chunk_id": "chunk-1"}
    assert results[0][1] == pytest.approx(1.0)


def test_persistence_round_trip(tmp_path):
    index = FAISSIndex(dimension=2, use_gpu=False)
    index.add("a", [0.1, 0.2], {"chunk_id": "chunk-1"})
    path = tmp_path / "index.faiss"
    index.save(path)

    meta_path = Path(str(path) + ".meta.json")
    assert path.exists()
    assert meta_path.exists()
    metadata = json.loads(meta_path.read_text("utf-8"))
    assert metadata["ids"] == ["a"]

    loaded = FAISSIndex.load(path, use_gpu=False)
    assert loaded.ids == ["a"]
    results = loaded.search([0.1, 0.2], k=1)
    assert results[0][0] == "a"
    assert results[0][2]["chunk_id"] == "chunk-1"
