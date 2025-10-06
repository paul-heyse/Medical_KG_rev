from __future__ import annotations

import numpy as np

from Medical_KG_rev.services.retrieval.faiss_index import FAISSIndex


def test_add_and_search_vectors():
    index = FAISSIndex(dimension=4)
    index.add("a", [1.0, 0.0, 0.0, 0.0], {"chunk_id": "chunk-1"})
    index.add("b", [0.0, 1.0, 0.0, 0.0], {"chunk_id": "chunk-2"})

    results = index.search([1.0, 0.0, 0.0, 0.0], k=1)

    assert results == [("a", 1.0, {"chunk_id": "chunk-1"})]


def test_persistence_round_trip(tmp_path):
    index = FAISSIndex(dimension=2)
    index.add("a", [0.1, 0.2])
    path = tmp_path / "index.json"
    index.save(path)

    loaded = FAISSIndex.load(path)
    assert loaded.ids == ["a"]
    np.testing.assert_allclose(loaded.vectors[0], np.asarray([0.1, 0.2]))
