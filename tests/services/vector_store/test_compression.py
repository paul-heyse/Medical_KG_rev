"""Unit tests for compression utilities."""

from __future__ import annotations

import numpy as np

from Medical_KG_rev.services.vector_store.compression import (
    CompressionError,
    CompressionManager,
    binary_quantize,
    learn_opq_rotation,
    quantize_fp16,
    quantize_int8,
    train_pq,
    two_stage_reorder,
)
from Medical_KG_rev.services.vector_store.models import CompressionPolicy


def _sample_vectors() -> list[list[float]]:
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.0, -0.1, -0.2, -0.3],
        [0.5, 0.4, 0.3, 0.2],
        [0.9, 0.8, 0.7, 0.6],
    ]


def test_quantize_int8_round_trip() -> None:
    vectors = _sample_vectors()
    payload = quantize_int8(vectors)
    assert payload["values"].dtype == np.int8
    assert payload["scales"].shape[0] == len(vectors)


def test_quantize_fp16() -> None:
    payload = quantize_fp16(_sample_vectors())
    assert payload["values"].dtype == np.float16


def test_binary_quantization() -> None:
    payload = binary_quantize(_sample_vectors())
    assert payload["values"].dtype == np.uint8


def test_product_quantization_training() -> None:
    vectors = _sample_vectors() * 2
    pq = train_pq(vectors, m=2, nbits=4)
    assert pq["m"] == 2
    assert len(pq["codebooks"]) == 2


def test_opq_rotation() -> None:
    rotation = learn_opq_rotation(_sample_vectors(), m=2)
    assert rotation["rotation"].shape[0] == 4


def test_two_stage_reorder_prefers_higher_scores() -> None:
    query = [1.0, 0.0, 0.0, 0.0]
    coarse = [
        ("q", 0.0, {"query": query}),
        ("a", 0.2, {}),
        ("b", 0.1, {}),
    ]
    reordered = two_stage_reorder(
        coarse,
        reorder_embeddings={
            "a": [0.8, 0.1, 0.0, 0.0],
            "b": [0.2, 0.1, 0.0, 0.0],
        },
        top_k=2,
    )
    assert reordered[0][0] == "a"


def test_compression_manager_validates_parameters() -> None:
    manager = CompressionManager()
    policy = CompressionPolicy(kind="pq", pq_m=2, pq_nbits=4)
    payload = manager.compress(_sample_vectors(), policy)
    assert "pq" in payload


def test_compression_manager_rejects_invalid_pq() -> None:
    manager = CompressionManager()
    policy = CompressionPolicy(kind="pq", pq_m=3, pq_nbits=None)
    try:
        manager.compress(_sample_vectors(), policy)
    except CompressionError as exc:
        assert "pq" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected CompressionError")
