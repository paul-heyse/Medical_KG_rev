import math

from Medical_KG_rev.services.reranking.fusion.normalization import (
    apply_normalization,
    min_max,
    softmax,
    z_score,
)


def test_min_max_normalization():
    values = [0.0, 5.0, 10.0]
    normalised = min_max(values)
    assert normalised == [0.0, 0.5, 1.0]


def test_z_score_normalization_handles_constant():
    assert z_score([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]


def test_softmax_normalization():
    result = softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(result), 1.0)
    assert result[-1] == max(result)


def test_apply_normalization_dispatch():
    values = [1.0, 2.0, 3.0]
    assert apply_normalization("bm25", values, "min_max") == min_max(values)
