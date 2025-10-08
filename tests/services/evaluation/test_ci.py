import pytest

from Medical_KG_rev.services.evaluation.ci import enforce_recall_threshold


def test_enforce_recall_threshold_allows_small_drop() -> None:
    enforce_recall_threshold(0.8, 0.77, tolerance=0.05)


def test_enforce_recall_threshold_raises_on_large_drop() -> None:
    with pytest.raises(RuntimeError):
        enforce_recall_threshold(0.8, 0.7, tolerance=0.05)
