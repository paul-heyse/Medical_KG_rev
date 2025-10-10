import pytest

from Medical_KG_rev.embeddings.utils.batching import BatchProgress, iter_with_progress


def test_iter_with_progress_updates_callback() -> None:
    processed: list[tuple[int, int]] = []
    progress = BatchProgress(
        total=5, callback=lambda processed_count, total: processed.append((processed_count, total))
    )
    batches = list(iter_with_progress([1, 2, 3, 4, 5], 2, progress=progress))
    assert len(batches) == 3
    assert processed[-1] == (5, 5)


def test_iter_with_progress_validates_batch_size() -> None:
    with pytest.raises(ValueError):
        list(iter_with_progress([], 0))
