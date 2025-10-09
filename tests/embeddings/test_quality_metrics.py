import types

import pytest

from Medical_KG_rev.embeddings.utils.normalization import normalize_batch
from Medical_KG_rev.embeddings.utils.tokenization import TokenLimitExceededError, TokenizerCache
from Medical_KG_rev.services.embedding.service import BatchController


def test_normalize_batch_unit_norm() -> None:
    vectors = normalize_batch([[3.0, 4.0], [0.0, 0.0]])
    assert sum(value * value for value in vectors[0]) == pytest.approx(1.0)
    assert vectors[1] == [0.0, 0.0]


def test_tokenizer_cache_reuses_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TokenizerCache()
    counts: list[str] = []

    class StubWrapper:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def count(self, text: str) -> int:
            counts.append(text)
            return len(text.split())

    monkeypatch.setattr(
        "Medical_KG_rev.embeddings.utils.tokenization.logger",
        types.SimpleNamespace(debug=lambda *args, **kwargs: None, error=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        "Medical_KG_rev.embeddings.utils.tokenization._TokenizerWrapper", StubWrapper
    )
    cache.ensure_within_limit(model_id="demo", texts=["a b", "c"], max_tokens=10)
    cache.ensure_within_limit(model_id="demo", texts=["d"], max_tokens=10)
    assert counts == ["a b", "c", "d"]


def test_tokenizer_cache_raises_on_exceed(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TokenizerCache()

    class StubWrapper:
        def __init__(self, model_id: str) -> None:
            pass

        def count(self, text: str) -> int:
            return 100

    monkeypatch.setattr(
        "Medical_KG_rev.embeddings.utils.tokenization.logger",
        types.SimpleNamespace(error=lambda *args, **kwargs: None, debug=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        "Medical_KG_rev.embeddings.utils.tokenization._TokenizerWrapper", StubWrapper
    )
    with pytest.raises(TokenLimitExceededError):
        cache.ensure_within_limit(model_id="demo", texts=["long text"], max_tokens=10)


def test_batch_controller_uses_candidates() -> None:
    controller = BatchController()
    size = controller.choose("ns", default=32, pending=5, candidates=[16, 8])
    assert size == 5


def test_batch_controller_prefers_history() -> None:
    controller = BatchController()
    controller.history["ns"] = [(8, 0.5), (16, 0.3)]
    size = controller.choose("ns", default=8, pending=20, candidates=[4, 8, 16])
    assert size == 16
    controller.record_success("ns", 16, 0.3)
    controller.reduce("ns", 4)
    assert controller.choose("ns", default=8, pending=10, candidates=[4, 8]) == 4
