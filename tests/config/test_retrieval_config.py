from pathlib import Path

import pytest

from Medical_KG_rev.config.retrieval_config import (
    BM25Config,
    Qwen3Config,
    RetrievalConfig,
    SPLADEConfig,
)


def test_retrieval_config_defaults():
    config = RetrievalConfig.from_dict({})
    assert config.default_backend == "hybrid"
    assert config.bm25.field_boosts["title"] > config.bm25.field_boosts["paragraph"]
    assert config.splade.max_tokens == 512
    assert config.qwen3.embedding_dimension == 4096


def test_retrieval_config_from_mapping(tmp_path: Path):
    synonyms = tmp_path / "mesh.txt"
    synonyms.write_text("heart attack => myocardial infarction", encoding="utf-8")
    payload = {
        "default_backend": "bm25",
        "bm25": {
            "index_path": tmp_path / "bm25",
            "synonyms_path": synonyms,
            "field_boosts": {"title": 4.0, "paragraph": 1.0},
        },
        "splade": {
            "model_name": "naver/splade-v3",
            "tokenizer_name": "naver/splade-v3",
            "max_tokens": 256,
        },
        "qwen3": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
            "embedding_dimension": 2048,
        },
    }
    config = RetrievalConfig.from_dict(payload)
    assert config.default_backend == "bm25"
    assert config.bm25.index_path == tmp_path / "bm25"
    assert config.bm25.synonyms_path == synonyms
    assert config.splade.max_tokens == 256
    assert config.qwen3.embedding_dimension == 2048


@pytest.mark.parametrize(
    "model_class, kwargs, message",
    [
        (BM25Config, {"field_boosts": {}}, "field_boosts"),
        (SPLADEConfig, {"tokenizer_name": "other"}, "tokenizer_name"),
        (Qwen3Config, {"embedding_dimension": 0}, "embedding_dimension"),
    ],
)
def test_retrieval_config_validation_errors(model_class, kwargs, message):
    with pytest.raises(ValueError) as exc:
        model_class(**kwargs)
    assert message in str(exc.value)
