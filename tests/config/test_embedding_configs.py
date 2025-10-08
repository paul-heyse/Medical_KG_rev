from pathlib import Path

from Medical_KG_rev.config import (
    DEFAULT_PYSERINI_CONFIG,
    DEFAULT_VLLM_CONFIG,
    load_pyserini_config,
    load_vllm_config,
)


def test_load_vllm_config_defaults() -> None:
    config = load_vllm_config(DEFAULT_VLLM_CONFIG)
    assert config.service.host == "0.0.0.0"
    assert 0.0 < config.service.gpu_memory_utilization <= 1.0
    assert config.model.name
    assert config.batching.max_batch_size >= 1


def test_load_pyserini_config_defaults(tmp_path: Path) -> None:
    config = load_pyserini_config(DEFAULT_PYSERINI_CONFIG)
    assert config.service.port == 8002
    assert config.model.cache_dir.endswith("splade")
    assert config.expansion.doc_side.top_k_terms >= 100
    assert config.opensearch.rank_features_field == "splade_terms"
