"""Tests for vector store configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from Medical_KG_rev.config.vector_store import (
    detect_backend_capabilities,
    load_vector_store_config,
    migrate_vector_store_config,
)
from Medical_KG_rev.services.vector_store.stores.memory import InMemoryVectorStore


def test_load_vector_store_config(tmp_path: Path) -> None:
    config_path = tmp_path / "vector_store.yaml"
    config_path.write_text(
        """
backends:
  milvus:
    uri: local
tenants:
  - tenant_id: t1
    namespaces:
      - name: dense
        driver: milvus
        params:
          dimension: 128
          metric: cosine
          kind: hnsw
        compression:
          kind: pq
          pq_m: 16
          pq_nbits: 8
""",
        encoding="utf-8",
    )
    config = load_vector_store_config(config_path)
    tenant = config.tenants[0]
    namespace = tenant.namespaces[0]
    assert namespace.compression.kind == "pq"
    assert namespace.to_index_params().dimension == 128


def test_invalid_dimension(tmp_path: Path) -> None:
    config_path = tmp_path / "vector_store.yaml"
    config_path.write_text(
        """
tenants:
  - tenant_id: t1
    namespaces:
      - name: dense
        driver: milvus
        params:
          dimension: 8
          metric: cosine
          kind: hnsw
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_vector_store_config(config_path)



def test_migrate_legacy_structure() -> None:
    raw = {
        "vector_store": {
            "default_driver": "memory",
            "namespaces": [
                {"name": "default", "params": {"dimension": 64}},
            ],
        }
    }
    migrated = migrate_vector_store_config(raw)
    assert migrated["tenants"][0]["namespaces"][0]["driver"] == "memory"


def test_detect_backend_capabilities_from_delegate() -> None:
    store = InMemoryVectorStore()
    capabilities = detect_backend_capabilities(store)
    assert capabilities["supports_hybrid"] is True
