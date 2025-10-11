"""Vector store configuration schema and loaders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
import yaml

from Medical_KG_rev.services.vector_store.models import CompressionPolicy, IndexParams
from Medical_KG_rev.services.vector_store.types import VectorStorePort



class CompressionConfig(BaseModel):
    kind: str = Field(default="none")
    pq_m: int | None = None
    pq_nbits: int | None = None
    opq_m: int | None = None

    def to_policy(self) -> CompressionPolicy:
        return CompressionPolicy(
            kind=self.kind,
            pq_m=self.pq_m,
            pq_nbits=self.pq_nbits,
            opq_m=self.opq_m,
        )

    @model_validator(mode="after")
    def _validate_params(self) -> CompressionConfig:
        if self.kind == "pq" and (self.pq_m is None or self.pq_nbits is None):
            raise ValueError("pq compression requires pq_m and pq_nbits")
        if self.kind == "opq" and (
            self.pq_m is None or self.pq_nbits is None or self.opq_m is None
        ):
            raise ValueError("opq compression requires pq_m, pq_nbits, and opq_m")
        return self


class NamedVectorConfig(BaseModel):
    name: str
    dimension: int
    metric: str = "cosine"
    kind: str = "hnsw"

    @field_validator("dimension")
    @classmethod
    def _validate_dimension(cls, value: int) -> int:
        if not 16 <= value <= 4096:
            raise ValueError("named vector dimension must be between 16 and 4096")
        return value

    def to_params(self) -> IndexParams:
        return IndexParams(dimension=self.dimension, metric=self.metric, kind=self.kind)


class NamespaceConfigModel(BaseModel):
    name: str
    driver: str = Field(default="milvus")
    params: dict[str, Any]
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    named_vectors: list[NamedVectorConfig] | None = None

    @field_validator("params")
    @classmethod
    def _validate_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "dimension" not in value:
            raise ValueError("index parameters must specify 'dimension'")
        dim = value["dimension"]
        if not 32 <= dim <= 4096:
            raise ValueError("vector dimension must be between 32 and 4096")
        return value

    def to_index_params(self) -> IndexParams:
        return IndexParams(**self.params)

    def named_vector_map(self) -> dict[str, IndexParams] | None:
        if not self.named_vectors:
            return None
        return {nv.name: nv.to_params() for nv in self.named_vectors}


class TenantVectorConfig(BaseModel):
    tenant_id: str
    namespaces: list[NamespaceConfigModel]


class VectorStoreConfig(BaseModel):
    backends: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tenants: list[TenantVectorConfig] = Field(default_factory=list)

    def namespace_for(self, tenant_id: str, namespace: str) -> NamespaceConfigModel | None:
        for tenant in self.tenants:
            if tenant.tenant_id != tenant_id:
                continue
            for item in tenant.namespaces:
                if item.name == namespace:
                    return item
        return None


def load_vector_store_config(path: Path | None = None) -> VectorStoreConfig:
    target = path or Path("config/vector_store.yaml")
    if not target.exists():
        return VectorStoreConfig()
    with target.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data = migrate_vector_store_config(data)
    try:
        return VectorStoreConfig.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - forwarded to caller
        raise ValueError(str(exc)) from exc


def detect_backend_capabilities(store: VectorStorePort) -> dict[str, Any]:
    """Return a normalised capability mapping for the supplied adapter."""
    capabilities = getattr(store, "capabilities", None)
    if callable(capabilities):
        result = capabilities()
        if isinstance(result, Mapping):
            return dict(result)
    return {
        "supports_hybrid": hasattr(store, "query"),
        "supports_named_vectors": hasattr(store, "create_or_update_collection"),
    }


def migrate_vector_store_config(data: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate legacy configuration structures into the current schema."""
    migrated: dict[str, Any] = dict(data)
    if "vector_store" in migrated:
        migrated = dict(migrated["vector_store"])
    if "namespaces" in migrated and "tenants" not in migrated:
        namespaces = migrated.pop("namespaces")
        migrated.setdefault("tenants", []).append(
            {"tenant_id": "default", "namespaces": namespaces}
        )
    for tenant in migrated.get("tenants", []):
        for namespace in tenant.get("namespaces", []):
            namespace.setdefault("driver", migrated.get("default_driver", "memory"))
    migrated.setdefault("backends", {})
    return migrated


__all__ = [
    "CompressionConfig",
    "NamedVectorConfig",
    "NamespaceConfigModel",
    "TenantVectorConfig",
    "VectorStoreConfig",
    "detect_backend_capabilities",
    "load_vector_store_config",
    "migrate_vector_store_config",
]
