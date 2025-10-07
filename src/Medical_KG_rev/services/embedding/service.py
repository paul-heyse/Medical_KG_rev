"""Universal embedding service coordinating multiple adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path
from typing import Iterator, Sequence

import structlog

from Medical_KG_rev.config.embeddings import load_embeddings_config
from Medical_KG_rev.embeddings.namespace import NamespaceManager
from Medical_KG_rev.embeddings.ports import (
    BaseEmbedder,
    EmbedderConfig,
    EmbeddingRecord,
    EmbeddingRequest as AdapterEmbeddingRequest,
)
from Medical_KG_rev.embeddings.providers import register_builtin_embedders
from Medical_KG_rev.embeddings.registry import EmbedderFactory, EmbedderRegistry
from Medical_KG_rev.embeddings.storage import StorageRouter
from Medical_KG_rev.embeddings.utils.batching import BatchProgress
from Medical_KG_rev.embeddings.utils.gpu import ensure_available
from Medical_KG_rev.services.vector_store.models import VectorRecord
from Medical_KG_rev.services.vector_store.service import VectorStoreService
from Medical_KG_rev.auth.context import SecurityContext

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class EmbeddingRequest:
    tenant_id: str
    chunk_ids: Sequence[str]
    texts: Sequence[str]
    normalize: bool = False
    batch_size: int = 8
    models: Sequence[str] | None = None
    namespaces: Sequence[str] | None = None
    correlation_id: str | None = None


@dataclass(slots=True)
class EmbeddingVector:
    id: str
    model: str
    namespace: str
    kind: str
    vectors: list[list[float]] | None
    terms: dict[str, float] | None
    dimension: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingResponse:
    vectors: list[EmbeddingVector] = field(default_factory=list)


@dataclass(slots=True)
class _LegacyEmbeddingModel:
    name: str
    dimension: int
    kind: str

    def embed(self, chunk_id: str, text: str) -> dict[str, object]:
        tokens = text.split()
        if self.kind == "dense":
            base = float(len(tokens) or 1)
            values = [round((base + index) % 7, 4) for index in range(self.dimension)]
            return {"id": chunk_id, "values": values}
        weights = {
            f"{tokens[index % max(1, len(tokens))]}_{index}": round(1.0 - (index * 0.1), 4)
            for index in range(min(self.dimension, max(1, len(tokens))))
        }
        return {"id": chunk_id, "terms": weights}


class EmbeddingModelRegistry:
    """Compatibility shim for legacy embedding registry usage in tests."""

    def __init__(self, gpu_manager: object | None = None) -> None:  # noqa: ARG002 - parity
        self.namespace_manager = NamespaceManager()
        self._models: dict[str, _LegacyEmbeddingModel] = {
            "splade": _LegacyEmbeddingModel(name="splade", dimension=64, kind="sparse"),
            "bge": _LegacyEmbeddingModel(name="bge", dimension=128, kind="dense"),
        }
        self._aliases = {
            "splade": "splade",
            "sparse": "splade",
            "bge": "bge",
            "bge-small": "bge",
            "dense": "bge",
        }

    def list_models(self) -> list[str]:
        return list(self._models)

    def get(self, name: str) -> _LegacyEmbeddingModel:
        key = self._aliases.get(name, name)
        try:
            return self._models[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown embedding model '{name}'") from exc


class EmbeddingWorker:
    """Coordinates config-driven embedding generation and validation."""

    def __init__(
        self,
        registry_or_namespace: EmbeddingModelRegistry | NamespaceManager | None = None,
        *,
        namespace_manager: NamespaceManager | None = None,
        config_path: str | None = None,
    ) -> None:
        self._legacy_registry: EmbeddingModelRegistry | None = None
        if isinstance(registry_or_namespace, EmbeddingModelRegistry):
            self._legacy_registry = registry_or_namespace
            self.namespace_manager = registry_or_namespace.namespace_manager
            self.registry = None
            self.factory = None
            self.storage_router = StorageRouter()
            self._config = None
            self._embedder_configs: list[EmbedderConfig] = []
            self._configs_by_name: dict[str, EmbedderConfig] = {}
            self._configs_by_namespace: dict[str, EmbedderConfig] = {}
            return

        if isinstance(registry_or_namespace, NamespaceManager) and namespace_manager is not None:
            raise TypeError("namespace_manager should not be provided twice")
        effective_namespace = namespace_manager
        if effective_namespace is None and isinstance(registry_or_namespace, NamespaceManager):
            effective_namespace = registry_or_namespace
        self.namespace_manager = effective_namespace or NamespaceManager()
        self.registry = EmbedderRegistry(namespace_manager=self.namespace_manager)
        register_builtin_embedders(self.registry)
        self.factory = EmbedderFactory(self.registry)
        loaded_config = load_embeddings_config(Path(config_path) if config_path else None)
        embedder_configs = loaded_config.to_embedder_configs()
        if _gpu_manager is not None:
            gpu_configs = [
                EmbedderConfig(
                    name="bge-gpu",
                    provider="sentence-transformers",
                    kind="single_vector",
                    namespace="single_vector.gpu_compat.64.v1",
                    model_id="BAAI/bge-small-en",
                    dim=64,
                    normalize=True,
                ),
                EmbedderConfig(
                    name="colbert-gpu",
                    provider="colbert",
                    kind="multi_vector",
                    namespace="multi_vector.gpu_compat.128.v1",
                    model_id="colbert/colbertv2",
                    dim=128,
                    normalize=False,
                ),
            ]
            embedder_configs = [*embedder_configs, *gpu_configs]
            active_namespaces = [config.namespace for config in gpu_configs]
            self._config = SimpleNamespace(active_namespaces=active_namespaces)
        else:
            self._config = loaded_config
        self._embedder_configs = embedder_configs
        self._configs_by_name = {config.name: config for config in self._embedder_configs}
        alias_map: dict[str, EmbedderConfig] = {}
        for config in self._embedder_configs:
            if "-" in config.name:
                base_name, _ = config.name.split("-", 1)
                if base_name and base_name not in self._configs_by_name:
                    alias_map[base_name] = config
                underscored = config.name.replace("-", "_")
                if underscored and underscored not in self._configs_by_name:
                    alias_map[underscored] = config
        self._configs_by_name.update(alias_map)

    def get(self, name: str) -> BaseEmbedder:
        config = self._configs_by_name[name]
        return self.factory.get(config)

    def configs(self) -> list[EmbedderConfig]:
        return list(self._embedder_configs)

    @property
    def active_namespaces(self) -> list[str]:
        return list(self._config.active_namespaces)


class EmbeddingWorker:
    """Coordinates config-driven embedding generation and validation."""

    def __init__(
        self,
        registry: EmbeddingModelRegistry | None = None,
        *,
        namespace_manager: NamespaceManager | None = None,
        config_path: str | None = None,
    ) -> None:
        if registry is not None:
            self.namespace_manager = registry.namespace_manager
            self.registry = registry.registry
            self.factory = registry.factory
            self._config = registry._config
            self._embedder_configs = registry.configs()
        else:
            self.namespace_manager = namespace_manager or NamespaceManager()
            self.registry = EmbedderRegistry(namespace_manager=self.namespace_manager)
            register_builtin_embedders(self.registry)
            self.factory = EmbedderFactory(self.registry)
            self._config = load_embeddings_config(Path(config_path) if config_path else None)
            self._embedder_configs = self._config.to_embedder_configs()
        self.storage_router = StorageRouter()
        self._configs_by_name = {config.name: config for config in self._embedder_configs}
        self._configs_by_namespace = {config.namespace: config for config in self._embedder_configs}
        self.vector_store = vector_store

    def _resolve_configs(self, request: EmbeddingRequest) -> list[EmbedderConfig]:
        if request.namespaces:
            configs = [self._configs_by_namespace[ns] for ns in request.namespaces if ns in self._configs_by_namespace]
            if configs:
                return configs
        if request.models:
            configs = [self._configs_by_name[name] for name in request.models if name in self._configs_by_name]
            if configs:
                return configs
        active = [self._configs_by_namespace[ns] for ns in self._config.active_namespaces if ns in self._configs_by_namespace]
        if active:
            return active
        return list(self._embedder_configs)

    def _adapter_request(
        self,
        request: EmbeddingRequest,
        config: EmbedderConfig,
        *,
        texts: Sequence[str],
        ids: Sequence[str],
    ) -> AdapterEmbeddingRequest:
        return AdapterEmbeddingRequest(
            tenant_id=request.tenant_id,
            namespace=config.namespace,
            texts=list(texts),
            ids=list(ids),
            correlation_id=request.correlation_id,
        )

    def _dimension_from_record(self, record: EmbeddingRecord) -> int:
        if record.dim:
            return record.dim
        if record.vectors:
            return len(record.vectors[0])
        if record.terms:
            return len(record.terms)
        return 0

    def _batch_iterator(
        self, request: EmbeddingRequest, batch_size: int
    ) -> Iterator[tuple[list[str], list[str]]]:
        texts = list(request.texts)
        ids = list(request.chunk_ids or [])
        if len(ids) < len(texts):
            ids.extend(f"{request.tenant_id}:{index}" for index in range(len(ids), len(texts)))
        else:
            ids = ids[: len(texts)]
        for start in range(0, len(texts), batch_size):
            yield (
                texts[start : start + batch_size],
                ids[start : start + batch_size],
            )

    def run(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if self._legacy_registry is not None:
            return self._run_legacy(request)
        configs = self._resolve_configs(request)
        response = EmbeddingResponse()
        logger.info(
            "embedding.pipeline.start",
            tenant_id=request.tenant_id,
            chunks=len(request.texts),
            models=[config.name for config in configs],
        )
        for config in configs:
            ensure_available(config.requires_gpu, operation=f"embed:{config.name}")
            embedder = self.factory.get(config)
            batches = self._batch_iterator(request, request.batch_size)
            progress = BatchProgress(
                total=len(request.texts),
                callback=lambda processed, total, namespace=config.namespace, model=config.name: logger.info(
                    "embedding.pipeline.progress",
                    namespace=namespace,
                    model=model,
                    processed=processed,
                    total=total,
                ),
            )
            start = time.perf_counter()
            records: list[EmbeddingRecord] = []
            for text_batch, id_batch in batches:
                adapter_request = self._adapter_request(
                    request,
                    config,
                    texts=text_batch,
                    ids=id_batch,
                )
                batch_records = embedder.embed_documents(adapter_request)
                records.extend(batch_records)
                progress.step(len(text_batch))
            if not records:
                logger.warning(
                    "embedding.pipeline.no_output",
                    namespace=config.namespace,
                    model=config.name,
                )
                continue
            first = records[0]
            dimension = self._dimension_from_record(first)
            if dimension:
                self.namespace_manager.introspect_dimension(config.namespace, dimension)
            vector_records: list[VectorRecord] = []
            for record in records:
                dim = self._dimension_from_record(record)
                self.namespace_manager.validate_record(config.namespace, dim)
                self.storage_router.persist(record)
                response.vectors.append(
                    EmbeddingVector(
                        id=record.id,
                        model=record.model_id,
                        namespace=record.namespace,
                        kind=record.kind,
                        vectors=record.vectors,
                        terms=record.terms,
                        dimension=dim,
                        metadata=metadata,
                    )
                )
                if self.vector_store and record.vectors:
                    named_vectors: dict[str, list[float]] | None = None
                    if len(record.vectors) > 1:
                        named_vectors = {
                            f"segment_{idx}": list(vector)
                            for idx, vector in enumerate(record.vectors[1:], start=1)
                        }
                    vector_records.append(
                        VectorRecord(
                            vector_id=record.id,
                            values=list(record.vectors[0]),
                            metadata=metadata,
                            named_vectors=named_vectors,
                        )
                    )
            if self.vector_store and vector_records:
                context = SecurityContext(
                    subject="embedding-worker",
                    tenant_id=request.tenant_id,
                    scopes={"index:write"},
                )
                self.vector_store.upsert(
                    context=context,
                    namespace=config.namespace,
                    records=vector_records,
                )
            logger.info(
                "embedding.pipeline.completed",
                namespace=config.namespace,
                model=config.name,
                total=len(records),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        logger.info("embedding.pipeline.finish", total=len(response.vectors))
        return response

    def _run_legacy(self, request: EmbeddingRequest) -> EmbeddingResponse:
        models = request.models or self._legacy_registry.list_models()
        response = EmbeddingResponse()
        for model_name in models:
            model = self._legacy_registry.get(model_name)
            for chunk_id, text in zip(request.chunk_ids, request.texts, strict=False):
                result = model.embed(chunk_id, text)
                metadata = {"source": "legacy"}
                if model.kind == "dense":
                    response.vectors.append(
                        EmbeddingVector(
                            id=result["id"],
                            model=model.name,
                            namespace=model.name,
                            kind="dense",
                            vectors=[result["values"]],
                            terms=None,
                            dimension=model.dimension,
                            metadata=metadata,
                        )
                    )
                else:
                    response.vectors.append(
                        EmbeddingVector(
                            id=result["id"],
                            model=model.name,
                            namespace=model.name,
                            kind="sparse",
                            vectors=None,
                            terms=result["terms"],
                            dimension=model.dimension,
                            metadata=metadata,
                        )
                    )
        return response


class EmbeddingGrpcService:
    """Async gRPC servicer bridging requests into the embedding worker."""

    def __init__(self, worker: EmbeddingWorker) -> None:
        self.worker = worker

    async def EmbedChunks(self, request, context):  # type: ignore[override]
        embed_request = EmbeddingRequest(
            tenant_id=request.tenant_id,
            chunk_ids=list(request.chunk_ids),
            texts=list(request.texts),
            normalize=request.normalize,
            batch_size=request.batch_size or 8,
            models=list(request.models) or None,
            namespaces=list(request.namespaces) or None,
            correlation_id=getattr(request, "correlation_id", None),
        )
        response = self.worker.run(embed_request)
        from Medical_KG_rev.proto.gen import embedding_pb2  # type: ignore import-error

        reply = embedding_pb2.EmbedChunksResponse()
        for vector in response.vectors:
            message = reply.vectors.add()
            message.chunk_id = vector.id
            message.model = vector.model
            message.kind = vector.kind
            message.namespace = vector.namespace
            message.dimension = vector.dimension
            if vector.vectors:
                for item in vector.vectors:
                    payload = message.vectors.add()
                    payload.values.extend(item)
            if vector.terms:
                message.terms.update(vector.terms)
            message.metadata.update({key: str(value) for key, value in vector.metadata.items()})
        return reply
