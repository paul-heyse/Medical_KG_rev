"""Universal embedding service coordinating multiple adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path
from typing import Sequence

import structlog

from Medical_KG_rev.config.embeddings import load_embeddings_config
from Medical_KG_rev.embeddings.namespace import NamespaceManager
from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest as AdapterEmbeddingRequest
from Medical_KG_rev.embeddings.providers import register_builtin_embedders
from Medical_KG_rev.embeddings.registry import EmbedderFactory, EmbedderRegistry
from Medical_KG_rev.embeddings.storage import StorageRouter
from Medical_KG_rev.embeddings.utils.batching import BatchProgress, iter_with_progress
from Medical_KG_rev.embeddings.utils.gpu import ensure_available

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
class EmbeddingModelRegistry:
    """Compatibility wrapper that exposes cached embedder instances by name."""

    namespace_manager: NamespaceManager
    registry: EmbedderRegistry
    factory: EmbedderFactory
    _config: object
    _embedder_configs: list[EmbedderConfig]
    _configs_by_name: dict[str, EmbedderConfig]

    def __init__(self, _gpu_manager: object | None = None, *, config_path: str | None = None) -> None:
        self.namespace_manager = NamespaceManager()
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

    def _batch_iterator(self, request: EmbeddingRequest, batch_size: int) -> tuple[list[list[str]], list[list[str]]]:
        texts = list(request.texts)
        ids = list(request.chunk_ids)
        text_batches = list(iter_with_progress(texts, batch_size))
        id_batches = list(iter_with_progress(ids or [f"{request.tenant_id}:{index}" for index in range(len(texts))], batch_size))
        return text_batches, id_batches

    def run(self, request: EmbeddingRequest) -> EmbeddingResponse:
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
            text_batches, id_batches = self._batch_iterator(request, request.batch_size)
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
            for text_batch, id_batch in zip(text_batches, id_batches, strict=True):
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
                        metadata={
                            **record.metadata,
                            "storage_target": self.storage_router.route(record.kind).name,
                        },
                    )
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

    def encode_queries(self, request: EmbeddingRequest) -> EmbeddingResponse:
        configs = self._resolve_configs(request)
        response = EmbeddingResponse()
        for config in configs:
            ensure_available(config.requires_gpu, operation=f"embed-query:{config.name}")
            embedder = self.factory.get(config)
            adapter_request = self._adapter_request(
                request,
                config,
                texts=request.texts,
                ids=request.chunk_ids or [f"query:{index}" for index in range(len(request.texts))],
            )
            records = embedder.embed_queries(adapter_request)
            for record in records:
                dim = self._dimension_from_record(record)
                self.namespace_manager.validate_record(config.namespace, dim)
                response.vectors.append(
                    EmbeddingVector(
                        id=record.id,
                        model=record.model_id,
                        namespace=record.namespace,
                        kind=record.kind,
                        vectors=record.vectors,
                        terms=record.terms,
                        dimension=dim,
                        metadata={**record.metadata},
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
