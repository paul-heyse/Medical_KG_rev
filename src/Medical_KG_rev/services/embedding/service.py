"""Universal embedding service coordinating multiple adapters."""

from __future__ import annotations

"""Embedding worker service coordinating adapter execution and persistence."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import structlog

from Medical_KG_rev.embeddings.ports import (
    EmbedderConfig,
    EmbeddingRecord,
    EmbeddingRequest as AdapterEmbeddingRequest,
)
from Medical_KG_rev.embeddings.namespace import NamespaceManager
from Medical_KG_rev.embeddings.utils.gpu import ensure_available
from Medical_KG_rev.services.vector_store.models import VectorRecord
from Medical_KG_rev.services.vector_store.service import VectorStoreService
from Medical_KG_rev.auth.context import SecurityContext

from .registry import EmbeddingModelRegistry

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
    metadatas: Sequence[Mapping[str, object]] | None = None
    actor: str | None = None


@dataclass(slots=True)
class EmbeddingVector:
    id: str
    model: str
    namespace: str
    kind: str
    vectors: list[list[float]] | None
    terms: dict[str, float] | None
    neural_fields: dict[str, object] | None = None
    dimension: int = 0
    model_version: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    record: EmbeddingRecord | None = None

    @classmethod
    def from_record(
        cls,
        record: EmbeddingRecord,
        *,
        storage_router: StorageRouter | None = None,
    ) -> "EmbeddingVector":
        if record.dim is not None:
            dimension = record.dim
        elif record.vectors:
            dimension = len(record.vectors[0])
        elif record.terms:
            dimension = len(record.terms)
        else:
            dimension = 0
        metadata = dict(record.metadata)
        if storage_router is not None:
            try:
                target = storage_router.route(record.kind).name
            except KeyError:  # pragma: no cover - defensive guard
                target = "unmapped"
            metadata.setdefault("storage_target", target)
        metadata.setdefault("model_version", record.model_version)
        return cls(
            id=record.id,
            model=record.model_id,
            namespace=record.namespace,
            kind=record.kind,
            vectors=record.vectors,
            terms=record.terms,
            neural_fields=record.neural_fields,
            dimension=dimension,
            model_version=record.model_version,
            metadata=metadata,
            record=record,
        )


@dataclass(slots=True)
class _VectorBuilder:
    """Helper that converts adapter records into response vectors."""

    storage_router: StorageRouter

    def build(self, record: EmbeddingRecord) -> EmbeddingVector:
        return EmbeddingVector.from_record(record, storage_router=self.storage_router)


@dataclass(slots=True)
class EmbeddingResponse:
    vectors: list[EmbeddingVector] = field(default_factory=list)

    @property
    def values(self) -> list[float] | None:
        """Compatibility accessor returning the primary dense payload."""

        if self.vectors:
            # Return a copy so downstream normalization does not mutate
            # the stored payload which may be re-used by other adapters.
            return list(self.vectors[0])
        return None


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
        registry: EmbeddingModelRegistry | None = None,
        *,
        namespace_manager: NamespaceManager | None = None,
        config_path: str | None = None,
        vector_store: VectorStoreService | None = None,
    ) -> None:
        if registry is None:
            registry = EmbeddingModelRegistry(
                namespace_manager=namespace_manager,
                config_path=config_path,
            )
        self.model_registry = registry
        self.namespace_manager = registry.namespace_manager
        self.storage_router = registry.storage_router
        self.registry = registry.registry
        self.factory = registry.factory
        self.vector_store = vector_store

    def _resolve_configs(self, request: EmbeddingRequest) -> list[EmbedderConfig]:
        return self.model_registry.resolve(
            models=request.models,
            namespaces=request.namespaces,
        )

    def _adapter_request(
        self,
        request: EmbeddingRequest,
        config: EmbedderConfig,
        *,
        texts: Sequence[str],
        ids: Sequence[str],
        metadata: Sequence[Mapping[str, object]],
    ) -> AdapterEmbeddingRequest:
        return AdapterEmbeddingRequest(
            tenant_id=request.tenant_id,
            namespace=config.namespace,
            texts=list(texts),
            ids=list(ids),
            correlation_id=request.correlation_id,
            metadata=[dict(item) for item in metadata],
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
    ) -> Iterator[tuple[list[str], list[str], list[dict[str, object]]]]:
        texts = list(request.texts)
        ids = list(request.chunk_ids or [])
        if len(ids) < len(texts):
            ids.extend(f"{request.tenant_id}:{index}" for index in range(len(ids), len(texts)))
        else:
            ids = ids[: len(texts)]
        metadata_list = [
            dict(item) for item in list(request.metadatas or [])[: len(texts)]
        ]
        while len(metadata_list) < len(texts):
            metadata_list.append({})
        for start in range(0, len(texts), batch_size):
            yield (
                texts[start : start + batch_size],
                ids[start : start + batch_size],
                metadata_list[start : start + batch_size],
            )

    def run(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if self._legacy_registry is not None:
            return self._run_legacy(request)
        configs = self._resolve_configs(request)
        response = EmbeddingResponse()
        logger.info(
            "embedding.pipeline.start",
            tenant_id=request.tenant_id,
            actor=request.actor,
            chunks=len(request.texts),
            models=[config.name for config in configs],
        )
        for config in configs:
            ensure_available(config.requires_gpu, operation=f"embed:{config.name}")
            try:
                embedder = self.model_registry.get(config)
            except KeyError:
                embedder = self.factory.get(config)
            adapter_request = self._adapter_request(request, config)
            records = embedder.embed_documents(adapter_request)
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
                metadata = {
                    **record.metadata,
                    "storage_target": self.storage_router.route(record.kind).name,
                }
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
                actor=request.actor,
                total=len(records),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        logger.info(
            "embedding.pipeline.finish",
            total=len(response.vectors),
            actor=request.actor,
            tenant_id=request.tenant_id,
        )
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
                ids=request.chunk_ids
                or [f"query:{index}" for index in range(len(request.texts))],
                metadata=request.metadatas or [{} for _ in request.texts],
            )
            records = embedder.embed_queries(adapter_request)
            for record in records:
                dim = self._dimension_from_record(record)
                self.namespace_manager.validate_record(config.namespace, dim)
                response.vectors.append(self._vector_builder.build(record))
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
