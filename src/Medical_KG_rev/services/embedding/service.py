"""Universal embedding service coordinating multiple adapters."""

from __future__ import annotations

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

    def _adapter_request(self, request: EmbeddingRequest, config: EmbedderConfig) -> AdapterEmbeddingRequest:
        return AdapterEmbeddingRequest(
            tenant_id=request.tenant_id,
            namespace=config.namespace,
            texts=list(request.texts),
            ids=list(request.chunk_ids),
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
                total=len(records),
            )
        logger.info("embedding.pipeline.finish", total=len(response.vectors))
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
