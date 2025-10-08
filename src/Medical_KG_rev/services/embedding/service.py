"""Embedding service orchestrating namespace-aware embedding adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import structlog

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.embeddings.namespace import NamespaceManager
from Medical_KG_rev.embeddings.ports import (
    EmbedderConfig,
    EmbeddingRecord,
)
from Medical_KG_rev.embeddings.ports import (
    EmbeddingRequest as AdapterEmbeddingRequest,
)
from Medical_KG_rev.embeddings.registry import EmbedderFactory, EmbedderRegistry
from Medical_KG_rev.embeddings.storage import StorageRouter
from Medical_KG_rev.embeddings.utils.gpu import ensure_available
from Medical_KG_rev.embeddings.utils.tokenization import (
    TokenLimitExceededError,
    TokenizerCache,
)
from Medical_KG_rev.services.vector_store.models import VectorRecord
from Medical_KG_rev.services.vector_store.service import VectorStoreService

from .registry import EmbeddingModelRegistry

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class EmbeddingRequest:
    """Service-level embedding request used by orchestration."""

    tenant_id: str
    texts: Sequence[str]
    chunk_ids: Sequence[str] | None = None
    models: Sequence[str] | None = None
    namespaces: Sequence[str] | None = None
    correlation_id: str | None = None
    metadatas: Sequence[Mapping[str, object]] | None = None
    actor: str | None = None


@dataclass(slots=True)
class EmbeddingVector:
    """High-level representation of an embedding returned to callers."""

    id: str
    model: str
    namespace: str
    kind: str
    vectors: list[list[float]] | None = None
    terms: dict[str, float] | None = None
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
        storage_router: StorageRouter,
    ) -> "EmbeddingVector":
        if record.dim is not None:
            dimension = record.dim
        elif record.vectors:
            dimension = len(record.vectors[0])
        elif record.terms:
            dimension = len(record.terms)
        else:
            dimension = 0
        metadata = {"provider": record.metadata.get("provider", ""), **record.metadata}
        try:
            metadata.setdefault("storage_target", storage_router.route(record.kind).name)
        except KeyError:  # pragma: no cover - defensive guard
            metadata.setdefault("storage_target", "unknown")
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
class EmbeddingResponse:
    vectors: list[EmbeddingVector] = field(default_factory=list)


@dataclass(slots=True)
class EmbeddingWorker:
    """Coordinates config-driven embedding generation and storage."""

    model_registry: EmbeddingModelRegistry | None = None
    namespace_manager: NamespaceManager | None = None
    vector_store: VectorStoreService | None = None
    config_path: str | None = None
    tokenizer_cache: TokenizerCache | None = None
    registry: EmbedderRegistry | None = field(init=False, default=None)
    factory: EmbedderFactory | None = field(init=False, default=None)
    storage_router: StorageRouter | None = field(init=False, default=None)
    tokenizers: TokenizerCache | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.model_registry is None:
            self.model_registry = EmbeddingModelRegistry(
                namespace_manager=self.namespace_manager,
                config_path=self.config_path,
            )
        self.namespace_manager = self.model_registry.namespace_manager
        self.registry = self.model_registry.registry
        self.factory = self.model_registry.factory
        self.storage_router = self.model_registry.storage_router
        self.tokenizers = self.tokenizer_cache or TokenizerCache()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
    ) -> AdapterEmbeddingRequest:
        ids = list(request.chunk_ids or [])
        if len(ids) < len(texts):
            ids.extend(
                f"{request.tenant_id}:{index}" for index in range(len(ids), len(texts))
            )
        else:
            ids = ids[: len(texts)]
        metadata = [dict(item) for item in list(request.metadatas or [])[: len(texts)]]
        while len(metadata) < len(texts):
            metadata.append({})
        return AdapterEmbeddingRequest(
            tenant_id=request.tenant_id,
            namespace=config.namespace,
            texts=list(texts),
            ids=ids,
            correlation_id=request.correlation_id,
            metadata=metadata,
        )

    def _dimension_from_record(self, record: EmbeddingRecord) -> int:
        if record.dim is not None:
            return record.dim
        if record.vectors:
            return len(record.vectors[0])
        if record.terms:
            return len(record.terms)
        return 0

    def _persist_dense(
        self,
        *,
        request: EmbeddingRequest,
        config: EmbedderConfig,
        vectors: list[EmbeddingRecord],
    ) -> None:
        if not self.vector_store:
            return
        records: list[VectorRecord] = []
        for record in vectors:
            if not record.vectors:
                continue
            named_vectors: dict[str, list[float]] | None = None
            if len(record.vectors) > 1:
                named_vectors = {
                    f"segment_{index}": list(vector)
                    for index, vector in enumerate(record.vectors[1:], start=1)
                }
            records.append(
                VectorRecord(
                    vector_id=record.id,
                    values=list(record.vectors[0]),
                    metadata=dict(record.metadata),
                    named_vectors=named_vectors,
                )
            )
        if not records:
            return
        context = SecurityContext(
            subject="embedding-worker",
            tenant_id=request.tenant_id,
            scopes={"index:write"},
        )
        self.vector_store.upsert(
            context=context,
            namespace=config.namespace,
            records=records,
        )

    def _token_budget(self, config: EmbedderConfig) -> int | None:
        return config.parameters.get("max_tokens") if config.parameters else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, request: EmbeddingRequest) -> EmbeddingResponse:
        configs = self._resolve_configs(request)
        response = EmbeddingResponse()
        texts = list(request.texts)
        for config in configs:
            ensure_available(config.requires_gpu, operation=f"embed:{config.namespace}")
            try:
                self.tokenizers.ensure_within_limit(
                    model_id=config.model_id,
                    texts=texts,
                    max_tokens=self._token_budget(config),
                    correlation_id=request.correlation_id,
                )
            except TokenLimitExceededError:
                logger.error(
                    "embedding.token_limit_exceeded",
                    namespace=config.namespace,
                    model=config.model_id,
                    correlation_id=request.correlation_id,
                )
                raise
            embedder = self.model_registry.get(config)
            adapter_request = self._adapter_request(request, config, texts=texts)
            records = embedder.embed_documents(adapter_request)
            if not records:
                logger.warning(
                    "embedding.no_records",
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
                vector = EmbeddingVector.from_record(record, storage_router=self.storage_router)
                response.vectors.append(vector)
            self._persist_dense(request=request, config=config, vectors=records)
            logger.info(
                "embedding.namespace.completed",
                namespace=config.namespace,
                model=config.name,
                records=len(records),
            )
        return response

    def encode_queries(self, request: EmbeddingRequest) -> EmbeddingResponse:
        configs = self._resolve_configs(request)
        response = EmbeddingResponse()
        for config in configs:
            ensure_available(config.requires_gpu, operation=f"embed-query:{config.namespace}")
            embedder = self.model_registry.get(config)
            adapter_request = self._adapter_request(request, config, texts=request.texts)
            records = embedder.embed_queries(adapter_request)
            for record in records:
                dim = self._dimension_from_record(record)
                self.namespace_manager.validate_record(config.namespace, dim)
                response.vectors.append(
                    EmbeddingVector.from_record(record, storage_router=self.storage_router)
                )
        return response


@dataclass(slots=True)
class EmbeddingGrpcService:
    """Async gRPC servicer bridging requests into the embedding worker."""

    worker: EmbeddingWorker

    async def EmbedChunks(self, request, context):  # type: ignore[override]
        embed_request = EmbeddingRequest(
            tenant_id=request.tenant_id,
            texts=list(request.texts),
            chunk_ids=list(request.chunk_ids),
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
                for payload in vector.vectors:
                    message_payload = message.vectors.add()
                    message_payload.values.extend(payload)
            if vector.terms:
                message.terms.update(vector.terms)
            message.metadata.update({key: str(value) for key, value in vector.metadata.items()})
        return reply


__all__ = [
    "EmbeddingGrpcService",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingVector",
    "EmbeddingWorker",
]
