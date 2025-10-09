"""Neural sparse adapter for OpenSearch ML plugin integrations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import structlog

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry
from ..utils.records import RecordBuilder

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class OpenSearchNeuralSparseEmbedder:
    config: EmbedderConfig
    _neural_field: str = "neural_embedding"
    _ml_model_id: str | None = None
    _external_endpoint: str | None = None
    _builder: RecordBuilder | None = None
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        self._neural_field = params.get("field", "neural_embedding")
        self._ml_model_id = params.get("ml_model_id")
        self._external_endpoint = params.get("external_endpoint")
        self._builder = RecordBuilder(self.config, normalized_override=self.config.normalize)
        self.name = self.config.name
        self.kind = self.config.kind

    def _neural_vector(self, text: str, dim: int) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeats = (dim * 4 + len(digest) - 1) // len(digest)
        data = (digest * repeats)[: dim * 4]
        ints = [int.from_bytes(data[i : i + 4], "big") for i in range(0, len(data), 4)]
        scale = float(2**32)
        return [(value / scale) * 2 - 1 for value in ints][:dim]

    def _metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {"provider": self.config.provider}
        if self._ml_model_id:
            metadata["ml_model_id"] = self._ml_model_id
        if self._external_endpoint:
            metadata["external_encoder"] = self._external_endpoint
        return metadata

    def _neural_query(self, text: str, namespace: str) -> dict[str, object]:
        return {
            "neural": {
                "query": text,
                "field": self._neural_field,
                "model_id": self._ml_model_id or "inline",
                "namespace": namespace,
            }
        }

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        assert self._builder is not None
        dim = int(self.config.dim or 768)
        vectors = [self._neural_vector(text, dim) for text in request.texts]
        return self._builder.neural_sparse(
            request,
            vectors,
            field_name=self._neural_field,
            dim=dim,
            extra_metadata=self._metadata(),
        )

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        assert self._builder is not None
        dim = int(self.config.dim or 768)
        vectors = [self._neural_vector(text, dim) for text in request.texts]
        combined_query = " ".join(request.texts)
        query_payload = self._neural_query(combined_query, request.namespace)
        extras = [
            {**self._metadata(), "neural_query": query_payload}
            for _ in vectors
        ]
        documents = self._builder.neural_sparse(
            request,
            vectors,
            field_name=self._neural_field,
            dim=dim,
            extra_metadata=extras,
        )
        logger.debug(
            "opensearch.neural.query",
            namespace=request.namespace,
            external_encoder=self._external_endpoint,
        )
        return documents


def register_neural_sparse(registry: EmbedderRegistry) -> None:
    registry.register("opensearch-neural", lambda config: OpenSearchNeuralSparseEmbedder(config=config))
