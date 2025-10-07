"""Adapter for llama_index embedding classes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry
from ..utils.normalization import normalize_batch


@dataclass(slots=True)
class LlamaIndexEmbedderAdapter:
    config: EmbedderConfig
    _delegate: object | None = None
    _normalize: bool = False
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        target = params.get("class_path")
        if not target:
            raise ValueError("LlamaIndex adapter requires 'class_path' parameter")
        module_name, _, class_name = str(target).rpartition(".")
        module = import_module(module_name)
        cls = getattr(module, class_name)
        init_kwargs = params.get("init", {})
        self._delegate = cls(**init_kwargs)
        self._normalize = bool(self.config.normalize)
        self.name = self.config.name
        self.kind = self.config.kind

    def _call(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self._delegate, "get_text_embedding"):
            vectors = [self._delegate.get_text_embedding(text) for text in texts]
        elif hasattr(self._delegate, "embed_documents"):
            vectors = self._delegate.embed_documents(texts)  # type: ignore[attr-defined]
        else:
            vectors = self._delegate.embed(texts)  # type: ignore[attr-defined]
        if self._normalize:
            vectors = normalize_batch(vectors)
        return [list(map(float, vector)) for vector in vectors]

    def _records(self, request: EmbeddingRequest, vectors: list[list[float]]) -> list[EmbeddingRecord]:
        ids = list(request.ids or [f"{request.namespace}:{index}" for index in range(len(vectors))])
        records: list[EmbeddingRecord] = []
        for chunk_id, vector in zip(ids, vectors, strict=False):
            records.append(
                EmbeddingRecord(
                    id=chunk_id,
                    tenant_id=request.tenant_id,
                    namespace=request.namespace,
                    model_id=self.config.model_id,
                    model_version=self.config.model_version,
                    kind=self.config.kind,
                    dim=len(vector),
                    vectors=[vector],
                    normalized=self._normalize,
                    metadata={"provider": self.config.provider},
                    correlation_id=request.correlation_id,
                )
            )
        return records

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._records(request, self._call(list(request.texts)))

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._records(request, self._call(list(request.texts)))


def register_llama_index(registry: EmbedderRegistry) -> None:
    registry.register("llama-index", lambda config: LlamaIndexEmbedderAdapter(config=config))
