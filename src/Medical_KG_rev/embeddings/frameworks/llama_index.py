"""Adapter for llama_index embedding classes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry
from ..utils.normalization import normalize_batch
from ..utils.offsets import batch_offsets


@dataclass(slots=True)
class LlamaIndexEmbedderAdapter:
    config: EmbedderConfig
    _delegate: object | None = None
    _normalize: bool = False
    _offsets: bool = True
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        self._validate_parameters(params)
        target = params["class_path"]
        module_name, _, class_name = str(target).rpartition(".")
        module = import_module(module_name)
        cls = getattr(module, class_name)
        init_kwargs = params.get("init", {})
        if not isinstance(init_kwargs, dict):
            raise ValueError("LlamaIndex adapter 'init' parameter must be a mapping")
        self._delegate = cls(**init_kwargs)
        self._normalize = bool(self.config.normalize)
        self._offsets = bool(params.get("include_offsets", True))
        self.name = self.config.name
        self.kind = self.config.kind

    def _validate_parameters(self, params: dict[str, object]) -> None:
        if "class_path" not in params:
            raise ValueError("LlamaIndex adapter requires 'class_path' parameter")

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

    def _records(
        self, request: EmbeddingRequest, vectors: list[list[float]], texts: list[str]
    ) -> list[EmbeddingRecord]:
        ids = list(request.ids or [f"{request.namespace}:{index}" for index in range(len(vectors))])
        offsets = batch_offsets(texts) if self._offsets else [[] for _ in texts]
        records: list[EmbeddingRecord] = []
        for chunk_id, vector, offset_map in zip(ids, vectors, offsets, strict=False):
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
                    metadata={
                        "provider": self.config.provider,
                        "offsets": offset_map,
                    },
                    correlation_id=request.correlation_id,
                )
            )
        return records

    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        texts = list(request.texts)
        return self._records(request, self._call(texts), texts)

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        texts = list(request.texts)
        return self._records(request, self._call(texts), texts)


def register_llama_index(registry: EmbedderRegistry) -> None:
    registry.register("llama-index", lambda config: LlamaIndexEmbedderAdapter(config=config))
