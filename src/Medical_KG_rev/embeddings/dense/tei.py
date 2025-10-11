"""HuggingFace Text Embeddings Inference HTTP adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry
from ..utils.normalization import normalize_batch



@dataclass(slots=True)
class TEIHTTPEmbedder:
    config: EmbedderConfig
    _endpoint: str = ""
    _timeout: float = 60.0
    _headers: dict[str, str] = field(default_factory=dict)
    _normalize: bool = False
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        if "endpoint" not in params:
            raise ValueError("TEI embedder requires 'endpoint' parameter")
        self._endpoint = str(params["endpoint"]).rstrip("/")
        self._timeout = float(params.get("timeout", 60))
        self._headers = params.get("headers") or {}
        self._normalize = bool(self.config.normalize)
        self.name = self.config.name
        self.kind = self.config.kind

    def _request(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"inputs": texts}
        response = httpx.post(
            f"{self._endpoint}/embeddings",
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "data" in data:
            vectors = [item["embedding"] for item in data["data"]]
        else:
            vectors = data
        if not isinstance(vectors, list):
            raise ValueError("Unexpected TEI response format")
        return [list(map(float, vector)) for vector in vectors]

    def _records(
        self, request: EmbeddingRequest, vectors: list[list[float]]
    ) -> list[EmbeddingRecord]:
        if self._normalize:
            vectors = normalize_batch(vectors)
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
        vectors = self._request(list(request.texts))
        return self._records(request, vectors)

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        vectors = self._request(list(request.texts))
        return self._records(request, vectors)


def register_tei(registry: EmbedderRegistry) -> None:
    registry.register("tei", lambda config: TEIHTTPEmbedder(config=config))
