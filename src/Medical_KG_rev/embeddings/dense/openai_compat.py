"""OpenAI-compatible embedding adapter used for vLLM-backed services."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional dependency guard
    import httpx
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments

    class _HttpxFallback:
        class HTTPError(Exception):
            pass

        class HTTPStatusError(HTTPError):
            pass

        def post(self, *args: Any, **kwargs: Any):
            raise RuntimeError("httpx is required for network operations")

    httpx = _HttpxFallback()  # type: ignore[assignment]

from Medical_KG_rev.services import GpuNotAvailableError

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..registry import EmbedderRegistry
from ..utils.normalization import normalize_batch


class _HttpError(RuntimeError):
    """Internal error used to unify HTTP error handling."""


@dataclass(slots=True)
class OpenAICompatEmbedder:
    """Delegate embeddings to an OpenAI-compatible `/v1/embeddings` endpoint."""

    config: EmbedderConfig
    _endpoint: str = ""
    _timeout: float = 60.0
    _api_key: str | None = None
    _normalize: bool = False
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        endpoint = params.get("endpoint")
        if not endpoint:
            raise ValueError("OpenAI compatible embedder requires 'endpoint' parameter")
        # The config provides the `/v1` prefix so that requests map to the
        # OpenAI API specification exactly.
        self._endpoint = str(endpoint).rstrip("/")
        self._timeout = float(params.get("timeout", 60))
        self._api_key = params.get("api_key")
        self._normalize = bool(self.config.normalize)
        self.name = self.config.name
        self.kind = self.config.kind

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _request(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "input": texts,
            "model": self.config.model_id,
        }
        try:
            response = httpx.post(
                f"{self._endpoint}/embeddings",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            raise _HttpError(f"Failed to call embeddings endpoint: {exc}") from exc
        if response.status_code == 503:
            detail = "GPU unavailable"
            try:
                error_payload = response.json()
                detail = error_payload.get("error", {}).get("message", detail)
            except Exception:  # pragma: no cover - defensive
                detail = "GPU unavailable"
            raise GpuNotAvailableError(detail)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - propagate with context
            raise _HttpError(str(exc)) from exc
        data = response.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        if not embeddings:
            raise ValueError("OpenAI compatible endpoint returned no embeddings")
        return [list(map(float, vector)) for vector in embeddings]

    # ------------------------------------------------------------------
    # Record helpers
    # ------------------------------------------------------------------
    def _records(
        self,
        request: EmbeddingRequest,
        vectors: Iterable[Iterable[float]],
    ) -> list[EmbeddingRecord]:
        converted = [list(map(float, vector)) for vector in vectors]
        if self._normalize:
            converted = normalize_batch(converted)
        ids = list(
            request.ids or [f"{request.namespace}:{index}" for index in range(len(converted))]
        )
        records: list[EmbeddingRecord] = []
        for chunk_id, vector in zip(ids, converted, strict=False):
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._records(request, self._request(list(request.texts)))

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._records(request, self._request(list(request.texts)))


def register_openai_compat(registry: EmbedderRegistry) -> None:
    factory = lambda config: OpenAICompatEmbedder(config=config)
    registry.register("openai-compat", factory)
    registry.register("vllm", factory)
