from __future__ import annotations

from dataclasses import replace
from math import isclose, sqrt
from typing import Any

import pytest

from Medical_KG_rev.embeddings.dense import openai_compat
from Medical_KG_rev.embeddings.ports import EmbedderConfig, EmbeddingRequest
from Medical_KG_rev.services import GpuNotAvailableError


class FakeHttpx:
    class HTTPError(Exception):
        pass

    class HTTPStatusError(HTTPError):
        pass

    def __init__(self) -> None:
        self._response: FakeResponse | None = None
        self._exception: Exception | None = None
        self.calls: list[dict[str, Any]] = []

    def configure(
        self,
        *,
        response: "FakeResponse" | None = None,
        exception: Exception | None = None,
    ) -> None:
        self._response = response
        self._exception = exception

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: float):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self._exception is not None:
            raise self._exception
        assert self._response is not None, "FakeHttpx must be configured with a response"
        return self._response


class FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {"data": [{"embedding": [1.0, 0.0]}]}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise FakeHttpx.HTTPStatusError(f"HTTP {self.status_code}")


@pytest.fixture()
def embedder_config() -> EmbedderConfig:
    return EmbedderConfig(
        name="qwen3",
        provider="vllm",
        kind="single_vector",
        namespace="single_vector.qwen3.4096.v1",
        model_id="Qwen/Qwen2.5-Embedding-8B-Instruct",
        model_version="v1",
        dim=4096,
        parameters={"endpoint": "http://localhost:8001/v1", "timeout": 5},
    )


@pytest.fixture()
def fake_httpx(monkeypatch: pytest.MonkeyPatch) -> FakeHttpx:
    client = FakeHttpx()
    monkeypatch.setattr(openai_compat, "httpx", client)
    return client


def _request(texts: list[str]) -> EmbeddingRequest:
    return EmbeddingRequest(
        tenant_id="tenant",
        namespace="single_vector.qwen3.4096.v1",
        texts=texts,
        ids=[f"chunk-{i}" for i in range(len(texts))],
    )


def test_openai_embedder_requires_endpoint() -> None:
    config = EmbedderConfig(
        name="bad",
        provider="vllm",
        kind="single_vector",
        namespace="single_vector.bad.1024.v1",
        model_id="demo",
        dim=1024,
        parameters={},
    )
    with pytest.raises(ValueError):
        openai_compat.OpenAICompatEmbedder(config=config)


def test_openai_embedder_normalizes_vectors(embedder_config, fake_httpx: FakeHttpx) -> None:
    fake_httpx.configure(response=FakeResponse(payload={"data": [{"embedding": [3.0, 4.0]}]}))
    embedder = openai_compat.OpenAICompatEmbedder(config=embedder_config)
    records = embedder.embed_documents(_request(["a"]))
    vector = records[0].vectors[0]
    assert isclose(sqrt(sum(value * value for value in vector)), 1.0)


def test_openai_embedder_respects_disable_normalization(
    embedder_config, fake_httpx: FakeHttpx
) -> None:
    config = replace(embedder_config, normalize=False)
    fake_httpx.configure(response=FakeResponse(payload={"data": [{"embedding": [0.25, 0.75]}]}))
    embedder = openai_compat.OpenAICompatEmbedder(config=config)
    vector = embedder.embed_documents(_request(["text"]))[0].vectors[0]
    assert vector == [0.25, 0.75]


def test_openai_embedder_includes_api_key(fake_httpx: FakeHttpx, embedder_config) -> None:
    config = replace(
        embedder_config,
        parameters={"endpoint": "http://localhost:8001/v1", "api_key": "secret"},
    )
    fake_httpx.configure(response=FakeResponse())
    embedder = openai_compat.OpenAICompatEmbedder(config=config)
    embedder.embed_documents(_request(["t"]))
    assert fake_httpx.calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_openai_embedder_raises_for_gpu_unavailable(embedder_config, fake_httpx: FakeHttpx) -> None:
    payload = {"error": {"message": "CUDA out of memory"}, "data": []}
    fake_httpx.configure(response=FakeResponse(status_code=503, payload=payload))
    embedder = openai_compat.OpenAICompatEmbedder(config=embedder_config)
    with pytest.raises(GpuNotAvailableError):
        embedder.embed_documents(_request(["boom"]))


def test_openai_embedder_raises_http_error(embedder_config, fake_httpx: FakeHttpx) -> None:
    fake_httpx.configure(response=FakeResponse(status_code=500))
    embedder = openai_compat.OpenAICompatEmbedder(config=embedder_config)
    with pytest.raises(openai_compat._HttpError):
        embedder.embed_documents(_request(["error"]))


def test_openai_embedder_wraps_network_error(embedder_config, fake_httpx: FakeHttpx) -> None:
    fake_httpx.configure(exception=FakeHttpx.HTTPError("network down"))
    embedder = openai_compat.OpenAICompatEmbedder(config=embedder_config)
    with pytest.raises(openai_compat._HttpError):
        embedder.embed_documents(_request(["fail"]))


def test_openai_embedder_validates_embeddings_present(
    embedder_config, fake_httpx: FakeHttpx
) -> None:
    fake_httpx.configure(response=FakeResponse(payload={"data": []}))
    embedder = openai_compat.OpenAICompatEmbedder(config=embedder_config)
    with pytest.raises(ValueError):
        embedder.embed_documents(_request(["missing"]))


def test_openai_embedder_supports_query_embedding(embedder_config, fake_httpx: FakeHttpx) -> None:
    fake_httpx.configure(response=FakeResponse(payload={"data": [{"embedding": [1.0, 0.0]}]}))
    embedder = openai_compat.OpenAICompatEmbedder(config=embedder_config)
    result = embedder.embed_queries(_request(["query"]))
    assert result[0].metadata["provider"] == "vllm"


def test_openai_embedder_preserves_correlation_id(embedder_config, fake_httpx: FakeHttpx) -> None:
    fake_httpx.configure(response=FakeResponse())
    embedder = openai_compat.OpenAICompatEmbedder(config=embedder_config)
    request = _request(["body"])
    request = EmbeddingRequest(
        tenant_id=request.tenant_id,
        namespace=request.namespace,
        texts=request.texts,
        ids=request.ids,
        correlation_id="cid-123",
    )
    record = embedder.embed_documents(request)[0]
    assert record.correlation_id == "cid-123"


def test_openai_embedder_propagates_metadata(embedder_config, fake_httpx: FakeHttpx) -> None:
    fake_httpx.configure(response=FakeResponse())
    embedder = openai_compat.OpenAICompatEmbedder(config=embedder_config)
    request = EmbeddingRequest(
        tenant_id="tenant",
        namespace=embedder_config.namespace,
        texts=["data"],
        ids=["chunk-1"],
        metadata=[{"source": "unit"}],
    )
    record = embedder.embed_documents(request)[0]
    assert record.metadata["provider"] == "vllm"
