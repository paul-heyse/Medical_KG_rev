"""Shared helpers for framework-backed embedding adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from importlib import import_module

from ..ports import EmbedderConfig, EmbeddingRecord, EmbeddingRequest
from ..utils.normalization import normalize_batch
from ..utils.offsets import batch_offsets
from ..utils.records import RecordBuilder


Strategy = Literal["batch", "per_text", "single"]


def _to_float_vector(values: Sequence[object]) -> list[float]:
    return [float(value) for value in values]


@dataclass(slots=True)
class DelegateCall:
    """Descriptor describing how to invoke a delegate embedding method."""

    method: str
    strategy: Strategy
    replicate_result: bool = False
    join_texts: bool = False

    def invoke(self, delegate: object, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        target = getattr(delegate, self.method, None)
        if target is None or not callable(target):  # pragma: no cover - defensive
            raise AttributeError(self.method)
        if self.strategy == "batch":
            result = target(list(texts))
            if not isinstance(result, Sequence):  # pragma: no cover - defensive
                raise TypeError(f"Delegate method '{self.method}' did not return a sequence")
            vectors = [_to_float_vector(vector) for vector in result]
            if len(vectors) != len(texts):
                raise ValueError(
                    f"Delegate method '{self.method}' returned {len(vectors)} vectors for {len(texts)} texts"
                )
            return vectors
        if self.strategy == "per_text":
            vectors: list[list[float]] = []
            for text in texts:
                output = target(text)
                vectors.append(_to_float_vector(output))
            return vectors
        # single strategy
        payload = " ".join(texts) if self.join_texts else texts[0]
        output = target(payload)
        vector = _to_float_vector(output)
        if self.replicate_result:
            return [list(vector) for _ in texts]
        return [vector]


@dataclass(slots=True)
class DelegatedFrameworkAdapter:
    """Base class encapsulating shared delegate behaviour."""

    config: EmbedderConfig
    document_calls: Sequence[DelegateCall] = field(
        default_factory=lambda: (
            DelegateCall("embed_documents", "batch"),
            DelegateCall("embed", "batch"),
        )
    )
    query_calls: Sequence[DelegateCall] = field(
        default_factory=lambda: (
            DelegateCall("embed_query", "per_text"),
            DelegateCall("embed_queries", "batch"),
            DelegateCall("embed_documents", "batch"),
            DelegateCall("embed", "batch"),
        )
    )
    _delegate: object | None = None
    _builder: RecordBuilder | None = None
    _normalize: bool = False
    _offsets: bool = True
    name: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        params = self.config.parameters
        self._delegate = self._load_delegate(params)
        self._normalize = bool(self.config.normalize)
        self._offsets = bool(params.get("include_offsets", True))
        self._builder = RecordBuilder(self.config, normalized_override=self._normalize)
        self.name = self.config.name
        self.kind = self.config.kind

    # ------------------------------------------------------------------
    def embed_documents(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._embed(request, self.document_calls)

    def embed_queries(self, request: EmbeddingRequest) -> list[EmbeddingRecord]:
        return self._embed(request, self.query_calls)

    # ------------------------------------------------------------------
    def _embed(
        self,
        request: EmbeddingRequest,
        calls: Sequence[DelegateCall],
    ) -> list[EmbeddingRecord]:
        delegate = self._delegate
        if delegate is None:  # pragma: no cover - safety guard
            raise RuntimeError("Delegate not initialised")
        texts = list(request.texts)
        vectors = self._invoke(delegate, calls, texts)
        if self._normalize and vectors:
            vectors = normalize_batch(vectors)
        assert self._builder is not None
        offsets = batch_offsets(texts) if self._offsets else None
        return self._builder.dense(
            request,
            vectors,
            dim=len(vectors[0]) if vectors else None,
            offsets=offsets,
        )

    def _invoke(
        self,
        delegate: object,
        calls: Sequence[DelegateCall],
        texts: Sequence[str],
    ) -> list[list[float]]:
        errors: list[Exception] = []
        for call in calls:
            try:
                vectors = call.invoke(delegate, texts)
            except AttributeError:
                continue
            except Exception as exc:  # pragma: no cover - fallback to next call
                errors.append(exc)
                continue
            if vectors:
                return vectors
            if not texts:
                return []
        if errors:  # pragma: no cover - assists debugging
            raise RuntimeError(
                "All delegate calls failed: " + ", ".join(str(error) for error in errors)
            )
        raise RuntimeError("No compatible delegate method found for adapter")

    def _load_delegate(self, params: dict[str, object]) -> object:
        if "class_path" not in params:
            raise ValueError("Framework adapter requires 'class_path' parameter")
        target = str(params["class_path"])
        module_name, _, class_name = target.rpartition(".")
        module = import_module(module_name)
        cls = getattr(module, class_name)
        init_kwargs = params.get("init", {})
        if init_kwargs is None:
            init_kwargs = {}
        if not isinstance(init_kwargs, dict):
            raise ValueError("Framework adapter 'init' parameter must be a mapping")
        return cls(**init_kwargs)
