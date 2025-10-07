"""Adapter for llama_index embedding classes."""

from __future__ import annotations

"""LlamaIndex embedding adapter built atop the delegate helper."""

from dataclasses import dataclass

from ..registry import EmbedderRegistry
from .delegate import DelegateCall, DelegatedFrameworkAdapter


@dataclass(slots=True)
class LlamaIndexEmbedderAdapter(DelegatedFrameworkAdapter):
    document_calls = (
        DelegateCall("get_text_embedding", "per_text"),
        DelegateCall("embed_documents", "batch"),
        DelegateCall("embed", "batch"),
    )
    query_calls = (
        DelegateCall("get_query_embedding", "per_text"),
        DelegateCall("get_text_embedding", "per_text"),
        DelegateCall("embed_queries", "batch"),
        DelegateCall("embed_documents", "batch"),
        DelegateCall("embed", "batch"),
    )


def register_llama_index(registry: EmbedderRegistry) -> None:
    registry.register("llama-index", lambda config: LlamaIndexEmbedderAdapter(config=config))
