"""Framework adapter for LangChain embedding implementations."""

from __future__ import annotations

from dataclasses import dataclass

from ..registry import EmbedderRegistry
from .delegate import DelegateCall, DelegatedFrameworkAdapter


@dataclass(slots=True)
class LangChainEmbedderAdapter(DelegatedFrameworkAdapter):
    document_calls = (
        DelegateCall("embed_documents", "batch"),
        DelegateCall("embed", "batch"),
    )
    query_calls = (
        DelegateCall("embed_query", "per_text"),
        DelegateCall("embed_queries", "batch"),
        DelegateCall("embed_documents", "batch"),
        DelegateCall("embed", "batch"),
    )


def register_langchain(registry: EmbedderRegistry) -> None:
    registry.register("langchain", lambda config: LangChainEmbedderAdapter(config=config))
