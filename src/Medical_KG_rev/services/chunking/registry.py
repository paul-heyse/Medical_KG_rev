"""Registration helpers for chunker implementations."""

from __future__ import annotations

from .wrappers import langchain_splitter, llamaindex_parser, simple


def register_defaults() -> None:
    """Register built-in chunker implementations."""

    simple.register()
    try:
        langchain_splitter.register()
    except RuntimeError:
        # LangChain dependencies are optional at runtime; environments without
        # the dependency may still rely on the simple chunker.
        pass
    try:
        llamaindex_parser.register()
    except RuntimeError:
        # LlamaIndex is optional; fallback behaviour is provided.
        pass
