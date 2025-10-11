"""Registration helpers for chunker implementations."""

from __future__ import annotations

from . import profile_chunkers
from .wrappers import langchain_splitter, llamaindex_parser, simple



def register_defaults() -> None:
    """Register built-in chunker implementations."""
    simple.register()
    langchain_splitter.register()
    llamaindex_parser.register()
    profile_chunkers.register()
