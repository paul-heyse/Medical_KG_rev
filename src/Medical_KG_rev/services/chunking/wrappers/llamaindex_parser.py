"""Chunker implementation that mimics LlamaIndex sentence window parsing."""

from __future__ import annotations

from typing import Any

from llama_index.core import Document as LlamaDocument  # type: ignore
from llama_index.core.node_parser import HttpClient
from Medical_KG_rev.models.ir import Document

from ..port import Chunk
from ..port import register_chunker
from .base import BaseProfileChunker


register_chunker(LlamaIndexChunker.name, lambda *, profile: LlamaIndexChunker(profile=profile))


__all__ = ["LlamaIndexChunker", "register"]
