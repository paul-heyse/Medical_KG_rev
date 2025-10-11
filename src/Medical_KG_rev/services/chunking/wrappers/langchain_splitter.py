"""Wrapper around langchain text splitters."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from Medical_KG_rev.models.ir import Document

from ..port import Chunk
from ..port import register_chunker
from .base import BaseProfileChunker


register_chunker(
LangChainChunker.name,
lambda *, profile: LangChainChunker(profile=profile),
)
