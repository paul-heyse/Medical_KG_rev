"""Lightweight simple chunker used in tests."""

from __future__ import annotations

from typing import Any

from Medical_KG_rev.models.ir import Document

from ..port import Chunk
from ..port import register_chunker
from .base import BaseProfileChunker


register_chunker(SimpleChunker.name, lambda *, profile: SimpleChunker(profile=profile))
