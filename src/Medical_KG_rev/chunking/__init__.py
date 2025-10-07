"""Domain-aware modular chunking system."""

from .configuration import ChunkingConfig, ChunkingProfile, ChunkerSettings
from .factory import ChunkerFactory
from .models import Chunk, ChunkerConfig, Granularity
from .pipeline import MultiGranularityPipeline
from .ports import BaseChunker
from .registry import ChunkerRegistry, default_registry
from .service import ChunkingService, ChunkingOptions

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkerConfig",
    "ChunkerFactory",
    "ChunkerRegistry",
    "ChunkingConfig",
    "ChunkingOptions",
    "ChunkingProfile",
    "ChunkingService",
    "ChunkerSettings",
    "Granularity",
    "MultiGranularityPipeline",
    "default_registry",
]
