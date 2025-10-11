"""Domain-aware modular chunking system."""

from .configuration import ChunkerSettings, ChunkingConfig, ChunkingProfile
from .factory import ChunkerFactory
from .models import Chunk, ChunkerConfig, Granularity
from .pipeline import MultiGranularityPipeline
from .ports import BaseChunker
from .registry import ChunkerRegistry, default_registry
from .runtime import ChunkerSession, ChunkingRuntime
from .segmentation import SectionSegmenter
from .service import ChunkingOptions, ChunkingService

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkerConfig",
    "ChunkerFactory",
    "ChunkerRegistry",
    "ChunkerSession",
    "ChunkerSettings",
    "ChunkingConfig",
    "ChunkingOptions",
    "ChunkingProfile",
    "ChunkingRuntime",
    "ChunkingService",
    "Granularity",
    "LayoutSegmenter",
    "MultiGranularityPipeline",
    "SectionSegmenter",
    "Segment",
    "SegmentAccumulator",
    "Segmenter",
    "SlidingWindowSegmenter",
    "default_registry",
]
