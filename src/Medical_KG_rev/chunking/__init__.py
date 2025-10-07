"""Domain-aware modular chunking system."""

from .configuration import ChunkingConfig, ChunkingProfile, ChunkerSettings
from .factory import ChunkerFactory
from .models import Chunk, ChunkerConfig, Granularity
from .pipeline import MultiGranularityPipeline
from .runtime import ChunkerSession, ChunkingRuntime
from .ports import BaseChunker
from .registry import ChunkerRegistry, default_registry
from .segmentation import (
    LayoutSegmenter,
    Segment,
    SegmentAccumulator,
    Segmenter,
    SectionSegmenter,
    SlidingWindowSegmenter,
)
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
    "ChunkingRuntime",
    "ChunkerSettings",
    "ChunkerSession",
    "Granularity",
    "LayoutSegmenter",
    "MultiGranularityPipeline",
    "Segment",
    "SegmentAccumulator",
    "Segmenter",
    "SectionSegmenter",
    "SlidingWindowSegmenter",
    "default_registry",
]
