"""Semantic chunking strategies using graph-based and embedding approaches."""

from __future__ import annotations

from collections.abc import Iterable

from Medical_KG_rev.models.ir import Document

from ..assembly import ChunkAssembler
from ..exceptions import ChunkerConfigurationError
from ..models import Chunk, Granularity
from ..ports import BaseChunker
from ..provenance import BlockContext
from ..segmentation import Segment
from ..tokenization import TokenCounter, default_token_counter


class SemanticChunker(BaseChunker):
    """Base class for semantic chunking strategies."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize the semantic chunker."""
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.counter = token_counter or default_token_counter()

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        """Chunk a document using semantic strategies."""
        # This is a base implementation - subclasses should override
        raise NotImplementedError("Subclasses must implement chunk method")

    def explain(self) -> dict[str, object]:
        """Explain the chunking strategy."""
        return {
            "strategy": "semantic",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


class GraphPartitionChunker(SemanticChunker):
    """Chunker that uses graph partitioning for semantic chunking."""

    def __init__(
        self,
        *,
        algorithm: str = "louvain",
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize the graph partition chunker."""
        super().__init__(token_counter=token_counter)
        try:
            import networkx as nx
        except ImportError as exc:
            raise ChunkerConfigurationError(
                "networkx must be installed for GraphPartitionChunker"
            ) from exc

        self.algorithm = algorithm
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.nx = nx

    def chunk(
        self,
        document: Document,
        *,
        tenant_id: str,
        granularity: Granularity | None = None,
        blocks: Iterable | None = None,
    ) -> list[Chunk]:
        """Chunk a document using graph partitioning."""
        # Extract text segments
        segments = []
        for block in document.content:
            if hasattr(block, 'text') and block.text:
                segments.append(Segment(
                    text=block.text,
                    start=0,
                    end=len(block.text),
                    metadata=getattr(block, 'metadata', {})
                ))

        if not segments:
            return []

        # Create similarity graph
        graph = self.nx.Graph()
        for i, segment in enumerate(segments):
            graph.add_node(i, text=segment.text, metadata=segment.metadata)

        # Add edges based on similarity (simplified implementation)
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                # Simple similarity based on shared words
                words_i = set(segment.text.lower().split())
                words_j = set(segments[j].text.lower().split())
                similarity = len(words_i & words_j) / max(len(words_i | words_j), 1)

                if similarity > 0.1:  # Threshold for similarity
                    graph.add_edge(i, j, weight=similarity)

        # Partition the graph
        if self.algorithm == "louvain" and hasattr(self.nx.algorithms.community, "louvain_communities"):
            communities = list(self.nx.algorithms.community.louvain_communities(graph))
        else:
            # Fallback to connected components
            communities = list(self.nx.connected_components(graph))

        # Create chunks from communities
        assembler = ChunkAssembler(
            document,
            tenant_id=tenant_id,
            chunker_name="graph_partition",
            chunker_version="v1",
            granularity=granularity or "semantic",
            token_counter=self.counter,
        )

        result_chunks = []
        for community_id, community in enumerate(communities):
            community_segments = [segments[i] for i in community]
            chunk_text = " ".join(seg.text for seg in community_segments)

            # Check chunk size constraints
            if len(chunk_text) < self.min_chunk_size or len(chunk_text) > self.max_chunk_size:
                continue

            chunk_meta = {
                "segment_type": "graph_partition",
                "algorithm": self.algorithm,
                "community_id": community_id,
                "segment_count": len(community_segments),
                "token_count": self.counter.count_tokens(chunk_text),
            }

            # Create context for the chunk
            context = BlockContext(
                text=chunk_text,
                block_id=f"graph-chunk-{community_id}",
                block_type="graph_partition",
                metadata=chunk_meta,
            )

            result_chunks.append(assembler.build([context], metadata=chunk_meta))

        return result_chunks

    def explain(self) -> dict[str, object]:
        """Explain the chunking strategy."""
        return {
            "strategy": "graph_partition",
            "algorithm": self.algorithm,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
        }
