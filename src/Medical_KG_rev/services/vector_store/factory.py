"""Factory helpers to construct vector store adapters from configuration."""

from __future__ import annotations

from collections.abc import Mapping

from .stores.external import HttpClient
from .stores.faiss import FaissVectorStore
from .stores.memory import InMemoryVectorStore
from .stores.milvus import MilvusVectorStore
from .stores.opensearch import OpenSearchKNNStore
from .stores.qdrant import QdrantVectorStore
from .types import VectorStorePort


_SUPPORTED_DRIVERS = {
    "memory": InMemoryVectorStore,
    "faiss": FaissVectorStore,
    "qdrant": QdrantVectorStore,
    "milvus": MilvusVectorStore,
    "opensearch": OpenSearchKNNStore,
    "weaviate": WeaviateStore,
    "vespa": VespaStore,
    "pgvector": PgvectorStore,
    "diskann": DiskANNStore,
    "hnswlib": HNSWLibIndex,
    "nmslib": NMSLibIndex,
    "annoy": AnnoyIndex,
    "scann": ScaNNIndex,
    "lancedb": LanceDBStore,
    "duckdbvss": DuckDBVSSStore,
    "chroma": ChromaStore,
}


class VectorStoreFactory:
    """Factory that instantiates vector store adapters."""

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        self.config = config or {}

    def build(self) -> VectorStorePort:
        driver = str(self.config.get("driver", "memory")).lower()
        if driver not in _SUPPORTED_DRIVERS:
            raise ValueError(f"Unsupported vector store driver '{driver}'")
        cls = _SUPPORTED_DRIVERS[driver]
        return cls()
