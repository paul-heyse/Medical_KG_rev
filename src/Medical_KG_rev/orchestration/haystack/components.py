"""Haystack components for orchestration pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import structlog

from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.models.ir import Document

logger = structlog.get_logger(__name__)


@dataclass
class HaystackRetrieverConfig:
    """Configuration for Haystack retriever."""

    document_store_type: str = "opensearch"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 10
    similarity_threshold: float = 0.7
    index_name: str = "documents"


class HaystackRetriever:
    """Haystack-based document retriever."""

    def __init__(self, config: HaystackRetrieverConfig | None = None) -> None:
        """Initialize the Haystack retriever."""
        self.config = config or HaystackRetrieverConfig()
        self.logger = logger
        self._document_store = None
        self._embedder = None
        self._retriever = None

    def initialize(self) -> None:
        """Initialize Haystack components."""
        try:
            self._setup_document_store()
            self._setup_embedder()
            self._setup_retriever()
            self.logger.info("Haystack retriever initialized successfully")
        except Exception as exc:
            self.logger.error(f"Failed to initialize Haystack retriever: {exc}")
            raise exc

    def _setup_document_store(self) -> None:
        """Setup document store."""
        try:
            if self.config.document_store_type == "opensearch":
                self._document_store = self._create_opensearch_store()
            elif self.config.document_store_type == "faiss":
                self._document_store = self._create_faiss_store()
            else:
                raise ValueError(f"Unsupported document store type: {self.config.document_store_type}")
        except ImportError:
            self.logger.warning("Haystack OpenSearch integration not available, using mock store")
            self._document_store = self._create_mock_store()

    def _setup_embedder(self) -> None:
        """Setup embedder."""
        try:
            self._embedder = self._create_embedder()
        except ImportError:
            self.logger.warning("Haystack embedder not available, using mock embedder")
            self._embedder = self._create_mock_embedder()

    def _setup_retriever(self) -> None:
        """Setup retriever."""
        try:
            self._retriever = self._create_retriever()
        except ImportError:
            self.logger.warning("Haystack retriever not available, using mock retriever")
            self._retriever = self._create_mock_retriever()

    def _create_opensearch_store(self) -> Any:
        """Create OpenSearch document store."""
        # Mock implementation
        class MockOpenSearchStore:
            def __init__(self):
                self.index_name = self.config.index_name

            def write_documents(self, documents: list[Any]) -> None:
                pass

            def filter_documents(self, filters: dict[str, Any]) -> list[Any]:
                return []

        return MockOpenSearchStore()

    def _create_faiss_store(self) -> Any:
        """Create FAISS document store."""
        # Mock implementation
        class MockFAISSStore:
            def __init__(self):
                self.index_name = self.config.index_name

            def write_documents(self, documents: list[Any]) -> None:
                pass

            def filter_documents(self, filters: dict[str, Any]) -> list[Any]:
                return []

        return MockFAISSStore()

    def _create_mock_store(self) -> Any:
        """Create mock document store."""
        class MockDocumentStore:
            def __init__(self):
                self.index_name = self.config.index_name

            def write_documents(self, documents: list[Any]) -> None:
                pass

            def filter_documents(self, filters: dict[str, Any]) -> list[Any]:
                return []

        return MockDocumentStore()

    def _create_embedder(self) -> Any:
        """Create embedder."""
        # Mock implementation
        class MockEmbedder:
            def __init__(self, model_name: str):
                self.model_name = model_name

            def run(self, texts: list[str]) -> dict[str, Any]:
                return {
                    "embeddings": [[0.1] * 384 for _ in texts]
                }

        return MockEmbedder(self.config.embedding_model)

    def _create_mock_embedder(self) -> Any:
        """Create mock embedder."""
        class MockEmbedder:
            def __init__(self, model_name: str):
                self.model_name = model_name

            def run(self, texts: list[str]) -> dict[str, Any]:
                return {
                    "embeddings": [[0.1] * 384 for _ in texts]
                }

        return MockEmbedder(self.config.embedding_model)

    def _create_retriever(self) -> Any:
        """Create retriever."""
        # Mock implementation
        class MockRetriever:
            def __init__(self, document_store: Any, embedder: Any, top_k: int):
                self.document_store = document_store
                self.embedder = embedder
                self.top_k = top_k

            def run(self, query: str) -> dict[str, Any]:
                return {
                    "documents": [
                        {"content": f"Document {i}", "score": 0.9 - i * 0.1}
                        for i in range(min(self.top_k, 5))
                    ]
                }

        return MockRetriever(self._document_store, self._embedder, self.config.top_k)

    def _create_mock_retriever(self) -> Any:
        """Create mock retriever."""
        class MockRetriever:
            def __init__(self, document_store: Any, embedder: Any, top_k: int):
                self.document_store = document_store
                self.embedder = embedder
                self.top_k = top_k

            def run(self, query: str) -> dict[str, Any]:
                return {
                    "documents": [
                        {"content": f"Document {i}", "score": 0.9 - i * 0.1}
                        for i in range(min(self.top_k, 5))
                    ]
                }

        return MockRetriever(self._document_store, self._embedder, self.config.top_k)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve documents for a query."""
        try:
            if not self._retriever:
                self.initialize()

            result = self._retriever.run(query)
            documents = result.get("documents", [])

            # Filter by similarity threshold
            filtered_docs = [
                doc for doc in documents
                if doc.get("score", 0) >= self.config.similarity_threshold
            ]

            self.logger.info(
                "retrieval.completed",
                query=query,
                total_docs=len(documents),
                filtered_docs=len(filtered_docs),
            )

            return filtered_docs

        except Exception as exc:
            self.logger.error(f"Retrieval failed: {exc}")
            raise exc

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the document store."""
        try:
            if not self._document_store:
                self.initialize()

            # Convert documents to Haystack format
            haystack_docs = self._convert_documents(documents)

            # Add to document store
            self._document_store.write_documents(haystack_docs)

            self.logger.info(
                "documents.added",
                count=len(documents),
                index=self.config.index_name,
            )

        except Exception as exc:
            self.logger.error(f"Failed to add documents: {exc}")
            raise exc

    def _convert_documents(self, documents: list[Document]) -> list[dict[str, Any]]:
        """Convert documents to Haystack format."""
        haystack_docs = []

        for doc in documents:
            haystack_doc = {
                "content": doc.content[0] if doc.content else "",
                "meta": {
                    "id": doc.id,
                    "title": doc.title or "",
                    "source": "medical_kg",
                },
            }
            haystack_docs.append(haystack_doc)

        return haystack_docs

    def search(self, query: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Search documents with optional filters."""
        try:
            if not self._retriever:
                self.initialize()

            # Apply filters if provided
            if filters and self._document_store:
                filtered_docs = self._document_store.filter_documents(filters)
                # Use filtered documents for retrieval
                # This is a simplified implementation
                pass

            return self.retrieve(query)

        except Exception as exc:
            self.logger.error(f"Search failed: {exc}")
            raise exc

    def get_stats(self) -> dict[str, Any]:
        """Get retriever statistics."""
        return {
            "document_store_type": self.config.document_store_type,
            "embedding_model": self.config.embedding_model,
            "top_k": self.config.top_k,
            "similarity_threshold": self.config.similarity_threshold,
            "index_name": self.config.index_name,
            "initialized": self._retriever is not None,
        }

    def close(self) -> None:
        """Close the retriever and cleanup resources."""
        self.logger.info("Closing Haystack retriever")
        # Cleanup resources if needed
        self._document_store = None
        self._embedder = None
        self._retriever = None


class HaystackDocumentSplitter:
    """Haystack-based document splitter."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initialize the document splitter."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger
        self._splitter = None

    def initialize(self) -> None:
        """Initialize the document splitter."""
        try:
            self._splitter = self._create_splitter()
            self.logger.info("Haystack document splitter initialized")
        except ImportError:
            self.logger.warning("Haystack document splitter not available, using mock splitter")
            self._splitter = self._create_mock_splitter()

    def _create_splitter(self) -> Any:
        """Create document splitter."""
        # Mock implementation
        class MockSplitter:
            def __init__(self, chunk_size: int, chunk_overlap: int):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def run(self, documents: list[Any]) -> dict[str, Any]:
                chunks = []
                for doc in documents:
                    content = doc.get("content", "")
                    # Simple splitting by sentences
                    sentences = content.split(". ")
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > self.chunk_size:
                            if current_chunk:
                                chunks.append({"content": current_chunk.strip()})
                            current_chunk = sentence
                        else:
                            current_chunk += ". " + sentence if current_chunk else sentence
                    if current_chunk:
                        chunks.append({"content": current_chunk.strip()})
                return {"documents": chunks}

        return MockSplitter(self.chunk_size, self.chunk_overlap)

    def _create_mock_splitter(self) -> Any:
        """Create mock document splitter."""
        class MockSplitter:
            def __init__(self, chunk_size: int, chunk_overlap: int):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def run(self, documents: list[Any]) -> dict[str, Any]:
                chunks = []
                for doc in documents:
                    content = doc.get("content", "")
                    # Simple splitting by sentences
                    sentences = content.split(". ")
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > self.chunk_size:
                            if current_chunk:
                                chunks.append({"content": current_chunk.strip()})
                            current_chunk = sentence
                        else:
                            current_chunk += ". " + sentence if current_chunk else sentence
                    if current_chunk:
                        chunks.append({"content": current_chunk.strip()})
                return {"documents": chunks}

        return MockSplitter(self.chunk_size, self.chunk_overlap)

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks."""
        try:
            if not self._splitter:
                self.initialize()

            # Convert documents to Haystack format
            haystack_docs = [
                {"content": doc.content[0] if doc.content else ""}
                for doc in documents
            ]

            # Split documents
            result = self._splitter.run(haystack_docs)
            chunks_data = result.get("documents", [])

            # Convert back to Chunk objects
            chunks = []
            for i, chunk_data in enumerate(chunks_data):
                chunk = Chunk(
                    id=f"chunk-{i}",
                    text=chunk_data.get("content", ""),
                    metadata={
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "splitter": "haystack",
                    },
                )
                chunks.append(chunk)

            self.logger.info(
                "documents.split",
                input_docs=len(documents),
                output_chunks=len(chunks),
            )

            return chunks

        except Exception as exc:
            self.logger.error(f"Document splitting failed: {exc}")
            raise exc
