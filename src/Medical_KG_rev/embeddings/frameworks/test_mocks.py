"""Mock classes for testing embedding framework delegates.

This module provides mock classes for testing embedding framework
delegates, simulating different embedding interface patterns and
behaviors commonly found in various embedding libraries.

The module defines:
- BatchOnly: Mock class supporting only batch embedding operations
- QueryOnly: Mock class supporting only query embedding operations
- LlamaStyle: Mock class with LlamaIndex-style interface

Architecture:
- Simple mock implementations for testing purposes
- Deterministic embedding generation based on text length
- Support for different interface patterns
- Call tracking for verification

Thread Safety:
- Mock classes are thread-safe for testing purposes.

Performance:
- Lightweight mock implementations with minimal overhead.
- Deterministic behavior for consistent test results.

Examples:
    # Test batch-only embedding
    batch_embedder = BatchOnly()
    embeddings = batch_embedder.embed_documents(["text1", "text2"])

    # Test query-only embedding with call tracking
    query_embedder = QueryOnly()
    embedding = query_embedder.embed_query("test query")
    assert "test query" in query_embedder.calls

"""

# IMPORTS
from __future__ import annotations


# MOCK CLASSES
class BatchOnly:
    """Mock embedding class that only supports batch embedding.

    This mock class simulates an embedding model that only supports
    batch embedding operations, returning deterministic embeddings
    based on text length.

    Thread Safety:
        Thread-safe for testing purposes.

    Examples:
        embedder = BatchOnly()
        embeddings = embedder.embed_documents(["text1", "text2"])
        # Returns: [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]

    """

    def embed_documents(self, texts):  # pragma: no cover - invoked via delegate helper
        """Embed multiple documents in batch.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        """
        return [[float(len(text))] * 3 for text in texts]

    def embed(self, texts):  # pragma: no cover - invoked via delegate helper
        """Embed multiple texts (alias for embed_documents).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        """
        return [[float(len(text))] * 3 for text in texts]


class QueryOnly:
    """Mock embedding class that only supports query embedding.

    This mock class simulates an embedding model that only supports
    query embedding operations, with call tracking for verification.

    Attributes:
        calls: List of texts that have been embedded

    Thread Safety:
        Thread-safe for testing purposes.

    Examples:
        embedder = QueryOnly()
        embedding = embedder.embed_query("test")
        assert "test" in embedder.calls

    """

    def __init__(self) -> None:
        """Initialize the query-only embedder.

        Raises:
            None: Initialization always succeeds.

        """
        self.calls: list[str] = []

    def embed_query(self, text):  # pragma: no cover - invoked via delegate helper
        """Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector

        """
        self.calls.append(text)
        length = float(len(text))
        return [length, length + 1.0]

    def embed_queries(self, texts):  # pragma: no cover - invoked via delegate helper
        """Embed multiple query texts.

        Args:
            texts: List of query texts to embed

        Returns:
            List of embedding vectors

        """
        self.calls.extend(texts)
        return [[float(len(text)), float(len(text)) + 1.0] for text in texts]


class LlamaStyle:
    """Mock embedding class with LlamaIndex-style interface.

    This mock class simulates an embedding model with LlamaIndex-style
    interface, supporting both single and batch operations.

    Thread Safety:
        Thread-safe for testing purposes.

    Examples:
        embedder = LlamaStyle()
        embedding = embedder.get_text_embedding("test")
        # Returns: [4.0, 2.0, 1.0]

    """

    def get_text_embedding(self, text):  # pragma: no cover - invoked via delegate helper
        """Get embedding for a single text (LlamaIndex style).

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        """
        base = float(len(text))
        return [base, base / 2.0, base / 4.0]

    def embed_documents(self, texts):  # pragma: no cover - invoked via delegate helper
        """Embed multiple documents in batch.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        """
        return [[float(len(text)), float(len(text)) / 2.0, float(len(text)) / 4.0] for text in texts]

    def embed(self, texts):  # pragma: no cover - invoked via delegate helper
        """Embed multiple texts (alias for embed_documents).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        """
        return [[float(len(text)), float(len(text)) / 2.0, float(len(text)) / 4.0] for text in texts]
