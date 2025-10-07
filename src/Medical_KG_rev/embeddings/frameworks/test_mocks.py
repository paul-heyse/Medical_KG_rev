"""Mock classes for testing embedding framework delegates."""

from __future__ import annotations


class BatchOnly:
    """Mock embedding class that only supports batch embedding."""

    def embed_documents(self, texts):  # pragma: no cover - invoked via delegate helper
        return [[float(len(text))] * 3 for text in texts]

    def embed(self, texts):  # pragma: no cover - invoked via delegate helper
        return [[float(len(text))] * 3 for text in texts]


class QueryOnly:
    """Mock embedding class that only supports query embedding."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, text):  # pragma: no cover - invoked via delegate helper
        self.calls.append(text)
        length = float(len(text))
        return [length, length + 1.0]

    def embed_queries(self, texts):  # pragma: no cover - invoked via delegate helper
        self.calls.extend(texts)
        return [[float(len(text)), float(len(text)) + 1.0] for text in texts]


class LlamaStyle:
    """Mock embedding class with LlamaIndex-style interface."""

    def get_text_embedding(self, text):  # pragma: no cover - invoked via delegate helper
        base = float(len(text))
        return [base, base / 2.0, base / 4.0]

    def embed_documents(self, texts):  # pragma: no cover - invoked via delegate helper
        return [[float(len(text)), float(len(text)) / 2.0, float(len(text)) / 4.0] for text in texts]

    def embed(self, texts):  # pragma: no cover - invoked via delegate helper
        return [[float(len(text)), float(len(text)) / 2.0, float(len(text)) / 4.0] for text in texts]
