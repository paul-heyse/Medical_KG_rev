"""MinerU GPU microservice implementation."""

__all__ = [
    "MineruGrpcService",
    "MineruProcessor",
    "MineruRequest",
    "MineruResponse",
    "MineruOutOfMemoryError",
]


def __getattr__(name: str):  # pragma: no cover - simple lazy import helper
    if name in __all__:
        from .service import (
            MineruGrpcService,
            MineruOutOfMemoryError,
            MineruProcessor,
            MineruRequest,
            MineruResponse,
        )

        return {
            "MineruGrpcService": MineruGrpcService,
            "MineruProcessor": MineruProcessor,
            "MineruRequest": MineruRequest,
            "MineruResponse": MineruResponse,
            "MineruOutOfMemoryError": MineruOutOfMemoryError,
        }[name]
    raise AttributeError(name)
