"""MinerU GPU microservice implementation."""

__all__ = [
    "MineruGrpcService",
    "MineruProcessor",
    "MineruRequest",
    "MineruResponse",
    "MineruOutOfMemoryError",
    "MineruGpuUnavailableError",
]


def __getattr__(name: str):  # pragma: no cover - simple lazy import helper
    if name in __all__:
        from .service import (
            MineruGrpcService,
            MineruGpuUnavailableError,
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
            "MineruGpuUnavailableError": MineruGpuUnavailableError,
        }[name]
    raise AttributeError(name)
