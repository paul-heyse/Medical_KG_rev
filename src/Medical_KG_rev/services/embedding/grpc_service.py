"""gRPC service for embedding operations."""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc
import structlog

# Proto imports - handle missing generated files gracefully
try:
    from ...proto.embedding_service_pb2 import (
        EmbedRequest,
        EmbedResponse,
        ListModelsRequest,
        ListModelsResponse,
    )
    from ...proto.embedding_service_pb2_grpc import EmbeddingServiceServicer
except ImportError:
    # Mock classes when proto files are not available
    class EmbedRequest:
        pass
    class EmbedResponse:
        pass
    class ListModelsRequest:
        pass
    class ListModelsResponse:
        pass
    EmbeddingServiceServicer = object

logger = structlog.get_logger(__name__)


class EmbeddingGRPCService(EmbeddingServiceServicer):
    """gRPC service implementation for embedding operations."""

    def __init__(self) -> None:
        """Initialize the embedding gRPC service."""
        self.logger = logger
        self._models: dict[str, Any] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load embedding models."""
        try:
            from sentence_transformers import SentenceTransformer

            # Load Qwen3 embedding model
            self._models["qwen3"] = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
            logger.info("Loaded Qwen3 embedding model")

        except Exception as e:
            logger.warning("Failed to load embedding models: %s", e)

    def Embed(
        self, request: EmbedRequest, context: grpc.ServicerContext
    ) -> EmbedResponse:
        """Generate embeddings for input texts."""
        try:
            if not request.inputs:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No input texts provided")
                return EmbedResponse()

            # Get model
            model_name = request.model or "qwen3"
            if model_name not in self._models:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Model not found: {model_name}")
                return EmbedResponse()

            # Generate embeddings
            start_time = time.time()
            model = self._models[model_name]
            embeddings = model.encode(request.inputs)
            duration_ms = (time.time() - start_time) * 1000

            # Create response
            response = EmbedResponse()
            response.embeddings.extend(embeddings.tolist())
            response.model = model_name
            response.dimensions = len(embeddings[0]) if len(embeddings) > 0 else 0
            response.duration_ms = duration_ms

            return response

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Embedding generation failed: {e}")
            return EmbedResponse()

    def ListModels(
        self, request: ListModelsRequest, context: grpc.ServicerContext
    ) -> ListModelsResponse:
        """List available embedding models."""
        try:
            response = ListModelsResponse()
            response.models.extend(list(self._models.keys()))
            return response

        except Exception as e:
            self.logger.error(f"List models failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"List models failed: {e}")
            return ListModelsResponse()

    def health_check(self) -> dict[str, Any]:
        """Check service health."""
        return {
            "service": "embedding_grpc",
            "status": "healthy",
            "models_loaded": len(self._models),
            "available_models": list(self._models.keys()),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        return {
            "models_loaded": len(self._models),
            "available_models": list(self._models.keys()),
        }


class EmbeddingGRPCServiceFactory:
    """Factory for creating embedding gRPC services."""

    @staticmethod
    def create() -> EmbeddingGRPCService:
        """Create an embedding gRPC service instance."""
        return EmbeddingGRPCService()

    @staticmethod
    def create_with_config(config: dict[str, Any]) -> EmbeddingGRPCService:
        """Create an embedding gRPC service with configuration."""
        service = EmbeddingGRPCService()
        # Apply configuration if needed
        return service


# Global embedding gRPC service instance
_embedding_grpc_service: EmbeddingGRPCService | None = None


def get_embedding_grpc_service() -> EmbeddingGRPCService:
    """Get the global embedding gRPC service instance."""
    global _embedding_grpc_service

    if _embedding_grpc_service is None:
        _embedding_grpc_service = EmbeddingGRPCServiceFactory.create()

    return _embedding_grpc_service


def create_embedding_grpc_service() -> EmbeddingGRPCService:
    """Create a new embedding gRPC service instance."""
    return EmbeddingGRPCServiceFactory.create()
