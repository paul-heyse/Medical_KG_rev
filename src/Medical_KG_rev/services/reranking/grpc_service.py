"""gRPC service implementation for reranking operations."""

from __future__ import annotations

import logging
import time

import grpc
from google.protobuf import empty_pb2

# Import generated protobuf classes
try:
    from proto import reranking_pb2, reranking_pb2_grpc
except ImportError:
    # Fallback for when protobuf files aren't generated yet
    reranking_pb2 = None
    reranking_pb2_grpc = None

logger = logging.getLogger(__name__)


class RerankingServiceServicer(reranking_pb2_grpc.RerankingServiceServicer):
    """gRPC servicer for reranking operations."""

    def __init__(self):
        self._models = {}
        self._load_default_models()

    def _load_default_models(self):
        """Load default reranking models."""
        try:
            from sentence_transformers import CrossEncoder

            # Load cross-encoder model for reranking
            self._models["ms-marco-MiniLM-L-6-v2"] = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            logger.info("Loaded cross-encoder reranking model")

        except Exception as e:
            logger.warning("Failed to load reranking models: %s", e)

    def RerankBatch(
        self, request: reranking_pb2.RerankBatchRequest, context: grpc.ServicerContext
    ) -> reranking_pb2.RerankBatchResponse:
        """Rerank a batch of documents."""
        try:
            if not request.documents:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No documents provided")
                return reranking_pb2.RerankBatchResponse()

            # Get model (default to ms-marco)
            model_name = request.model or "ms-marco-MiniLM-L-6-v2"
            if model_name not in self._models:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(f"Model {model_name} not available")
                return reranking_pb2.RerankBatchResponse()

            model = self._models[model_name]

            # Prepare query-document pairs
            pairs = [(request.query, doc) for doc in request.documents]

            # Generate scores
            start_time = time.time()
            scores = model.predict(pairs)
            duration_ms = (time.time() - start_time) * 1000

            # Create results with scores
            results = []
            for i, (doc, score) in enumerate(zip(request.documents, scores, strict=False)):
                result = reranking_pb2.RerankedDocument(
                    index=i,
                    text=doc,
                    score=float(score),
                    metadata={
                        "model": model_name,
                        "tenant_id": request.tenant_id,
                    },
                )
                results.append(result)

            # Sort by score (descending)
            results.sort(key=lambda x: x.score, reverse=True)

            # Apply top_k limit
            if request.top_k > 0:
                results = results[: request.top_k]

            return reranking_pb2.RerankBatchResponse(
                results=results,
                model=model_name,
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error("Error reranking documents: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return reranking_pb2.RerankBatchResponse()

    def ListModels(
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> reranking_pb2.ListModelsResponse:
        """List available reranking models."""
        try:
            models = []
            for model_name, model in self._models.items():
                model_info = reranking_pb2.ModelInfo(
                    name=model_name,
                    description="Cross-encoder model for reranking",
                    max_length=512,  # Typical max length for cross-encoders
                    available=True,
                )
                models.append(model_info)

            return reranking_pb2.ListModelsResponse(models=models)

        except Exception as e:
            logger.error("Error listing models: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return reranking_pb2.ListModelsResponse()
