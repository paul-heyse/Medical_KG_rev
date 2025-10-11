"""gRPC service implementation for embedding operations."""

from __future__ import annotations

import logging

import grpc

# Import generated protobuf classes
try:
    from proto import embedding_pb2, embedding_pb2_grpc
except ImportError:
    # Fallback for when protobuf files aren't generated yet
    embedding_pb2 = None
    embedding_pb2_grpc = None

logger = logging.getLogger(__name__)


class EmbeddingServiceServicer(embedding_pb2_grpc.EmbeddingServiceServicer):
    """gRPC servicer for embedding operations."""

    def __init__(self):
        self._models = {}
        self._load_default_models()

    def _load_default_models(self):
        """Load default embedding models."""
        try:
            from sentence_transformers import SentenceTransformer

            # Load Qwen3 embedding model
            self._models["qwen3"] = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
            logger.info("Loaded Qwen3 embedding model")

        except Exception as e:
            logger.warning("Failed to load embedding models: %s", e)

    def Embed(
        self, request: embedding_pb2.EmbedRequest, context: grpc.ServicerContext
    ) -> embedding_pb2.EmbedResponse:
        """Generate embeddings for input texts."""
        try:
            if not request.inputs:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No input texts provided")
                return embedding_pb2.EmbedResponse()

            # Get model (default to qwen3)
            model_name = "qwen3"
            if model_name not in self._models:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(f"Model {model_name} not available")
                return embedding_pb2.EmbedResponse()

            model = self._models[model_name]

            # Generate embeddings
            embeddings = model.encode(request.inputs, convert_to_numpy=True)

            # Convert to protobuf format
            embedding_vectors = []
            for i, embedding in enumerate(embeddings):
                vector = embedding_pb2.EmbeddingVector(
                    id=f"embedding_{i}",
                    model=model_name,
                    namespace=request.namespace or "default",
                    kind="dense",
                    dimension=len(embedding),
                    values=embedding.tolist(),
                    metadata={
                        "input_text": request.inputs[i][:100],  # Truncate for metadata
                        "tenant_id": request.tenant_id,
                    },
                )
                embedding_vectors.append(vector)

            # Create metadata
            metadata = embedding_pb2.EmbeddingMetadata(
                provider="sentence-transformers",
                dimension=len(embeddings[0]) if len(embeddings) > 0 else 0,
                duration_ms=0.0,  # TODO: Measure actual duration
                model=model_name,
            )

            return embedding_pb2.EmbedResponse(
                namespace=request.namespace or "default",
                embeddings=embedding_vectors,
                metadata=metadata,
            )

        except Exception as e:
            logger.error("Error generating embeddings: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return embedding_pb2.EmbedResponse()

    def ListNamespaces(
        self, request: embedding_pb2.ListNamespacesRequest, context: grpc.ServicerContext
    ) -> embedding_pb2.ListNamespacesResponse:
        """List available embedding namespaces."""
        try:
            # Return default namespaces
            namespaces = [
                embedding_pb2.NamespaceInfo(
                    id="default",
                    provider="sentence-transformers",
                    kind="dense",
                    dimension=4096,  # Qwen3 embedding dimension
                    max_tokens=8192,
                    enabled=True,
                    allowed_tenants=[request.tenant_id] if request.tenant_id else [],
                    allowed_scopes=["embedding:read"],
                )
            ]

            return embedding_pb2.ListNamespacesResponse(namespaces=namespaces)

        except Exception as e:
            logger.error("Error listing namespaces: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return embedding_pb2.ListNamespacesResponse()

    def ValidateTexts(
        self, request: embedding_pb2.ValidateTextsRequest, context: grpc.ServicerContext
    ) -> embedding_pb2.ValidateTextsResponse:
        """Validate texts for embedding generation."""
        try:
            results = []
            valid = True

            for i, text in enumerate(request.texts):
                # Simple validation - check length and basic content
                token_count = len(text.split())  # Rough token count
                exceeds_budget = token_count > 8192  # Max tokens for Qwen3

                if exceeds_budget:
                    valid = False

                result = embedding_pb2.TextValidationResult(
                    text_index=i,
                    token_count=token_count,
                    exceeds_budget=exceeds_budget,
                    warning="Text exceeds token budget" if exceeds_budget else "",
                )
                results.append(result)

            return embedding_pb2.ValidateTextsResponse(
                namespace=request.namespace or "default",
                valid=valid,
                results=results,
            )

        except Exception as e:
            logger.error("Error validating texts: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return embedding_pb2.ValidateTextsResponse()
