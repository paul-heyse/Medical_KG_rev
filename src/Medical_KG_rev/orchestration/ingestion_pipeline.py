"""Ingestion pipeline for document processing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from Medical_KG_rev.orchestration.stages.types import PipelineContext
from Medical_KG_rev.services.embedding.service import EmbeddingResponse, EmbeddingVector


class EmbeddingStage:
    """Stage for embedding processing in the ingestion pipeline."""

    def __init__(self, worker=None, namespaces=None, models=None):
        """Initialize the embedding stage.

        Args:
            worker: Embedding worker
            namespaces: List of namespaces
            models: List of models
        """
        self.worker = worker
        self.namespaces = namespaces or []
        self.models = models or []

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute the embedding stage.

        Args:
            context: Pipeline context

        Returns:
            Updated pipeline context
        """
        if not context.data.get("chunks"):
            from Medical_KG_rev.orchestration.stages import StageFailure
            raise StageFailure("No chunks provided", error_type="validation")

        try:
            # Extract chunks from context
            chunks = context.data["chunks"]
            texts = [chunk["body"] for chunk in chunks]
            
            # Create embedding request
            request = type('Request', (), {
                'texts': texts,
                'namespaces': self.namespaces,
                'models': self.models
            })()

            # Call worker
            response = self.worker.run(request)
            
            # Update context with embeddings
            context.data["embeddings"] = response.vectors
            
            # Add metrics
            context.data["metrics"] = {
                "embedding": {
                    "vectors": len(response.vectors)
                }
            }
            
            # Add embedding summary
            per_namespace = {}
            for vector in response.vectors:
                ns = vector.metadata.get("namespace", "default")
                if ns not in per_namespace:
                    per_namespace[ns] = 0
                per_namespace[ns] += 1
            
            context.data["embedding_summary"] = {
                "vectors": len(response.vectors),
                "per_namespace": per_namespace
            }
            
            return context
            
        except Exception as e:
            from Medical_KG_rev.orchestration.stages import StageFailure
            if "GPU" in str(e) or "gpu" in str(e).lower():
                raise StageFailure(str(e), error_type="gpu_unavailable")
            else:
                raise StageFailure(str(e), error_type="validation")
