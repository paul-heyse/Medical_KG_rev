"""Embedding stage with typed contracts."""

from __future__ import annotations

from typing import Protocol

import structlog

from Medical_KG_rev.orchestration.stages.contracts import PipelineState, StageContext
from Medical_KG_rev.orchestration.stages.embedding.contracts import (
    EmbeddingProcessingError,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingValidationError,
)

logger = structlog.get_logger(__name__)


class EmbeddingWorkerProtocol(Protocol):
    """Protocol for embedding workers."""

    def run(self, request: EmbeddingRequest) -> EmbeddingResult: ...


class EmbeddingStageV2:
    """Embedding stage with typed contracts."""

    def __init__(
        self,
        worker: EmbeddingWorkerProtocol,
        namespace: str,
        model_id: str,
    ):
        self.worker = worker
        self.namespace = namespace
        self.model_id = model_id

    def execute(self, context: StageContext, state: PipelineState) -> EmbeddingResult:
        """Execute embedding with typed contracts.

        Args:
            context: Pipeline context containing chunks
            state: Pipeline state

        Returns:
            EmbeddingResult with vectors and metadata

        Raises:
            EmbeddingValidationError: Invalid input
            EmbeddingProcessingError: Processing failed
        """
        # Validate input
        if not state.chunks:
            raise EmbeddingValidationError(
                "No chunks provided",
                field="chunks",
                correlation_id=context.correlation_id
            )

        try:
            # Extract chunks
            chunks = state.chunks
            texts = tuple(chunk.content for chunk in chunks)

            # Create typed request
            request = EmbeddingRequest(
                texts=texts,
                namespace=self.namespace,
                model_id=self.model_id,
                correlation_id=context.correlation_id,
                metadata=context.metadata
            )

            logger.info(
                "embedding.stage.execute",
                text_count=len(texts),
                namespace=self.namespace,
                model_id=self.model_id,
                correlation_id=context.correlation_id
            )

            # Call worker with typed request
            result = self.worker.run(request)

            logger.info(
                "embedding.stage.complete",
                vector_count=result.vector_count,
                processing_time_ms=result.processing_time_ms,
                correlation_id=context.correlation_id
            )

            return result

        except ValueError as e:
            raise EmbeddingValidationError(
                str(e),
                correlation_id=context.correlation_id
            ) from e
        except Exception as e:
            # Check for GPU errors
            if "GPU" in str(e) or "gpu" in str(e).lower():
                raise EmbeddingProcessingError(
                    str(e),
                    error_type="gpu_unavailable",
                    correlation_id=context.correlation_id
                ) from e
            raise EmbeddingProcessingError(
                str(e),
                correlation_id=context.correlation_id
            ) from e
