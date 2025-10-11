"""Qwen3 embedding service for dense retrieval.

This module implements Qwen3 4096-dimension embedding generation for
semantic search with contextualized text processing.
"""

import logging
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from Medical_KG_rev.config.settings import get_settings
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
QWEN3_PROCESSING_SECONDS = Histogram(
    "qwen3_processing_seconds",
    "Time spent on Qwen3 operations",
    ["operation", "status"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

QWEN3_OPERATIONS_TOTAL = Counter(
    "qwen3_operations_total",
    "Total number of Qwen3 operations",
    ["operation", "status"]
)

QWEN3_EMBEDDING_DIMENSIONS = Histogram(
    "qwen3_embedding_dimensions",
    "Qwen3 embedding dimensions",
    ["operation"],
    buckets=[512, 1024, 2048, 4096, 8192]
)

QWEN3_BATCH_SIZE = Histogram(
    "qwen3_batch_size",
    "Qwen3 batch size",
    ["operation"],
    buckets=[1, 2, 4, 8, 16, 32, 64]
)


class Qwen3Embedding(BaseModel):
    """Qwen3 embedding representation."""

    chunk_id: str = Field(..., description="Chunk identifier")
    embedding: list[float] = Field(..., description="4096-dimension embedding vector")
    model_name: str = Field(..., description="Qwen3 model name")
    preprocessing_version: str = Field(..., description="Preprocessing version")
    contextualized_text: str = Field(..., description="Contextualized text used for embedding")


class Qwen3Result(BaseModel):
    """Qwen3 embedding result."""

    chunk_id: str = Field(..., description="Chunk identifier")
    embedding: list[float] = Field(..., description="4096-dimension embedding vector")
    model_name: str = Field(..., description="Qwen3 model name")
    preprocessing_version: str = Field(..., description="Preprocessing version")
    processing_time_seconds: float = Field(..., description="Processing time")
    gpu_memory_used_mb: int = Field(..., description="GPU memory usage")


class Qwen3ProcessingError(Exception):
    """Exception raised during Qwen3 processing."""
    pass


class Qwen3ModelLoadError(Exception):
    """Exception raised when Qwen3 model fails to load."""
    pass


class Qwen3Service:
    """Qwen3 embedding service for dense retrieval.

    This service implements Qwen3 4096-dimension embedding generation
    for semantic search with contextualized text processing.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embedding_dimension: int = 4096,
        batch_size: int = 8,
        max_seq_length: int = 2048,
        gpu_manager: Any = None,
    ):
        """Initialize Qwen3 service.

        Args:
            model_name: Name of the Qwen3 model to use
            embedding_dimension: Dimension of embedding vectors
            batch_size: Batch size for processing
            max_seq_length: Maximum sequence length
            gpu_manager: GPU manager for resource allocation
        """
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.gpu_manager = gpu_manager

        # Model components (loaded lazily)
        self._tokenizer = None
        self._model = None
        self._is_loaded = False

        # Settings
        self._settings = get_settings()

        logger.info(
            "Initialized Qwen3 service",
            extra={
                "model_name": self.model_name,
                "embedding_dimension": self.embedding_dimension,
                "batch_size": self.batch_size,
                "max_seq_length": self.max_seq_length,
            }
        )

    def _load_model(self) -> None:
        """Load Qwen3 model and tokenizer."""
        if self._is_loaded:
            return

        start_time = time.perf_counter()

        try:
            # Check GPU availability if gpu_manager is provided
            if self.gpu_manager and hasattr(self.gpu_manager, 'is_gpu_available'):
                if not self.gpu_manager.is_gpu_available():
                    raise Qwen3ModelLoadError("GPU not available for Qwen3 model")

            # Import transformers components
            try:
# import torch  # type: ignore  # Removed for torch isolation
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as e:
                raise Qwen3ModelLoadError(f"Transformers not available: {e}")

            # Load tokenizer
            logger.info("Loading Qwen3 tokenizer", extra={"model_name": self.model_name})
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model
            logger.info("Loading Qwen3 model", extra={"model_name": self.model_name})
            self._model = AutoModel.from_pretrained(self.model_name)

            # Move to GPU if available
            # GPU functionality moved to gRPC services
            logger.info("Qwen3 model loaded (GPU operations handled by gRPC services)")

            # Set model to evaluation mode
            self._model.eval()

            self._is_loaded = True

            load_time = time.perf_counter() - start_time
            QWEN3_PROCESSING_SECONDS.labels(operation="load_model", status="ok").observe(load_time)
            QWEN3_OPERATIONS_TOTAL.labels(operation="load_model", status="ok").inc()

            logger.info(
                "Qwen3 model loaded successfully",
                extra={
                    "model_name": self.model_name,
                    "load_time_seconds": load_time,
                }
            )

        except Exception as e:
            load_time = time.perf_counter() - start_time
            QWEN3_PROCESSING_SECONDS.labels(operation="load_model", status="error").observe(load_time)
            QWEN3_OPERATIONS_TOTAL.labels(operation="load_model", status="error").inc()

            logger.error(
                "Failed to load Qwen3 model",
                extra={
                    "model_name": self.model_name,
                    "error": str(e),
                    "load_time_seconds": load_time,
                }
            )
            raise Qwen3ModelLoadError(f"Failed to load Qwen3 model: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for Qwen3 embedding.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Basic text preprocessing
        text = text.strip()

        # Truncate if too long
        if len(text) > self.max_seq_length * 4:  # Rough character to token ratio
            text = text[:self.max_seq_length * 4]

        return text

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if not self._is_loaded:
            self._load_model()

        try:
# import torch  # type: ignore  # Removed for torch isolation

            # Preprocess text
            processed_text = self._preprocess_text(text)

            # Tokenize
            inputs = self._tokenizer(
                processed_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            )

            # Move to GPU if available
            # GPU functionality moved to gRPC services
            logger.info("Qwen3 model loaded (GPU operations handled by gRPC services)")

            # Generate embedding
            # Torch functionality moved to gRPC services
            raise NotImplementedError("Torch functionality moved to gRPC services")

        except Exception as e:
            logger.error(
                "Failed to generate Qwen3 embedding",
                extra={
                    "text_length": len(text),
                    "error": str(e),
                }
            )
            raise Qwen3ProcessingError(f"Failed to generate embedding: {e}")

    def generate_embedding(self, chunk_id: str, contextualized_text: str) -> Qwen3Result:
        """Generate embedding for a single chunk.

        Args:
            chunk_id: Chunk identifier
            contextualized_text: Contextualized text for embedding

        Returns:
            Qwen3 embedding result
        """
        start_time = time.perf_counter()

        try:
            # Generate embedding
            embedding_vector = self._generate_embedding(contextualized_text)

            processing_time = time.perf_counter() - start_time

            # Get GPU memory usage if available
            gpu_memory_used_mb = 0
            if self.gpu_manager and hasattr(self.gpu_manager, 'get_gpu_memory_used'):
                gpu_memory_used_mb = self.gpu_manager.get_gpu_memory_used()

            result = Qwen3Result(
                chunk_id=chunk_id,
                embedding=embedding_vector,
                model_name=self.model_name,
                preprocessing_version="1.0.0",
                processing_time_seconds=processing_time,
                gpu_memory_used_mb=gpu_memory_used_mb,
            )

            QWEN3_PROCESSING_SECONDS.labels(operation="generate_embedding", status="ok").observe(processing_time)
            QWEN3_OPERATIONS_TOTAL.labels(operation="generate_embedding", status="ok").inc()
            QWEN3_EMBEDDING_DIMENSIONS.labels(operation="generate_embedding").observe(len(embedding_vector))

            logger.info(
                "Qwen3 embedding generated",
                extra={
                    "chunk_id": chunk_id,
                    "embedding_dimension": len(embedding_vector),
                    "processing_time_seconds": processing_time,
                    "gpu_memory_used_mb": gpu_memory_used_mb,
                }
            )

            return result

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            QWEN3_PROCESSING_SECONDS.labels(operation="generate_embedding", status="error").observe(processing_time)
            QWEN3_OPERATIONS_TOTAL.labels(operation="generate_embedding", status="error").inc()

            logger.error(
                "Failed to generate Qwen3 embedding",
                extra={
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                }
            )
            raise Qwen3ProcessingError(f"Failed to generate embedding: {e}")

    def generate_embeddings_batch(self, chunks: list[tuple[str, str]]) -> list[Qwen3Result]:
        """Generate embeddings for multiple chunks in batch.

        Args:
            chunks: List of (chunk_id, contextualized_text) tuples

        Returns:
            List of Qwen3 embedding results
        """
        start_time = time.perf_counter()

        try:
            if not self._is_loaded:
                self._load_model()

            results = []

            # Process in batches
            for i in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[i:i + self.batch_size]

                # Generate embeddings for batch
                batch_results = []
                for chunk_id, contextualized_text in batch_chunks:
                    result = self.generate_embedding(chunk_id, contextualized_text)
                    batch_results.append(result)

                results.extend(batch_results)

            processing_time = time.perf_counter() - start_time

            QWEN3_PROCESSING_SECONDS.labels(operation="generate_embeddings_batch", status="ok").observe(processing_time)
            QWEN3_OPERATIONS_TOTAL.labels(operation="generate_embeddings_batch", status="ok").inc()
            QWEN3_BATCH_SIZE.labels(operation="generate_embeddings_batch").observe(len(chunks))

            logger.info(
                "Qwen3 embeddings generated in batch",
                extra={
                    "chunk_count": len(chunks),
                    "processing_time_seconds": processing_time,
                }
            )

            return results

        except Exception as e:
            processing_time = time.perf_counter() - start_time

            QWEN3_PROCESSING_SECONDS.labels(operation="generate_embeddings_batch", status="error").observe(processing_time)
            QWEN3_OPERATIONS_TOTAL.labels(operation="generate_embeddings_batch", status="error").inc()

            logger.error(
                "Failed to generate Qwen3 embeddings in batch",
                extra={
                    "chunk_count": len(chunks),
                    "error": str(e),
                    "processing_time_seconds": processing_time,
                }
            )
            raise Qwen3ProcessingError(f"Failed to generate embeddings in batch: {e}")

    def health_check(self) -> dict[str, Any]:
        """Check Qwen3 service health.

        Returns:
            Health status information
        """
        try:
            # Check GPU availability
            gpu_available = False
            if self.gpu_manager and hasattr(self.gpu_manager, 'is_gpu_available'):
                gpu_available = self.gpu_manager.is_gpu_available()

            # Check model loading
            model_loaded = self._is_loaded

            # Try to load model if not loaded
            if not model_loaded:
                try:
                    self._load_model()
                    model_loaded = True
                except Exception as e:
                    logger.warning("Qwen3 model failed to load during health check", extra={"error": str(e)})

            health_status = {
                "status": "healthy" if gpu_available and model_loaded else "unhealthy",
                "gpu_available": gpu_available,
                "model_loaded": model_loaded,
                "model_name": self.model_name,
                "embedding_dimension": self.embedding_dimension,
                "batch_size": self.batch_size,
                "max_seq_length": self.max_seq_length,
            }

            return health_status

        except Exception as e:
            logger.error("Qwen3 health check failed", extra={"error": str(e)})
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.model_name,
            }

    def get_service_stats(self) -> dict[str, Any]:
        """Get Qwen3 service statistics.

        Returns:
            Dictionary with service statistics
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "model_loaded": self._is_loaded,
            "gpu_available": (
                self.gpu_manager.is_gpu_available()
                if self.gpu_manager and hasattr(self.gpu_manager, 'is_gpu_available')
                else False
            ),
        }
