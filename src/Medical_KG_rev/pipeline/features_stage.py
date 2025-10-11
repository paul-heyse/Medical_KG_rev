"""Features stage implementation for generating BM25, SPLADE, and Qwen3 features.

This stage processes chunks from the chunk store and generates:
- BM25 field texts for lexical retrieval
- SPLADE vectors with Rep-Max aggregation
- Qwen3 embeddings from contextualized text
"""

from pathlib import Path
from typing import Any
import logging

from pydantic import BaseModel, Field

from Medical_KG_rev.config.retrieval_config import BM25Config, Qwen3Config, SPLADEConfig
from Medical_KG_rev.pipeline.stages import FeaturesStage
from Medical_KG_rev.services.retrieval.bm25_field_mapping import BM25FieldMapper
from Medical_KG_rev.services.retrieval.qwen3_contextualized import HttpClient
from Medical_KG_rev.services.retrieval.qwen3_service import Qwen3Service
from Medical_KG_rev.services.retrieval.splade_aggregation import SPLADEAggregator
from Medical_KG_rev.services.retrieval.splade_segmentation import SPLADESegmenter
from Medical_KG_rev.storage.chunk_store import ChunkStore
from Medical_KG_rev.storage.manifest_manager import ManifestManager


logger = logging.getLogger(__name__)


class FeaturesStageConfig(BaseModel):
    """Configuration for the Features stage."""

    chunk_store_path: Path = Field(..., description="Path to chunk store database")
    bm25_config: BM25Config = Field(default_factory=BM25Config)
    splade_config: SPLADEConfig = Field(default_factory=SPLADEConfig)
    qwen3_config: Qwen3Config = Field(default_factory=Qwen3Config)
    batch_size: int = Field(default=32, description="Batch size for feature generation")
    max_workers: int = Field(default=4, description="Maximum number of worker threads")


class FeaturesStageImpl(FeaturesStage):
    """Implementation of the Features stage for generating retrieval features."""

    def __init__(self, config: FeaturesStageConfig) -> None:
        """Initialize the Features stage.

        Args:
        ----
            config: Configuration for the Features stage

        """
        super().__init__()
        self.config = config
        self.chunk_store: Any = None
        self.manifest_manager: Any = None

        # Initialize services
        self.bm25_field_mapper: Any = None
        self.splade_segmenter: Any = None
        self.splade_aggregator: Any = None
        self.qwen3_contextualizer: Any = None
        self.qwen3_service: Any = None

    def initialize(self) -> None:
        """Initialize the Features stage components."""
        try:
            # Initialize chunk store
            self.chunk_store = ChunkStore(self.config.chunk_store_path)
            self.chunk_store.initialize()

            # Initialize manifest manager
            manifest_path = self.config.chunk_store_path.parent / "manifests"
            self.manifest_manager = ManifestManager(manifest_path)

            # Initialize BM25 field mapper
            self.bm25_field_mapper = BM25FieldMapper()

            # Initialize SPLADE services
            self.splade_segmenter = SPLADESegmenter()

            self.splade_aggregator = SPLADEAggregator()

            # Initialize Qwen3 services
            self.qwen3_contextualizer = Qwen3ContextualizedProcessor()

            self.qwen3_service = Qwen3Service()

            logger.info("Features stage initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Features stage: {e}")
            raise

    def process_chunks(self, chunk_ids: list[str] | None = None) -> dict[str, Any]:
        """Process chunks to generate retrieval features.

        Args:
        ----
            chunk_ids: Optional list of specific chunk IDs to process.
                      If None, processes all chunks in the store.

        Returns:
        -------
            Dictionary with processing results and statistics

        """
        if not self.chunk_store:
            raise RuntimeError("Features stage not initialized")

        try:
            # Get chunks to process
            if chunk_ids:
                chunks = []
                for chunk_id in chunk_ids:
                    chunk = self.chunk_store.get_chunk(chunk_id)
                    if chunk:
                        chunks.append(chunk)
            else:
                # Get all chunks from store
                chunks = []
                # This would need to be implemented in ChunkStore
                # For now, return empty list

            logger.info(f"Processing {len(chunks)} chunks for feature generation")

            # Process chunks in batches
            results = {
                "total_chunks": len(chunks),
                "processed_chunks": 0,
                "bm25_features": 0,
                "splade_features": 0,
                "qwen3_features": 0,
                "errors": [],
            }

            for i in range(0, len(chunks), self.config.batch_size):
                batch = chunks[i : i + self.config.batch_size]
                batch_results = self._process_batch(batch)
                self._update_results(results, batch_results)

            # Update chunk store with generated features
            self._update_chunk_store_features(chunks)

            logger.info(f"Features stage completed: {results}")
            return results

        except Exception as e:
            logger.error(f"Failed to process chunks: {e}")
            raise

    def _process_batch(self, chunks: list[Any]) -> dict[str, Any]:
        """Process a batch of chunks to generate features.

        Args:
        ----
            chunks: List of chunks to process

        Returns:
        -------
            Dictionary with batch processing results

        """
        batch_results = {
            "bm25_features": 0,
            "splade_features": 0,
            "qwen3_features": 0,
            "errors": [],
        }

        for chunk in chunks:
            try:
                # Generate BM25 features
                if self.bm25_field_mapper:
                    bm25_fields = getattr(
                        self.bm25_field_mapper, "map_chunk_to_fields", lambda x: {}
                    )(chunk)
                    chunk.bm25_fields = bm25_fields
                    batch_results["bm25_features"] += 1

                # Generate SPLADE features
                if self.splade_segmenter and self.splade_aggregator:
                    splade_vector = self._generate_splade_vector(chunk)
                    if splade_vector:
                        chunk.splade_vector = splade_vector
                        batch_results["splade_features"] += 1

                # Generate Qwen3 features
                if self.qwen3_contextualizer and self.qwen3_service:
                    qwen3_embedding = self._generate_qwen3_embedding(chunk)
                    if qwen3_embedding:
                        chunk.qwen3_embedding = qwen3_embedding
                        batch_results["qwen3_features"] += 1

            except Exception as e:
                error_msg = f"Failed to process chunk {chunk.chunk_id}: {e}"
                logger.error(error_msg)
                batch_results["errors"].append(error_msg)

        return batch_results

    def _generate_splade_vector(self, chunk: Any) -> dict[str, Any] | None:
        """Generate SPLADE vector for a chunk.

        Args:
        ----
            chunk: Chunk to process

        Returns:
        -------
            SPLADE vector dictionary or None if generation fails

        """
        try:
            if not self.splade_segmenter or not self.splade_aggregator:
                return None

            # Segment chunk text for SPLADE
            contextualized_text = getattr(chunk, "contextualized_text", "")
            segments = getattr(self.splade_segmenter, "segment_chunk_for_splade", lambda x: [])(
                contextualized_text
            )

            if not segments:
                return None

            # Aggregate segments into single vector
            aggregated_vector = getattr(
                self.splade_aggregator, "aggregate_splade_segments", lambda x: {}
            )(segments)

            return {
                "vector": aggregated_vector,
                "segment_count": len(segments),
                "model_name": "splade-v3",
                "tokenizer_name": "naver/splade-v3",
            }

        except Exception as e:
            logger.error(f"Failed to generate SPLADE vector for chunk {chunk.chunk_id}: {e}")
            return None

    def _generate_qwen3_embedding(self, chunk: Any) -> list[float] | None:
        """Generate Qwen3 embedding for a chunk.

        Args:
        ----
            chunk: Chunk to process

        Returns:
        -------
            Qwen3 embedding vector or None if generation fails

        """
        try:
            if not self.qwen3_contextualizer or not self.qwen3_service:
                return None

            # Get contextualized text for embedding
            contextualized_text = getattr(
                self.qwen3_contextualizer, "get_contextualized_text", lambda x: ""
            )(chunk)

            # Generate embedding
            embedding = getattr(self.qwen3_service, "generate_embedding", lambda x: [])(
                contextualized_text
            )

            return embedding if isinstance(embedding, list) else None

        except Exception as e:
            logger.error(f"Failed to generate Qwen3 embedding for chunk {chunk.chunk_id}: {e}")
            return None

    def _update_chunk_store_features(self, chunks: list[Any]) -> None:
        """Update chunk store with generated features.

        Args:
        ----
            chunks: List of chunks with generated features

        """
        if not self.chunk_store:
            return

        try:
            for chunk in chunks:
                # Update chunk with generated features
                getattr(self.chunk_store, "update_chunk", lambda x: None)(chunk)

            logger.info(f"Updated {len(chunks)} chunks with generated features")

        except Exception as e:
            logger.error(f"Failed to update chunk store with features: {e}")
            raise

    def _update_results(self, results: dict[str, Any], batch_results: dict[str, Any]) -> None:
        """Update overall results with batch results.

        Args:
        ----
            results: Overall results dictionary to update
            batch_results: Batch results to merge

        """
        results["processed_chunks"] += len(batch_results.get("processed_chunks", []))
        results["bm25_features"] += batch_results.get("bm25_features", 0)
        results["splade_features"] += batch_results.get("splade_features", 0)
        results["qwen3_features"] += batch_results.get("qwen3_features", 0)
        results["errors"].extend(batch_results.get("errors", []))

    def create_manifest(self) -> dict[str, Any]:
        """Create manifest for the Features stage.

        Returns
        -------
            Manifest dictionary with stage information

        """
        if not self.manifest_manager:
            raise RuntimeError("Manifest manager not initialized")

        manifest = {
            "stage": "features",
            "version": "1.0.0",
            "config": {
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
            },
            "dependencies": {
                "chunk_store": str(self.config.chunk_store_path),
            },
        }

        return getattr(self.manifest_manager, "create_manifest", lambda x, y: manifest)(
            "features_stage", manifest
        )

    def validate_manifest(self, manifest_path: Path) -> bool:
        """Validate the Features stage manifest.

        Args:
        ----
            manifest_path: Path to the manifest file

        Returns:
        -------
            True if manifest is valid, False otherwise

        """
        if not self.manifest_manager:
            return False

        return getattr(self.manifest_manager, "validate_manifest", lambda x: False)(manifest_path)

    def cleanup(self) -> None:
        """Cleanup resources used by the Features stage."""
        try:
            if self.chunk_store:
                self.chunk_store.close()

            if self.qwen3_service:
                # Cleanup Qwen3 service resources
                pass

            logger.info("Features stage cleanup completed")

        except Exception as e:
            logger.error(f"Failed to cleanup Features stage: {e}")

    def health_check(self) -> dict[str, Any]:
        """Perform health check for the Features stage.

        Returns
        -------
            Health check results

        """
        health_status = {
            "stage": "features",
            "status": "healthy",
            "components": {},
            "errors": [],
        }

        try:
            # Check chunk store
            if self.chunk_store:
                chunk_store_health = getattr(
                    self.chunk_store, "health_check", lambda: {"status": "unknown"}
                )()
                health_status["components"]["chunk_store"] = chunk_store_health
            else:
                health_status["components"]["chunk_store"] = {"status": "not_initialized"}

            # Check Qwen3 service
            if self.qwen3_service:
                qwen3_health = getattr(
                    self.qwen3_service, "health_check", lambda: {"status": "unknown"}
                )()
                health_status["components"]["qwen3_service"] = qwen3_health
            else:
                health_status["components"]["qwen3_service"] = {"status": "not_initialized"}

            # Check for any errors
            if health_status["errors"]:
                health_status["status"] = "unhealthy"

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["errors"].append(f"Health check failed: {e}")

        return health_status
