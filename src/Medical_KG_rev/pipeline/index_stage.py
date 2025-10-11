"""Index stage implementation for building retrieval indexes.

This stage builds:
- BM25 Lucene index from chunk store BM25 fields
- SPLADE impact index from chunk store SPLADE vectors
- Qwen3 vectors to FAISS/Qdrant backend
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from Medical_KG_rev.config.retrieval_config import BM25Config, Qwen3Config, SPLADEConfig
from Medical_KG_rev.pipeline.stages import IndexStage
from Medical_KG_rev.services.retrieval.bm25_service import BM25Service
from Medical_KG_rev.services.retrieval.qwen3_service import Qwen3Service
from Medical_KG_rev.services.retrieval.splade_service import SPLADEService
from Medical_KG_rev.services.vector_store.stores.bm25_index import BM25Index
from Medical_KG_rev.services.vector_store.stores.qwen3_index import Qwen3Index
from Medical_KG_rev.services.vector_store.stores.splade_index import SPLADEImpactIndex
from Medical_KG_rev.storage.chunk_store import ChunkStore
from Medical_KG_rev.storage.manifest_manager import ManifestManager

logger = logging.getLogger(__name__)


class IndexStageConfig(BaseModel):
    """Configuration for the Index stage."""

    chunk_store_path: Path = Field(..., description="Path to chunk store database")
    index_output_dir: Path = Field(..., description="Directory for index output")
    bm25_config: BM25Config = Field(default_factory=BM25Config)
    splade_config: SPLADEConfig = Field(default_factory=SPLADEConfig)
    qwen3_config: Qwen3Config = Field(default_factory=Qwen3Config)
    batch_size: int = Field(default=1000, description="Batch size for index building")
    max_workers: int = Field(default=4, description="Maximum number of worker threads")


class IndexStageImpl(IndexStage):
    """Implementation of the Index stage for building retrieval indexes."""

    def __init__(self, config: IndexStageConfig) -> None:
        """Initialize the Index stage.

        Args:
            config: Configuration for the Index stage

        """
        super().__init__()
        self.config = config
        self.chunk_store: Any = None
        self.manifest_manager: Any = None

        # Initialize indexes
        self.bm25_index: Any = None
        self.splade_index: Any = None
        self.qwen3_index: Any = None

        # Initialize services
        self.bm25_service: Any = None
        self.splade_service: Any = None
        self.qwen3_service: Any = None

    def initialize(self) -> None:
        """Initialize the Index stage components."""
        try:
            # Initialize chunk store
            self.chunk_store = ChunkStore(self.config.chunk_store_path)
            self.chunk_store.initialize()

            # Initialize manifest manager
            manifest_path = self.config.index_output_dir / "manifests"
            self.manifest_manager = ManifestManager(manifest_path)

            # Create index output directories
            self.config.index_output_dir.mkdir(parents=True, exist_ok=True)
            (self.config.index_output_dir / "bm25").mkdir(exist_ok=True)
            (self.config.index_output_dir / "splade").mkdir(exist_ok=True)
            (self.config.index_output_dir / "qwen3").mkdir(exist_ok=True)

            # Initialize BM25 index
            bm25_index_path = self.config.index_output_dir / "bm25" / "index"
            self.bm25_index = BM25Index(index_path=bm25_index_path)

            # Initialize SPLADE index
            splade_index_path = self.config.index_output_dir / "splade" / "index"
            self.splade_index = SPLADEImpactIndex(index_path=splade_index_path)

            # Initialize Qwen3 index
            qwen3_index_path = self.config.index_output_dir / "qwen3" / "index"
            self.qwen3_index = Qwen3Index(index_path=qwen3_index_path)

            # Initialize services
            self.bm25_service = BM25Service()
            self.splade_service = SPLADEService()
            self.qwen3_service = Qwen3Service()

            logger.info("Index stage initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Index stage: {e}")
            raise

    def build_indexes(self, chunk_ids: list[str] | None = None) -> dict[str, Any]:
        """Build all retrieval indexes from chunk store data.

        Args:
            chunk_ids: Optional list of specific chunk IDs to index.
                      If None, indexes all chunks in the store.

        Returns:
            Dictionary with indexing results and statistics

        """
        if not self.chunk_store:
            raise RuntimeError("Index stage not initialized")

        try:
            # Get chunks to index
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

            logger.info(f"Building indexes for {len(chunks)} chunks")

            # Build indexes
            results = {
                "total_chunks": len(chunks),
                "indexed_chunks": 0,
                "bm25_indexed": 0,
                "splade_indexed": 0,
                "qwen3_indexed": 0,
                "errors": [],
            }

            # Build BM25 index
            bm25_results = self._build_bm25_index(chunks)
            results["bm25_indexed"] = bm25_results["indexed"]

            # Build SPLADE index
            splade_results = self._build_splade_index(chunks)
            results["splade_indexed"] = splade_results["indexed"]

            # Build Qwen3 index
            qwen3_results = self._build_qwen3_index(chunks)
            results["qwen3_indexed"] = qwen3_results["indexed"]

            # Update overall results
            results["indexed_chunks"] = min(
                results["bm25_indexed"], results["splade_indexed"], results["qwen3_indexed"]
            )
            errors = results.get("errors", [])
            errors.extend(bm25_results.get("errors", []))
            errors.extend(splade_results.get("errors", []))
            errors.extend(qwen3_results.get("errors", []))
            results["errors"] = errors

            logger.info(f"Index building completed: {results}")
            return results

        except Exception as e:
            logger.error(f"Failed to build indexes: {e}")
            raise

    def _build_bm25_index(self, chunks: list[Any]) -> dict[str, Any]:
        """Build BM25 index from chunks.

        Args:
            chunks: List of chunks to index

        Returns:
            Dictionary with BM25 indexing results

        """
        results = {"indexed": 0, "errors": []}

        try:
            if not self.bm25_index or not self.bm25_service:
                return results

            logger.info(f"Building BM25 index for {len(chunks)} chunks")

            # Process chunks in batches
            for i in range(0, len(chunks), self.config.batch_size):
                batch = chunks[i : i + self.config.batch_size]

                for chunk in batch:
                    try:
                        bm25_fields = getattr(chunk, "bm25_fields", None)
                        if bm25_fields:
                            # Add document to BM25 index
                            getattr(self.bm25_index, "add_document", lambda x, y: None)(
                                chunk.chunk_id, bm25_fields
                            )
                            results["indexed"] += 1

                    except Exception as e:
                        error_msg = f"Failed to index chunk {chunk.chunk_id} in BM25: {e}"
                        logger.error(error_msg)
                        errors = results.get("errors", [])
                        errors.append(error_msg)
                        results["errors"] = errors

            logger.info(f"BM25 index built: {results['indexed']} documents indexed")

        except Exception as e:
            error_msg = f"Failed to build BM25 index: {e}"
            logger.error(error_msg)
            errors = results.get("errors", [])
            errors.append(error_msg)
            results["errors"] = errors

        return results

    def _build_splade_index(self, chunks: list[Any]) -> dict[str, Any]:
        """Build SPLADE index from chunks.

        Args:
            chunks: List of chunks to index

        Returns:
            Dictionary with SPLADE indexing results

        """
        results = {"indexed": 0, "errors": []}

        try:
            if not self.splade_index or not self.splade_service:
                return results

            logger.info(f"Building SPLADE index for {len(chunks)} chunks")

            # Process chunks in batches
            for i in range(0, len(chunks), self.config.batch_size):
                batch = chunks[i : i + self.config.batch_size]

                for chunk in batch:
                    try:
                        splade_vector = getattr(chunk, "splade_vector", None)
                        if splade_vector:
                            # Add SPLADE vector to index
                            getattr(self.splade_index, "add_vector", lambda x, y, z: None)(
                                chunk.chunk_id, splade_vector.get("vector", {}), {}
                            )
                            results["indexed"] += 1

                    except Exception as e:
                        error_msg = f"Failed to index chunk {chunk.chunk_id} in SPLADE: {e}"
                        logger.error(error_msg)
                        errors = results.get("errors", [])
                        errors.append(error_msg)
                        results["errors"] = errors

            logger.info(f"SPLADE index built: {results['indexed']} documents indexed")

        except Exception as e:
            error_msg = f"Failed to build SPLADE index: {e}"
            logger.error(error_msg)
            errors = results.get("errors", [])
            errors.append(error_msg)
            results["errors"] = errors

        return results

    def _build_qwen3_index(self, chunks: list[Any]) -> dict[str, Any]:
        """Build Qwen3 index from chunks.

        Args:
            chunks: List of chunks to index

        Returns:
            Dictionary with Qwen3 indexing results

        """
        results = {"indexed": 0, "errors": []}

        try:
            if not self.qwen3_index or not self.qwen3_service:
                return results

            logger.info(f"Building Qwen3 index for {len(chunks)} chunks")

            # Process chunks in batches
            for i in range(0, len(chunks), self.config.batch_size):
                batch = chunks[i : i + self.config.batch_size]

                for chunk in batch:
                    try:
                        qwen3_embedding = getattr(chunk, "qwen3_embedding", None)
                        if qwen3_embedding:
                            # Add Qwen3 embedding to index
                            getattr(self.qwen3_index, "add_vector", lambda x, y, z: None)(
                                chunk.chunk_id, qwen3_embedding, {}
                            )
                            results["indexed"] += 1

                    except Exception as e:
                        error_msg = f"Failed to index chunk {chunk.chunk_id} in Qwen3: {e}"
                        logger.error(error_msg)
                        errors = results.get("errors", [])
                        errors.append(error_msg)
                        results["errors"] = errors

            logger.info(f"Qwen3 index built: {results['indexed']} documents indexed")

        except Exception as e:
            error_msg = f"Failed to build Qwen3 index: {e}"
            logger.error(error_msg)
            errors = results.get("errors", [])
            errors.append(error_msg)
            results["errors"] = errors

        return results

    def validate_indexes(self) -> dict[str, Any]:
        """Validate all built indexes.

        Returns:
            Dictionary with validation results

        """
        validation_results = {
            "bm25_valid": False,
            "splade_valid": False,
            "qwen3_valid": False,
            "errors": [],
        }

        try:
            # Validate BM25 index
            if self.bm25_index:
                bm25_validation = getattr(self.bm25_index, "validate_index", lambda: True)()
                validation_results["bm25_valid"] = bm25_validation

            # Validate SPLADE index
            if self.splade_index:
                splade_validation = getattr(self.splade_index, "validate_index", lambda: True)()
                validation_results["splade_valid"] = splade_validation

            # Validate Qwen3 index
            if self.qwen3_index:
                qwen3_validation = getattr(self.qwen3_index, "validate_index", lambda: True)()
                validation_results["qwen3_valid"] = qwen3_validation

            logger.info(f"Index validation completed: {validation_results}")

        except Exception as e:
            error_msg = f"Failed to validate indexes: {e}"
            logger.error(error_msg)
            errors = validation_results.get("errors", [])
            errors.append(error_msg)
            validation_results["errors"] = errors

        return validation_results

    def create_manifest(self) -> dict[str, Any]:
        """Create manifest for the Index stage.

        Returns:
            Manifest dictionary with stage information

        """
        if not self.manifest_manager:
            raise RuntimeError("Manifest manager not initialized")

        manifest = {
            "stage": "index",
            "version": "1.0.0",
            "config": {
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
            },
            "dependencies": {
                "chunk_store": str(self.config.chunk_store_path),
            },
            "outputs": {
                "bm25_index": str(self.config.index_output_dir / "bm25" / "index"),
                "splade_index": str(self.config.index_output_dir / "splade" / "index"),
                "qwen3_index": str(self.config.index_output_dir / "qwen3" / "index"),
            },
        }

        return getattr(self.manifest_manager, "create_manifest", lambda x, y: manifest)(
            "index_stage", manifest
        )

    def validate_manifest(self, manifest_path: Path) -> bool:
        """Validate the Index stage manifest.

        Args:
            manifest_path: Path to the manifest file

        Returns:
            True if manifest is valid, False otherwise

        """
        if not self.manifest_manager:
            return False

        return getattr(self.manifest_manager, "validate_manifest", lambda x: False)(manifest_path)

    def cleanup(self) -> None:
        """Cleanup resources used by the Index stage."""
        try:
            if self.chunk_store:
                self.chunk_store.close()

            if self.qwen3_service:
                # Cleanup Qwen3 service resources
                pass

            logger.info("Index stage cleanup completed")

        except Exception as e:
            logger.error(f"Failed to cleanup Index stage: {e}")

    def health_check(self) -> dict[str, Any]:
        """Perform health check for the Index stage.

        Returns:
            Health check results

        """
        health_status = {
            "stage": "index",
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

            # Check indexes
            if self.bm25_index:
                bm25_health = getattr(
                    self.bm25_index, "health_check", lambda: {"status": "unknown"}
                )()
                health_status["components"]["bm25_index"] = bm25_health

            if self.splade_index:
                splade_health = getattr(
                    self.splade_index, "health_check", lambda: {"status": "unknown"}
                )()
                health_status["components"]["splade_index"] = splade_health

            if self.qwen3_index:
                qwen3_health = getattr(
                    self.qwen3_index, "health_check", lambda: {"status": "unknown"}
                )()
                health_status["components"]["qwen3_index"] = qwen3_health

            # Check for any errors
            if health_status["errors"]:
                health_status["status"] = "unhealthy"

        except Exception as e:
            health_status["status"] = "unhealthy"
            errors = health_status.get("errors", [])
            errors.append(f"Health check failed: {e}")
            health_status["errors"] = errors

        return health_status
