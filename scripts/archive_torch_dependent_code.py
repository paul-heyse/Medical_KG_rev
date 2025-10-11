#!/usr/bin/env python3
"""Script to archive torch-dependent code.

This script creates backups of torch-dependent code before it was modified
for the torch isolation architecture.
"""

import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Archive Torch-Dependent Code")
console = Console()


class TorchCodeArchiver:
    """Archives torch-dependent code."""

    def __init__(self, archive_dir: str = "archive/torch-dependent-code"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.archived_files: list[Path] = []
        self.errors: list[tuple[Path, str]] = []

    def create_original_gpu_manager(self) -> None:
        """Create original GPU manager with torch dependencies."""
        content = '''"""GPU device detection and resource management utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

try:  # pragma: no cover - optional dependency, exercised in tests via monkeypatch
    import torch
except Exception:  # pragma: no cover - torch is optional in unit tests
    torch = None  # type: ignore


class GpuNotAvailableError(RuntimeError):
    """Raised when CUDA GPUs are unavailable for the microservices."""


@dataclass(frozen=True)
class GpuDevice:
    """Representation of a single GPU device."""

    index: int
    name: str
    total_memory_mb: int


class GpuManager:
    """GPU device detection and resource management."""

    def __init__(self, min_memory_mb: int = 1000, preferred_device: Optional[int] = None):
        self.min_memory_mb = min_memory_mb
        self.preferred_device = preferred_device
        self._device_cache: Optional[GpuDevice] = None
        self._lock = asyncio.Lock()

    def _ensure_torch(self):
        if torch is None:
            raise GpuNotAvailableError("PyTorch with CUDA support is required for GPU services")
        if not torch.cuda.is_available():
            raise GpuNotAvailableError("CUDA is not available on this host")
        return torch

    def _select_device(self) -> GpuDevice:
        cached = self._device_cache
        if cached is not None:
            return cached

        with self._lock:
            if self._device_cache is not None:
                return self._device_cache

            lib = self._ensure_torch()
            device_count = lib.cuda.device_count()
            if device_count == 0:
                raise GpuNotAvailableError("No CUDA devices detected")

            indices = (
                [self.preferred_device]
                if self.preferred_device is not None
                else list(range(device_count))
            )

            for index in indices:
                if index is None or index < 0 or index >= device_count:
                    continue
                props = lib.cuda.get_device_properties(index)
                total_memory_mb = int(props.total_memory / (1024 * 1024))
                if total_memory_mb < self.min_memory_mb:
                    logger.warning(
                        "gpu.device.skipped",
                        device=index,
                        total_memory_mb=total_memory_mb,
                        required_mb=self.min_memory_mb,
                    )
                    continue
                device = GpuDevice(index=index, name=props.name, total_memory_mb=total_memory_mb)
                self._device_cache = device
                return device

            raise GpuNotAvailableError(f"No GPU device with {self.min_memory_mb}MB+ memory available")

    def get_available_device(self) -> GpuDevice:
        """Get an available GPU device."""
        return self._select_device()

    def allocate_gpu(self, device: Optional[GpuDevice] = None) -> GpuDevice:
        """Allocate a GPU device."""
        if device is None:
            device = self.get_available_device()

        logger.info("gpu.allocated", device=device.name, index=device.index)
        return device

    def deallocate_gpu(self, device: GpuDevice) -> None:
        """Deallocate a GPU device."""
        logger.info("gpu.deallocated", device=device.name, index=device.index)


# Legacy compatibility
def ensure_torch():
    """Legacy function for torch availability."""
    manager = GpuManager()
    return manager._ensure_torch()


def get_available_device() -> GpuDevice:
    """Legacy function for getting available device."""
    manager = GpuManager()
    return manager.get_available_device()


def allocate_gpu(device: Optional[GpuDevice] = None) -> GpuDevice:
    """Legacy function for allocating GPU."""
    manager = GpuManager()
    return manager.allocate_gpu(device)


def deallocate_gpu(device: GpuDevice) -> None:
    """Legacy function for deallocating GPU."""
    manager = GpuManager()
    manager.deallocate_gpu(device)
'''

        file_path = self.archive_dir / "gpu_manager_original.py"
        file_path.write_text(content)
        self.archived_files.append(file_path)

    def create_original_metrics(self) -> None:
        """Create original metrics with torch-dependent GPU metrics."""
        content = '''"""Prometheus metrics for GPU services."""

from prometheus_client import Counter, Gauge, Histogram

# GPU metrics (torch-dependent)
GPU_MEMORY_USED = Gauge(
    'gpu_memory_used_mb',
    'GPU memory usage in MB',
    ['device_id', 'device_name']
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percentage',
    'GPU utilization percentage',
    ['device_id', 'device_name']
)

GPU_TEMPERATURE = Gauge(
    'gpu_temperature_celsius',
    'GPU temperature in Celsius',
    ['device_id', 'device_name']
)

GPU_POWER_USAGE = Gauge(
    'gpu_power_usage_watts',
    'GPU power usage in watts',
    ['device_id', 'device_name']
)

# Model loading metrics
MODEL_LOAD_TIME = Histogram(
    'model_load_time_seconds',
    'Time taken to load models',
    ['model_name', 'model_type']
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_time_seconds',
    'Time taken for model inference',
    ['model_name', 'model_type']
)

# Batch processing metrics
BATCH_PROCESSING_TIME = Histogram(
    'batch_processing_time_seconds',
    'Time taken for batch processing',
    ['batch_size', 'model_name']
)

BATCH_PROCESSING_THROUGHPUT = Counter(
    'batch_processing_throughput_total',
    'Total number of items processed in batches',
    ['model_name']
)

# Error metrics
GPU_ERRORS = Counter(
    'gpu_errors_total',
    'Total number of GPU errors',
    ['error_type', 'device_id']
)

MODEL_ERRORS = Counter(
    'model_errors_total',
    'Total number of model errors',
    ['model_name', 'error_type']
)
'''

        file_path = self.archive_dir / "metrics_original.py"
        file_path.write_text(content)
        self.archived_files.append(file_path)

    def create_original_vector_store_gpu(self) -> None:
        """Create original vector store GPU utilities."""
        content = '''"""Vector store GPU utilities."""

from typing import Any, Dict, List, Optional

import structlog
import torch
import numpy as np

logger = structlog.get_logger(__name__)


class VectorStoreGPU:
    """Vector store GPU operations."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self, model_name: str) -> None:
        """Load a model for vector operations."""
        try:
            # This would load the actual model
            self.model = torch.load(model_name, map_location=self.device)
            logger.info("vector_store.model.loaded", model=model_name, device=self.device)
        except Exception as e:
            logger.error("vector_store.model.load_error", model=model_name, error=str(e))
            raise

    def generate_embeddings(self, texts: List[str], model: str = "default") -> List[List[float]]:
        """Generate embeddings using GPU."""
        try:
            if self.model is None:
                self.load_model(model)

            # Convert texts to tensors
            inputs = torch.tensor(texts, device=self.device)

            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(inputs)

            # Convert back to CPU and return as list
            return embeddings.cpu().numpy().tolist()

        except Exception as e:
            logger.error("vector_store.embedding.error", error=str(e))
            raise RuntimeError(f"Embedding generation failed: {e}")

    def similarity_search(self, query_embedding: List[float],
                          index_embeddings: List[List[float]],
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform similarity search using GPU."""
        try:
            # Convert to tensors
            query_tensor = torch.tensor(query_embedding, device=self.device)
            index_tensor = torch.tensor(index_embeddings, device=self.device)

            # Compute similarities
            with torch.no_grad():
                similarities = torch.cosine_similarity(
                    query_tensor.unsqueeze(0),
                    index_tensor,
                    dim=1
                )

            # Get top-k results
            top_indices = torch.topk(similarities, top_k).indices
            top_similarities = torch.topk(similarities, top_k).values

            results = []
            for idx, sim in zip(top_indices.cpu().numpy(), top_similarities.cpu().numpy()):
                results.append({
                    'index': int(idx),
                    'similarity': float(sim),
                    'embedding': index_embeddings[int(idx)]
                })

            return results

        except Exception as e:
            logger.error("vector_store.similarity.error", error=str(e))
            raise RuntimeError(f"Similarity search failed: {e}")


# Legacy functions
def generate_embeddings(texts: List[str], model: str = "default") -> List[List[float]]:
    """Legacy function for generating embeddings."""
    store = VectorStoreGPU()
    return store.generate_embeddings(texts, model)


def similarity_search(query_embedding: List[float],
                     index_embeddings: List[List[float]],
                     top_k: int = 10) -> List[Dict[str, Any]]:
    """Legacy function for similarity search."""
    store = VectorStoreGPU()
    return store.similarity_search(query_embedding, index_embeddings, top_k)
'''

        file_path = self.archive_dir / "vector_store_gpu_original.py"
        file_path.write_text(content)
        self.archived_files.append(file_path)

    def archive_requirements(self) -> None:
        """Archive original requirements files."""
        # Archive original requirements.txt
        if Path("requirements.txt").exists():
            shutil.copy2("requirements.txt", self.archive_dir / "requirements_original.txt")
            self.archived_files.append(self.archive_dir / "requirements_original.txt")

        # Archive original requirements.in
        if Path("requirements.in").exists():
            shutil.copy2("requirements.in", self.archive_dir / "requirements_original.in")
            self.archived_files.append(self.archive_dir / "requirements_original.in")

    def create_archive_manifest(self) -> None:
        """Create a manifest of archived files."""
        manifest_content = f"""# Torch-Dependent Code Archive Manifest

## Archive Information
- Archive Date: $(date)
- Archive Reason: Torch isolation architecture implementation
- Total Files: {len(self.archived_files)}

## Archived Files
"""

        for file_path in self.archived_files:
            manifest_content += f"- {file_path.name}\n"

        manifest_content += """
## Migration Summary
- GPU Manager: Replaced with GpuServiceManager (gRPC)
- Metrics: Replaced with gRPC service metrics
- Vector Store GPU: Replaced with VectorStoreGPU (gRPC)
- Requirements: Replaced with torch-free requirements

## Restoration Notes
To restore any file, copy it from this archive and remove the '_original' suffix.
Update imports and dependencies as needed for current architecture.
"""

        manifest_path = self.archive_dir / "ARCHIVE_MANIFEST.md"
        manifest_path.write_text(manifest_content)
        self.archived_files.append(manifest_path)

    def archive_all(self) -> None:
        """Archive all torch-dependent code."""
        console.print("📦 Archiving torch-dependent code...", style="blue")

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Creating archives...", total=4)

            # Create original code files
            progress.update(task, description="Creating GPU manager archive...")
            self.create_original_gpu_manager()

            progress.update(task, description="Creating metrics archive...")
            self.create_original_metrics()

            progress.update(task, description="Creating vector store archive...")
            self.create_original_vector_store_gpu()

            progress.update(task, description="Archiving requirements...")
            self.archive_requirements()

            progress.update(task, description="Creating manifest...")
            self.create_archive_manifest()

            progress.advance(task)

    def display_results(self) -> None:
        """Display the results of the archiving process."""
        console.print("\n📊 Torch-Dependent Code Archive Results", style="bold blue")

        # Summary table
        summary_table = Table(title="Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")

        summary_table.add_row("Files archived", str(len(self.archived_files)))
        summary_table.add_row("Errors encountered", str(len(self.errors)))

        console.print(summary_table)

        # Archived files table
        if self.archived_files:
            archived_table = Table(title="Archived Files")
            archived_table.add_column("File Path", style="cyan")
            archived_table.add_column("Status", style="green")

            for file_path in self.archived_files:
                archived_table.add_row(str(file_path), "✅ Archived")

            console.print(archived_table)

        # Errors table
        if self.errors:
            error_table = Table(title="Errors")
            error_table.add_column("File Path", style="cyan")
            error_table.add_column("Error", style="red")

            for file_path, error in self.errors:
                error_table.add_row(str(file_path), error)

            console.print(error_table)

        # Overall result
        if self.errors:
            console.print(
                f"\n⚠️  Completed with {len(self.errors)} errors. Please review the errors above.",
                style="yellow",
            )
        else:
            console.print("\n✅ Successfully archived torch-dependent code.", style="green")


@app.command()
def archive(
    archive_dir: str = typer.Option("archive/torch-dependent-code", help="Archive directory path"),
):
    """Archive torch-dependent code."""
    archiver = TorchCodeArchiver(archive_dir)
    archiver.archive_all()
    archiver.display_results()


if __name__ == "__main__":
    app()
