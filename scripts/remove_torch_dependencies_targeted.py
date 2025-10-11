#!/usr/bin/env python3
"""Targeted script to remove torch dependencies from specific files.

This script removes torch-dependent functionality and replaces it with
appropriate gRPC service calls or removes it entirely.
"""

import re
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Remove Torch Dependencies - Targeted Approach")
console = Console()


class TargetedTorchRemover:
    """Removes torch dependencies with targeted replacements."""

    def __init__(self):
        self.modified_files: list[Path] = []
        self.errors: list[tuple[Path, str]] = []

    def process_gpu_manager(self, file_path: Path) -> bool:
        """Process GPU manager file to remove torch dependencies."""
        try:
            content = file_path.read_text()
            original_content = content

            # Replace the entire GPU manager with a gRPC client-based implementation
            new_content = '''"""GPU device detection and resource management utilities - Torch-free version."""

from __future__ import annotations

import asyncio
from typing import Optional

import structlog

from ..clients.gpu_client import GPUClientManager

logger = structlog.get_logger(__name__)


class GpuNotAvailableError(RuntimeError):
    """Raised when CUDA GPUs are unavailable for the microservices."""


class GpuDevice:
    """Representation of a single GPU device."""

    def __init__(self, index: int, name: str, total_memory_mb: int):
        self.index = index
        self.name = name
        self.total_memory_mb = total_memory_mb


class GpuServiceManager:
    """GPU service manager that uses gRPC to communicate with GPU services."""

    def __init__(self, min_memory_mb: int = 1000, preferred_device: Optional[int] = None):
        self.min_memory_mb = min_memory_mb
        self.preferred_device = preferred_device
        self._client_manager: Optional[GPUClientManager] = None
        self._device_cache: Optional[GpuDevice] = None

    async def _get_client_manager(self) -> GPUClientManager:
        """Get or create GPU client manager."""
        if self._client_manager is None:
            self._client_manager = GPUClientManager()
            await self._client_manager.initialize()
        return self._client_manager

    async def get_available_device(self) -> GpuDevice:
        """Get an available GPU device via gRPC service."""
        try:
            client_manager = await self._get_client_manager()

            # Get GPU status from service
            status = await client_manager.get_status()
            if not status or not status.get('available', False):
                raise GpuNotAvailableError("No GPU devices available via gRPC service")

            # Get device list from service
            devices = await client_manager.list_devices()
            if not devices:
                raise GpuNotAvailableError("No GPU devices detected via gRPC service")

            # Select appropriate device
            for device_info in devices:
                if device_info.get('total_memory_mb', 0) >= self.min_memory_mb:
                    device = GpuDevice(
                        index=device_info.get('index', 0),
                        name=device_info.get('name', 'Unknown'),
                        total_memory_mb=device_info.get('total_memory_mb', 0)
                    )
                    self._device_cache = device
                    return device

            raise GpuNotAvailableError(f"No GPU device with {self.min_memory_mb}MB+ memory available")

        except Exception as e:
            logger.error("gpu.service.error", error=str(e))
            raise GpuNotAvailableError(f"GPU service communication failed: {e}")

    async def allocate_gpu(self, device: Optional[GpuDevice] = None) -> GpuDevice:
        """Allocate a GPU device via gRPC service."""
        if device is None:
            device = await self.get_available_device()

        try:
            client_manager = await self._get_client_manager()
            allocation = await client_manager.allocate_gpu(device.index)

            if not allocation or not allocation.get('success', False):
                raise GpuNotAvailableError("Failed to allocate GPU via gRPC service")

            logger.info("gpu.allocated", device=device.name, index=device.index)
            return device

        except Exception as e:
            logger.error("gpu.allocation.error", error=str(e))
            raise GpuNotAvailableError(f"GPU allocation failed: {e}")

    async def deallocate_gpu(self, device: GpuDevice) -> None:
        """Deallocate a GPU device via gRPC service."""
        try:
            client_manager = await self._get_client_manager()
            success = await client_manager.deallocate_gpu(device.index)

            if not success:
                logger.warning("gpu.deallocation.failed", device=device.name, index=device.index)
            else:
                logger.info("gpu.deallocated", device=device.name, index=device.index)

        except Exception as e:
            logger.error("gpu.deallocation.error", error=str(e))

    async def close(self) -> None:
        """Close the GPU client manager."""
        if self._client_manager:
            await self._client_manager.close()


# Legacy compatibility - raise error for torch-dependent functionality
def ensure_torch():
    """Legacy function - GPU functionality moved to gRPC services."""
    raise GpuNotAvailableError("GPU functionality moved to gRPC services. Use GpuServiceManager instead.")


def get_available_device() -> GpuDevice:
    """Legacy function - GPU functionality moved to gRPC services."""
    raise GpuNotAvailableError("GPU functionality moved to gRPC services. Use GpuServiceManager instead.")


def allocate_gpu(device: Optional[GpuDevice] = None) -> GpuDevice:
    """Legacy function - GPU functionality moved to gRPC services."""
    raise GpuNotAvailableError("GPU functionality moved to gRPC services. Use GpuServiceManager instead.")


def deallocate_gpu(device: GpuDevice) -> None:
    """Legacy function - GPU functionality moved to gRPC services."""
    raise GpuNotAvailableError("GPU functionality moved to gRPC services. Use GpuServiceManager instead.")
'''

            if new_content != original_content:
                file_path.write_text(new_content)
                self.modified_files.append(file_path)
                return True

            return False

        except Exception as e:
            self.errors.append((file_path, f"Error processing GPU manager: {e}"))
            return False

    def process_metrics(self, file_path: Path) -> bool:
        """Process metrics file to remove torch dependencies."""
        try:
            content = file_path.read_text()
            original_content = content

            # Replace torch-dependent metrics with gRPC service metrics
            new_content = '''"""Prometheus metrics for GPU services - Torch-free version."""

from prometheus_client import Counter, Gauge, Histogram

# GPU service metrics (replacing torch-dependent metrics)
GPU_SERVICE_CALLS_TOTAL = Counter(
    'gpu_service_calls_total',
    'Total number of GPU service calls',
    ['service', 'method', 'status']
)

GPU_SERVICE_CALL_DURATION_SECONDS = Histogram(
    'gpu_service_call_duration_seconds',
    'Duration of GPU service calls',
    ['service', 'method'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

GPU_SERVICE_ERRORS_TOTAL = Counter(
    'gpu_service_errors_total',
    'Total number of GPU service errors',
    ['service', 'method', 'error_type']
)

GPU_MEMORY_USAGE_MB = Gauge(
    'gpu_memory_usage_mb',
    'GPU memory usage in MB',
    ['device_id', 'device_name']
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percentage',
    'GPU utilization percentage',
    ['device_id', 'device_name']
)

GPU_SERVICE_HEALTH_STATUS = Gauge(
    'gpu_service_health_status',
    'GPU service health status (1=healthy, 0=unhealthy)',
    ['service_name']
)

CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['service_name']
)

# Legacy metrics (replaced with service metrics)
GPU_MEMORY_USED = GPU_MEMORY_USAGE_MB  # Alias for compatibility
'''

            if new_content != original_content:
                file_path.write_text(new_content)
                self.modified_files.append(file_path)
                return True

            return False

        except Exception as e:
            self.errors.append((file_path, f"Error processing metrics: {e}"))
            return False

    def process_vector_store_gpu(self, file_path: Path) -> bool:
        """Process vector store GPU file to remove torch dependencies."""
        try:
            content = file_path.read_text()
            original_content = content

            # Replace with gRPC service-based implementation
            new_content = '''"""Vector store GPU utilities - Torch-free version."""

from typing import Any, Dict, List, Optional

import structlog

from ..clients.embedding_client import EmbeddingClientManager

logger = structlog.get_logger(__name__)


class VectorStoreGPU:
    """Vector store GPU operations via gRPC services."""

    def __init__(self):
        self._client_manager: Optional[EmbeddingClientManager] = None

    async def _get_client_manager(self) -> EmbeddingClientManager:
        """Get or create embedding client manager."""
        if self._client_manager is None:
            self._client_manager = EmbeddingClientManager()
            await self._client_manager.initialize()
        return self._client_manager

    async def generate_embeddings(self, texts: List[str], model: str = "default") -> List[List[float]]:
        """Generate embeddings via gRPC service."""
        try:
            client_manager = await self._get_client_manager()
            embeddings = await client_manager.generate_embeddings_batch(texts, model)
            return embeddings
        except Exception as e:
            logger.error("vector_store.embedding.error", error=str(e))
            raise RuntimeError(f"Embedding generation failed: {e}")

    async def similarity_search(self, query_embedding: List[float],
                              index_embeddings: List[List[float]],
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform similarity search using embeddings."""
        try:
            # Simple cosine similarity implementation
            import numpy as np

            query_np = np.array(query_embedding)
            similarities = []

            for i, embedding in enumerate(index_embeddings):
                embedding_np = np.array(embedding)
                similarity = np.dot(query_np, embedding_np) / (
                    np.linalg.norm(query_np) * np.linalg.norm(embedding_np)
                )
                similarities.append({
                    'index': i,
                    'similarity': float(similarity),
                    'embedding': embedding
                })

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error("vector_store.similarity.error", error=str(e))
            raise RuntimeError(f"Similarity search failed: {e}")

    async def close(self) -> None:
        """Close the embedding client manager."""
        if self._client_manager:
            await self._client_manager.close()


# Legacy functions (replaced with gRPC service calls)
def generate_embeddings(texts: List[str], model: str = "default") -> List[List[float]]:
    """Legacy function - embedding generation moved to gRPC services."""
    raise NotImplementedError("Embedding generation moved to gRPC services. Use VectorStoreGPU instead.")


def similarity_search(query_embedding: List[float],
                     index_embeddings: List[List[float]],
                     top_k: int = 10) -> List[Dict[str, Any]]:
    """Legacy function - similarity search moved to gRPC services."""
    raise NotImplementedError("Similarity search moved to gRPC services. Use VectorStoreGPU instead.")
'''

            if new_content != original_content:
                file_path.write_text(new_content)
                self.modified_files.append(file_path)
                return True

            return False

        except Exception as e:
            self.errors.append((file_path, f"Error processing vector store GPU: {e}"))
            return False

    def process_other_files(self, file_path: Path) -> bool:
        """Process other files to remove torch dependencies."""
        try:
            content = file_path.read_text()
            original_content = content

            # Remove torch imports and replace functionality
            lines = content.split("\n")
            new_lines = []

            for line in lines:
                # Remove torch imports
                if re.match(r"^\s*(import torch|from torch)", line):
                    new_lines.append(f"# {line.strip()}  # Removed for torch isolation")
                    continue

                # Remove conditional torch imports
                if "import torch" in line and ("try:" in line or "except:" in line):
                    new_lines.append(f"# {line.strip()}  # Removed for torch isolation")
                    continue

                # Replace torch functionality with errors
                if "torch." in line:
                    # Replace torch calls with NotImplementedError
                    new_line = re.sub(
                        r"torch\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)",
                        'raise NotImplementedError("Torch functionality moved to gRPC services")',
                        line,
                    )
                    if new_line != line:
                        new_lines.append(f"# {line.strip()}  # Replaced for torch isolation")
                        new_lines.append(new_line)
                        continue

                new_lines.append(line)

            new_content = "\n".join(new_lines)

            if new_content != original_content:
                file_path.write_text(new_content)
                self.modified_files.append(file_path)
                return True

            return False

        except Exception as e:
            self.errors.append((file_path, f"Error processing file: {e}"))
            return False

    def process_file(self, file_path: Path) -> bool:
        """Process a single file based on its type."""
        try:
            if "gpu/manager.py" in str(file_path):
                return self.process_gpu_manager(file_path)
            elif "metrics.py" in str(file_path):
                return self.process_metrics(file_path)
            elif "vector_store/gpu.py" in str(file_path):
                return self.process_vector_store_gpu(file_path)
            else:
                return self.process_other_files(file_path)

        except Exception as e:
            self.errors.append((file_path, f"Error processing file: {e}"))
            return False

    def process_all_files(self, src_path: str = "src/Medical_KG_rev") -> None:
        """Process all files with torch dependencies."""
        src_path = Path(src_path)

        # Find all Python files with torch imports
        torch_files = []
        for py_file in src_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                if "import torch" in content or "from torch" in content:
                    torch_files.append(py_file)
            except Exception:
                continue

        if not torch_files:
            console.print("✅ No files with torch dependencies found", style="green")
            return

        console.print(f"📁 Found {len(torch_files)} files with torch dependencies", style="blue")

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("Processing files...", total=len(torch_files))

            for file_path in torch_files:
                progress.update(task, description=f"Processing {file_path.name}...")

                try:
                    modified = self.process_file(file_path)
                    if modified:
                        console.print(f"✅ Modified: {file_path}", style="green")
                    else:
                        console.print(f"ℹ️  No changes needed: {file_path}", style="blue")

                except Exception as e:
                    console.print(f"❌ Error processing {file_path}: {e}", style="red")
                    self.errors.append((file_path, str(e)))

                progress.advance(task)

    def display_results(self) -> None:
        """Display the results of the torch removal process."""
        console.print("\n📊 Targeted Torch Dependency Removal Results", style="bold blue")

        # Summary table
        summary_table = Table(title="Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")

        summary_table.add_row("Files modified", str(len(self.modified_files)))
        summary_table.add_row("Errors encountered", str(len(self.errors)))

        console.print(summary_table)

        # Modified files table
        if self.modified_files:
            modified_table = Table(title="Modified Files")
            modified_table.add_column("File Path", style="cyan")
            modified_table.add_column("Status", style="green")

            for file_path in self.modified_files:
                modified_table.add_row(str(file_path), "✅ Modified")

            console.print(modified_table)

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
            console.print(
                "\n✅ Successfully removed torch dependencies from all files.", style="green"
            )


@app.command()
def remove(src_path: str = typer.Option("src/Medical_KG_rev", help="Source path to process")):
    """Remove torch dependencies with targeted replacements."""
    remover = TargetedTorchRemover()
    remover.process_all_files(src_path)
    remover.display_results()


if __name__ == "__main__":
    app()
