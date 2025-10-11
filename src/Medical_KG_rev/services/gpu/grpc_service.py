"""gRPC service for GPU operations."""

from __future__ import annotations

import logging
from typing import Any

import grpc
from google.protobuf import empty_pb2

try:
    from proto import gpu_pb2, gpu_pb2_grpc
except ImportError:
    # Fallback for when protobuf files aren't generated yet
    gpu_pb2 = None
    gpu_pb2_grpc = None

from Medical_KG_rev.services.gpu.manager import (
    GpuServiceManager,
)

logger = logging.getLogger(__name__)


class GPUService(gpu_pb2_grpc.GPUServiceServicer if gpu_pb2_grpc else object):
    """gRPC service implementation for GPU operations."""

    def __init__(self) -> None:
        """Initialize the GPU gRPC service."""
        self.logger = logger
        self.gpu_manager = GpuServiceManager()

    def GetGPUStatus(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        """Get GPU status."""
        try:
            if gpu_pb2:
                # Real implementation
                status = self.gpu_manager.get_status()
                response = gpu_pb2.GetGPUStatusResponse()
                response.available = status.get("available", False)
                response.device_count = status.get("device_count", 0)
                response.total_memory = status.get("total_memory", 0)
                response.free_memory = status.get("free_memory", 0)
                return response
            else:
                # Mock implementation
                return {"available": False, "device_count": 0}

        except Exception as e:
            self.logger.error(f"GetGPUStatus failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"GetGPUStatus failed: {e}")
            return gpu_pb2.GetGPUStatusResponse() if gpu_pb2 else {}

    def ListGPUs(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        """List available GPUs."""
        try:
            if gpu_pb2:
                # Real implementation
                gpus = self.gpu_manager.list_gpus()
                response = gpu_pb2.ListGPUsResponse()
                for gpu in gpus:
                    gpu_info = response.gpus.add()
                    gpu_info.id = gpu.get("id", "")
                    gpu_info.name = gpu.get("name", "")
                    gpu_info.memory_total = gpu.get("memory_total", 0)
                    gpu_info.memory_free = gpu.get("memory_free", 0)
                return response
            else:
                # Mock implementation
                return {"gpus": []}

        except Exception as e:
            self.logger.error(f"ListGPUs failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"ListGPUs failed: {e}")
            return gpu_pb2.ListGPUsResponse() if gpu_pb2 else {}

    def AllocateGPU(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        """Allocate GPU resources."""
        try:
            if gpu_pb2:
                # Real implementation
                allocation = self.gpu_manager.allocate_gpu(
                    memory_mb=request.memory_mb,
                    timeout=request.timeout,
                )
                response = gpu_pb2.AllocateGPUResponse()
                response.success = allocation.get("success", False)
                response.gpu_id = allocation.get("gpu_id", "")
                response.allocation_id = allocation.get("allocation_id", "")
                return response
            else:
                # Mock implementation
                return {"success": False, "gpu_id": "", "allocation_id": ""}

        except Exception as e:
            self.logger.error(f"AllocateGPU failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"AllocateGPU failed: {e}")
            return gpu_pb2.AllocateGPUResponse() if gpu_pb2 else {}

    def ReleaseGPU(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        """Release GPU resources."""
        try:
            if gpu_pb2:
                # Real implementation
                success = self.gpu_manager.release_gpu(request.allocation_id)
                response = gpu_pb2.ReleaseGPUResponse()
                response.success = success
                return response
            else:
                # Mock implementation
                return {"success": True}

        except Exception as e:
            self.logger.error(f"ReleaseGPU failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"ReleaseGPU failed: {e}")
            return gpu_pb2.ReleaseGPUResponse() if gpu_pb2 else {}

    def health_check(self) -> dict[str, Any]:
        """Check service health."""
        return {
            "service": "gpu_grpc",
            "status": "healthy",
            "gpu_manager": self.gpu_manager.health_check(),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        return {
            "service": "gpu_grpc",
            "gpu_manager_metrics": self.gpu_manager.get_metrics(),
        }


class GPUServiceFactory:
    """Factory for creating GPU gRPC services."""

    @staticmethod
    def create() -> GPUService:
        """Create a GPU gRPC service instance."""
        return GPUService()

    @staticmethod
    def create_with_config(config: dict[str, Any]) -> GPUService:
        """Create a GPU gRPC service with configuration."""
        service = GPUService()
        # Apply configuration if needed
        return service


# Global GPU gRPC service instance
_gpu_grpc_service: GPUService | None = None


def get_gpu_grpc_service() -> GPUService:
    """Get the global GPU gRPC service instance."""
    global _gpu_grpc_service

    if _gpu_grpc_service is None:
        _gpu_grpc_service = GPUServiceFactory.create()

    return _gpu_grpc_service


def create_gpu_grpc_service() -> GPUService:
    """Create a new GPU gRPC service instance."""
    return GPUServiceFactory.create()
