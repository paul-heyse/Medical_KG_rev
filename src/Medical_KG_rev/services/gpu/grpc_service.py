"""gRPC service implementation for GPU operations."""

from __future__ import annotations

import logging

import grpc
from google.protobuf import empty_pb2

from Medical_KG_rev.services.gpu.manager import GpuDevice, GpuNotAvailableError, GpuServiceManager

# Import generated protobuf classes
try:
    from proto import gpu_pb2, gpu_pb2_grpc
except ImportError:
    # Fallback for when protobuf files aren't generated yet
    gpu_pb2 = None
    gpu_pb2_grpc = None

logger = logging.getLogger(__name__)


class GPUServiceServicer(gpu_pb2_grpc.GPUServiceServicer):
    """gRPC servicer for GPU operations."""

    def __init__(self, gpu_manager: GpuServiceManager | None = None):
        self.gpu_manager = gpu_manager or GpuServiceManager()
        self._device_cache: dict[int, GpuDevice] = {}

    def GetStatus(
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> gpu_pb2.StatusResponse:
        """Get GPU service status."""
        try:
            device = self.gpu_manager.get_device()
            return gpu_pb2.StatusResponse(
                available=True,
                device_count=1,  # TODO: Get actual device count
                device_names=[device.name],
            )
        except GpuNotAvailableError as e:
            logger.warning("GPU not available: %s", e)
            return gpu_pb2.StatusResponse(
                available=False,
                device_count=0,
                error_message=str(e),
            )
        except Exception as e:
            logger.error("Error getting GPU status: %s", e)
            return gpu_pb2.StatusResponse(
                available=False,
                device_count=0,
                error_message=f"Internal error: {e}",
            )

    def ListDevices(
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> gpu_pb2.ListDevicesResponse:
        """List available GPU devices."""
        try:
            device = self.gpu_manager.get_device()
            devices = [
                gpu_pb2.GpuDevice(
                    index=device.index,
                    name=device.name,
                    total_memory_mb=device.total_memory_mb,
                    free_memory_mb=self.gpu_manager.available_memory_mb(device),
                    utilization_percent=0.0,  # TODO: Get actual utilization
                    available=True,
                )
            ]
            return gpu_pb2.ListDevicesResponse(devices=devices)
        except GpuNotAvailableError as e:
            logger.warning("No GPU devices available: %s", e)
            return gpu_pb2.ListDevicesResponse(devices=[])
        except Exception as e:
            logger.error("Error listing GPU devices: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return gpu_pb2.ListDevicesResponse(devices=[])

    def AllocateGPU(
        self, request: gpu_pb2.AllocationRequest, context: grpc.ServicerContext
    ) -> gpu_pb2.AllocationResponse:
        """Allocate GPU resources for a service."""
        try:
            # Check if GPU is available
            device = self.gpu_manager.get_device()

            # Check memory requirements
            if request.required_total_memory_mb > 0:
                self.gpu_manager.assert_total_memory(request.required_total_memory_mb)

            if request.required_memory_mb > 0:
                self.gpu_manager.assert_available_memory(request.required_memory_mb, device=device)

            return gpu_pb2.AllocationResponse(
                success=True,
                device_index=device.index,
                device_name=device.name,
                allocated_memory_mb=request.required_memory_mb,
            )
        except GpuNotAvailableError as e:
            logger.warning("GPU allocation failed: %s", e)
            return gpu_pb2.AllocationResponse(
                success=False,
                error_message=str(e),
            )
        except Exception as e:
            logger.error("Error allocating GPU: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return gpu_pb2.AllocationResponse(
                success=False,
                error_message=f"Internal error: {e}",
            )

    def GetDeviceInfo(
        self, request: gpu_pb2.DeviceInfoRequest, context: grpc.ServicerContext
    ) -> gpu_pb2.DeviceInfoResponse:
        """Get information about a specific GPU device."""
        try:
            device = self.gpu_manager.get_device()
            if device.index == request.device_index:
                gpu_device = gpu_pb2.GpuDevice(
                    index=device.index,
                    name=device.name,
                    total_memory_mb=device.total_memory_mb,
                    free_memory_mb=self.gpu_manager.available_memory_mb(device),
                    utilization_percent=0.0,  # TODO: Get actual utilization
                    available=True,
                )
                return gpu_pb2.DeviceInfoResponse(device=gpu_device, found=True)
            else:
                return gpu_pb2.DeviceInfoResponse(found=False)
        except GpuNotAvailableError as e:
            logger.warning("GPU device not available: %s", e)
            return gpu_pb2.DeviceInfoResponse(found=False)
        except Exception as e:
            logger.error("Error getting device info: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return gpu_pb2.DeviceInfoResponse(found=False)

    def GetMemoryInfo(
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> gpu_pb2.MemoryInfoResponse:
        """Get GPU memory information."""
        try:
            device = self.gpu_manager.get_device()
            free_memory = self.gpu_manager.available_memory_mb(device)
            allocated_memory = device.total_memory_mb - free_memory

            gpu_device = gpu_pb2.GpuDevice(
                index=device.index,
                name=device.name,
                total_memory_mb=device.total_memory_mb,
                free_memory_mb=free_memory,
                utilization_percent=0.0,  # TODO: Get actual utilization
                available=True,
            )

            return gpu_pb2.MemoryInfoResponse(
                total_memory_mb=device.total_memory_mb,
                free_memory_mb=free_memory,
                allocated_memory_mb=allocated_memory,
                devices=[gpu_device],
            )
        except GpuNotAvailableError as e:
            logger.warning("GPU memory info not available: %s", e)
            return gpu_pb2.MemoryInfoResponse(
                total_memory_mb=0,
                free_memory_mb=0,
                allocated_memory_mb=0,
                devices=[],
            )
        except Exception as e:
            logger.error("Error getting memory info: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return gpu_pb2.MemoryInfoResponse(
                total_memory_mb=0,
                free_memory_mb=0,
                allocated_memory_mb=0,
                devices=[],
            )
