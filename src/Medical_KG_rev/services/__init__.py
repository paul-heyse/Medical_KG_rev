"""Service layer utilities and shared functionality.

Key Responsibilities:
    - Provide lazy loading of service components to avoid optional dependency issues
    - Export commonly used service utilities and types
    - Manage GPU service integration and fallbacks
    - Offer shared service health monitoring and metrics

Collaborators:
    - Upstream: Gateway services and orchestration layers use these utilities
    - Downstream: Individual service modules (embedding, reranking, retrieval, etc.)

Side Effects:
    - May trigger module imports and GPU service initialization
    - Emits structured logs during service discovery

Thread Safety:
    - Thread-safe: Lazy loading is protected by import guards

Performance Characteristics:
    - Lazy loading reduces startup time when GPU services are unavailable
    - Module imports are cached to avoid repeated discovery overhead

Example:
    >>> from Medical_KG_rev.services import GPU_MEMORY_USED
    >>> print(f"GPU memory metric available: {GPU_MEMORY_USED is not None}")
    True
"""

from __future__ import annotations

import importlib.util
from typing import Any

from .gpu.manager import GpuNotAvailableError  # type: ignore
from .gpu.metrics import GPU_MEMORY_USED, GPU_UTILIZATION  # type: ignore

__all__ = ["GpuNotAvailableError", "GPU_MEMORY_USED", "GPU_UTILIZATION"]

try:
    from .gpu.manager import GpuServiceManager  # type: ignore
    __all__.extend(["GpuServiceManager"])
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "Medical_KG_rev.services.gpu requires the torch-based GPU service package"
    ) from exc
