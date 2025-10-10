from __future__ import annotations

from pathlib import Path

from Medical_KG_rev.config.docling_config import DoclingVLMConfig
from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMService


class _GpuStub:
    def __init__(self, required_total: int, available: int) -> None:
        self.required_total = required_total
        self.available = available
        self.requested_total: int | None = None
        self.requested_available: int | None = None

    def assert_total_memory(self, required_mb: int):
        self.requested_total = required_mb
        return type("Device", (), {"index": 0})()

    def assert_available_memory(self, required_free_mb: int, device=None):
        self.requested_available = required_free_mb
        if self.available < required_free_mb:
            from Medical_KG_rev.services import GpuNotAvailableError

            raise GpuNotAvailableError("insufficient")
        return self.available


def test_health_reports_available_memory(tmp_path):
    model_path = tmp_path / "models"
    model_path.mkdir()
    config = DoclingVLMConfig(
        model_path=model_path,
        required_total_memory_mb=10_000,
        gpu_memory_fraction=0.5,
    )
    gpu = _GpuStub(required_total=10_000, available=6000)
    service = DoclingVLMService(config=config, gpu_manager=gpu, eager=False)

    health = service.health()

    assert gpu.requested_total == 10_000
    assert gpu.requested_available == 5000
    assert health["status"] == "ok"
    assert health["available_memory_mb"] == 6000
    assert health["cache_exists"] is True
