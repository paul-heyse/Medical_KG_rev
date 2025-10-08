import types

import pytest

from Medical_KG_rev.embeddings.utils import gpu
from Medical_KG_rev.services import GpuNotAvailableError


def test_memory_info_unavailable_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "torch", None)
    info = gpu.memory_info()
    assert info.available is False


def test_ensure_memory_budget_noop_when_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_info():
        return gpu.GPUMemoryInfo(available=True, total_mb=16000, free_mb=8000, used_mb=8000)

    monkeypatch.setattr(gpu, "memory_info", fake_info)
    gpu.ensure_memory_budget(True, operation="embed", fraction=None, reserve_mb=None)


def test_ensure_memory_budget_fraction_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_info():
        return gpu.GPUMemoryInfo(available=True, total_mb=1000, free_mb=100, used_mb=900)

    monkeypatch.setattr(gpu, "logger", types.SimpleNamespace(warning=lambda *args, **kwargs: None))
    monkeypatch.setattr(gpu, "memory_info", fake_info)
    with pytest.raises(GpuNotAvailableError):
        gpu.ensure_memory_budget(True, operation="embed", fraction=0.5)


def test_ensure_memory_budget_reserve_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_info():
        return gpu.GPUMemoryInfo(available=True, total_mb=1000, free_mb=50, used_mb=950)

    monkeypatch.setattr(gpu, "logger", types.SimpleNamespace(warning=lambda *args, **kwargs: None))
    monkeypatch.setattr(gpu, "memory_info", fake_info)
    with pytest.raises(GpuNotAvailableError):
        gpu.ensure_memory_budget(True, operation="embed", reserve_mb=100)


def test_memory_info_populates_values(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda device=0: (500 * 1024 * 1024, 1000 * 1024 * 1024),
        device_count=lambda: 1,
        get_device_name=lambda index: "Fake GPU",
    )
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr(gpu, "torch", fake_torch)
    info = gpu.memory_info()
    assert info.available is True
    assert info.total_mb == 1000
    assert info.free_mb == 500
    assert info.used_mb == 500
