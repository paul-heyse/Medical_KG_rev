"""Unit tests for the DoclingVLMService."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from Medical_KG_rev.config.docling_config import DoclingVLMConfig
from Medical_KG_rev.services import GpuNotAvailableError
from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMResult, DoclingVLMService
from Medical_KG_rev.services.parsing.exceptions import (
    DoclingModelUnavailableError,
    DoclingOutOfMemoryError,
    DoclingProcessingTimeoutError,
)


class DummyGpu:
    """Test double implementing the GPU manager interface."""

    def __init__(self, *, should_fail: Exception | None = None) -> None:
        self._error = should_fail

    @contextlib.contextmanager
    def device_session(self, *_, **__):
        if self._error:
            raise self._error
        yield None

    def assert_total_memory(self, *_: object) -> None:
        if isinstance(self._error, GpuNotAvailableError):
            raise self._error


@pytest.fixture()
def docling_config(tmp_path: Path) -> DoclingVLMConfig:
    config = DoclingVLMConfig(
        model_path=tmp_path,
        model_name="test/docling",
        batch_size=1,
        retry_attempts=0,
        warmup_prompts=0,
        required_total_memory_mb=1024,
    )
    return config


def _build_service(config: DoclingVLMConfig, gpu_error: Exception | None = None) -> DoclingVLMService:
    service = DoclingVLMService(config=config, gpu_manager=DummyGpu(should_fail=gpu_error))
    fake_pipeline = MagicMock(
        side_effect=lambda **_: {
            "text": "docling text",
            "tables": ["table-1"],
            "figures": [],
            "metadata": {"provenance": {"model_name": config.model_name}},
        }
    )
    service._ensure_pipeline = MagicMock(return_value=fake_pipeline)
    return service


def test_process_pdf_success_returns_normalised_result(docling_config: DoclingVLMConfig) -> None:
    service = _build_service(docling_config)
    result = service.process_pdf("/tmp/test.pdf", document_id="doc-1")
    assert isinstance(result, DoclingVLMResult)
    assert result.document_id == "doc-1"
    assert result.text == "docling text"
    assert result.tables == ["table-1"]
    assert "provenance" in result.metadata


def test_process_pdf_raises_model_unavailable_on_gpu_failure(docling_config: DoclingVLMConfig) -> None:
    service = _build_service(docling_config, gpu_error=GpuNotAvailableError("no gpu"))
    with pytest.raises(DoclingModelUnavailableError):
        service.process_pdf("/tmp/failure.pdf", document_id="doc-2")


def test_process_pdf_raises_out_of_memory_on_runtime_error(docling_config: DoclingVLMConfig) -> None:
    service = _build_service(docling_config)
    failing_pipeline = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
    service._ensure_pipeline = MagicMock(return_value=failing_pipeline)
    with pytest.raises(DoclingOutOfMemoryError):
        service.process_pdf("/tmp/failure.pdf", document_id="doc-3")


def test_process_pdf_raises_timeout_when_pipeline_times_out(docling_config: DoclingVLMConfig) -> None:
    service = _build_service(docling_config)
    timeout_pipeline = MagicMock(side_effect=TimeoutError("deadline"))
    service._ensure_pipeline = MagicMock(return_value=timeout_pipeline)
    with pytest.raises(DoclingProcessingTimeoutError):
        service.process_pdf("/tmp/timeout.pdf", document_id="doc-4")


def test_process_pdf_batch_handles_partial_failures(docling_config: DoclingVLMConfig) -> None:
    service = _build_service(docling_config)
    service.process_pdf = MagicMock(side_effect=[
        DoclingVLMResult("doc-1", "text", [], [], {}),
        DoclingVLMResult("doc-2", "text", [], [], {}),
    ])
    results = service.process_pdf_batch([("doc-1", "/tmp/1.pdf"), ("doc-2", "/tmp/2.pdf")])
    assert len(results) == 2
    assert [r.document_id for r in results] == ["doc-1", "doc-2"]
