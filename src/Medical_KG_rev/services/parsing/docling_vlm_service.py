"""Docling Gemma3 12B VLM integration service."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import structlog

from Medical_KG_rev.config.docling_config import DoclingVLMConfig
from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.services import GpuManager, GpuNotAvailableError

from .exceptions import DoclingModelLoadError, DoclingProcessingError, DoclingVLMError
from .metrics import (
    DOCLING_GPU_MEMORY_MB,
    DOCLING_PROCESSING_SECONDS,
    DOCLING_RETRIES_TOTAL,
)

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class DoclingVLMResult:
    """Structured result returned by the Docling VLM pipeline."""

    document_id: str
    text: str
    tables: list[dict[str, Any]]
    figures: list[dict[str, Any]]
    metadata: dict[str, Any]


class DoclingVLMService:
    """High-level interface around Docling's Gemma3 12B VLM pipeline."""

    def __init__(
        self,
        config: DoclingVLMConfig | None = None,
        gpu_manager: GpuManager | None = None,
        *,
        eager: bool = False,
    ) -> None:
        self._config = config or get_settings().docling_vlm.as_config()
        self._config.ensure_model_path()
        self._gpu = gpu_manager or GpuManager(min_memory_mb=self._config.required_total_memory_mb)
        self._pipeline: Any | None = None
        self._lock = threading.Lock()
        if eager:
            self._ensure_pipeline(warmup=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_transformers_pipeline(self) -> Any:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            raise DoclingModelLoadError("transformers is required for Docling VLM usage") from exc

        model_dir = str(self._config.model_path)
        kwargs = {
            "model": self._config.model_name,
            "model_kwargs": {
                "cache_dir": model_dir,
                "revision": None,
                "trust_remote_code": True,
            },
            "device_map": "auto",
        }
        logger.info(
            "docling_vlm.loading_pipeline",
            model=self._config.model_name,
            cache_dir=model_dir,
        )
        try:
            return pipeline("document-question-answering", **kwargs)
        except Exception as exc:  # pragma: no cover - heavy dependency load
            raise DoclingModelLoadError("Failed to load Gemma3 VLM pipeline") from exc

    def _ensure_pipeline(self, *, warmup: bool = False) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is None:
                self._pipeline = self._load_transformers_pipeline()
                if warmup:
                    self._run_warmup()
        return self._pipeline

    def _run_warmup(self) -> None:
        prompts = [
            {
                "question": "Summarise the document",
                "context": "Docling warmup",
            }
            for _ in range(max(self._config.warmup_prompts, 1))
        ]
        try:
            with self._gpu.device_session(
                "docling_vlm_warmup",
                required_total_memory_mb=self._config.required_total_memory_mb,
            ):
                pipeline = self._ensure_pipeline()
                for prompt in prompts:
                    pipeline(prompt["question"], prompt["context"])
        except Exception as exc:  # pragma: no cover - warmup best effort
            logger.warning("docling_vlm.warmup_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_pdf(self, pdf_path: str, *, document_id: str) -> DoclingVLMResult:
        start = time.perf_counter()
        pipeline = self._ensure_pipeline()
        attempts = max(self._config.retry_attempts, 0) + 1
        delay = 1.0
        last_exc: Exception | None = None
        requested_memory_mb = int(
            self._config.gpu_memory_fraction * self._config.required_total_memory_mb
        )
        for attempt in range(attempts):
            try:
                with self._gpu.device_session(
                    "docling_vlm",
                    required_memory_mb=requested_memory_mb,
                    required_total_memory_mb=self._config.required_total_memory_mb,
                ):
                    outputs = pipeline(
                        question="Extract structured content",
                        document=pdf_path,
                        top_k=1,
                        max_len=self._config.max_model_len,
                    )
                break
            except GpuNotAvailableError:
                raise
            except Exception as exc:  # pragma: no cover - depends on docling runtime
                last_exc = exc
                if attempt == attempts - 1:
                    DOCLING_PROCESSING_SECONDS.labels(status="error").observe(
                        time.perf_counter() - start
                    )
                    raise DoclingProcessingError(str(exc)) from exc
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                logger.warning(
                    "docling_vlm.retry", attempt=attempt + 1, max_attempts=attempts, error=str(exc)
                )
                DOCLING_RETRIES_TOTAL.inc()
        else:  # pragma: no cover - defensive
            raise DoclingProcessingError(str(last_exc))

        duration = time.perf_counter() - start
        DOCLING_GPU_MEMORY_MB.observe(requested_memory_mb)
        DOCLING_PROCESSING_SECONDS.labels(status="ok").observe(duration)
        return self._map_outputs(document_id=document_id, outputs=outputs, duration=duration)

    def process_pdf_batch(self, items: Sequence[tuple[str, str]]) -> list[DoclingVLMResult]:
        results: list[DoclingVLMResult] = []
        failures: list[tuple[str, Exception]] = []
        for document_id, path in items:
            try:
                results.append(self.process_pdf(path, document_id=document_id))
            except DoclingVLMError as exc:
                failures.append((document_id, exc))
        if failures:
            summary = "; ".join(f"{doc_id}: {exc}" for doc_id, exc in failures)
            logger.error("docling_vlm.batch_failures", failures=summary)
        return results

    def health(self) -> dict[str, Any]:
        try:
            self._gpu.assert_total_memory(self._config.required_total_memory_mb)
        except GpuNotAvailableError as exc:
            return {"status": "error", "detail": str(exc)}
        cache_exists = Path(self._config.model_path).exists()
        pipeline_ready = self._pipeline is not None
        return {
            "status": "ok" if cache_exists else "degraded",
            "model_path": str(self._config.model_path),
            "pipeline_ready": pipeline_ready,
            "cache_exists": cache_exists,
        }

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------
    def _map_outputs(
        self,
        *,
        document_id: str,
        outputs: Any,
        duration: float,
    ) -> DoclingVLMResult:
        data = self._normalise_output(outputs)
        metadata = data.get("metadata", {})
        provenance = {
            "model_name": self._config.model_name,
            "processing_time_seconds": round(duration, 3),
            "gpu_memory_fraction": self._config.gpu_memory_fraction,
        }
        metadata.setdefault("provenance", provenance)
        return DoclingVLMResult(
            document_id=document_id,
            text=str(data.get("text", "")),
            tables=list(data.get("tables", [])),
            figures=list(data.get("figures", [])),
            metadata=metadata,
        )

    def _normalise_output(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, Mapping):
            return dict(payload)
        if isinstance(payload, list) and payload:
            candidate = payload[0]
            if isinstance(candidate, Mapping):
                return dict(candidate)
            if hasattr(candidate, "model_dump"):
                return candidate.model_dump()
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        try:
            return json.loads(json.dumps(payload))
        except Exception:  # pragma: no cover - fallback path
            return {
                "text": "",
                "tables": [],
                "figures": [],
                "metadata": {},
            }
