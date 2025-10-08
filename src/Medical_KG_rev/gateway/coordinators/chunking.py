"""Chunking coordinator implementation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from Medical_KG_rev.chunking.exceptions import (
    ChunkerConfigurationError,
    ChunkingFailedError,
    ChunkingUnavailableError,
    InvalidDocumentError,
    MineruGpuUnavailableError,
    MineruOutOfMemoryError,
    ProfileNotFoundError,
    TokenizerMismatchError,
)
from Medical_KG_rev.gateway.models import DocumentChunk, ProblemDetail
from Medical_KG_rev.observability.metrics import record_chunking_failure
from Medical_KG_rev.services.retrieval.chunking import ChunkingOptions, ChunkingService
from Medical_KG_rev.gateway.models import DocumentChunk
from Medical_KG_rev.observability.metrics import record_chunking_failure
from Medical_KG_rev.services.retrieval.chunking import ChunkCommand, ChunkingService

from .base import (
    BaseCoordinator,
    CoordinatorConfig,
    CoordinatorError,
    CoordinatorRequest,
    CoordinatorResult,
)
from .job_lifecycle import JobLifecycleManager
from ..chunking_errors import ChunkingErrorReport, ChunkingErrorTranslator


@dataclass
class ChunkingRequest(CoordinatorRequest):
    document_id: str
    text: str | None = None
    strategy: str | None = None
    chunk_size: int | None = None
    overlap: int | None = None
    options: Mapping[str, Any] | None = None


@dataclass
class ChunkingResult(CoordinatorResult):
    chunks: Sequence[DocumentChunk] = ()


class ChunkingCoordinator(BaseCoordinator[ChunkingRequest, ChunkingResult]):
    """Coordinate synchronous chunking jobs."""

    def __init__(
        self,
        lifecycle: JobLifecycleManager,
        chunker: ChunkingService,
        config: CoordinatorConfig,
        *,
        errors: ChunkingErrorTranslator | None = None,
    ) -> None:
        super().__init__(config=config, metrics=self._metrics(config))
        self._lifecycle = lifecycle
        self._chunker = chunker
        strategies = chunker.available_strategies()
        self._errors = errors or ChunkingErrorTranslator(strategies=strategies)

    @staticmethod
    def _metrics(config: CoordinatorConfig):
        from .base import CoordinatorMetrics

        return CoordinatorMetrics.create(config.name)

    def _execute(self, request: ChunkingRequest, /, **_: Any) -> ChunkingResult:
        job_id = self._lifecycle.create_job(request.tenant_id, "chunk")
        text = self._extract_text(job_id, request)
        command = ChunkCommand(
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            text=text,
            strategy=request.strategy or "section",
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            options=dict(request.options or {}),
        metadata = dict(self._metadata_without_text(request.options))
        options = ChunkingOptions(
            strategy=request.strategy,
            max_tokens=request.chunk_size,
            overlap=request.overlap,
            metadata=metadata,
        )

        started = time.perf_counter()
        try:
            raw_chunks = self._chunker.chunk(command)
        except ProfileNotFoundError as exc:
            raise self._translate_error(job_id, command, exc)
        except TokenizerMismatchError as exc:
            raise self._translate_error(job_id, command, exc)
        except ChunkingFailedError as exc:
            raise self._translate_error(job_id, command, exc)
        except InvalidDocumentError as exc:
            raise self._translate_error(job_id, command, exc)
        except ChunkerConfigurationError as exc:
            raise self._translate_error(job_id, command, exc)
        except ChunkingUnavailableError as exc:
            raise self._translate_error(job_id, command, exc)
        except MineruOutOfMemoryError as exc:
            raise self._translate_error(job_id, command, exc)
        except MineruGpuUnavailableError as exc:
            raise self._translate_error(job_id, command, exc)
        except MemoryError as exc:
            raise self._translate_error(job_id, command, exc)
        except TimeoutError as exc:
            raise self._translate_error(job_id, command, exc)
        except RuntimeError as exc:
            message = str(exc)
            if "GPU semantic checks" in message:
                raise self._translate_error(job_id, command, exc)
            raw_chunks = self._chunker.chunk(
                request.tenant_id,
                request.document_id,
                text,
                options,
            )
        except ProfileNotFoundError as exc:
            raise self._error(job_id, request, "Chunking profile not found", 400, exc, "ProfileNotFoundError")
        except TokenizerMismatchError as exc:
            raise self._error(job_id, request, "Tokenizer mismatch", 500, exc, "TokenizerMismatchError")
        except ChunkingFailedError as exc:
            message = exc.detail or str(exc) or "Chunking process failed"
            raise self._error(job_id, request, message, 500, exc, "ChunkingFailedError")
        except InvalidDocumentError as exc:
            raise self._error(job_id, request, "Invalid document payload", 400, exc, "InvalidDocumentError")
        except ChunkerConfigurationError as exc:
            detail = ProblemDetail(
                title="Chunker configuration invalid",
                status=422,
                type="https://httpstatuses.com/422",
                detail=str(exc),
                extensions={"valid_strategies": self._chunker.available_strategies()},
            )
            self._record_failure(job_id, request, detail, exc)
            raise CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id}) from exc
        except ChunkingUnavailableError as exc:
            retry_after = max(1, int(round(exc.retry_after)))
            detail = ProblemDetail(
                title="Chunking temporarily unavailable",
                status=503,
                type="https://httpstatuses.com/503",
                detail=str(exc),
                extensions={"retry_after": retry_after},
            )
            self._record_failure(job_id, request, detail, exc)
            raise CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id}) from exc
        except MineruOutOfMemoryError as exc:
            detail = ProblemDetail(
                title="MinerU out of memory",
                status=503,
                type="https://medical-kg/errors/mineru-oom",
                detail=str(exc),
                extensions={"reason": "gpu_out_of_memory"},
            )
            self._record_failure(job_id, request, detail, exc)
            raise CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id}) from exc
        except MineruGpuUnavailableError as exc:
            detail = ProblemDetail(
                title="MinerU GPU unavailable",
                status=503,
                type="https://medical-kg/errors/mineru-gpu-unavailable",
                detail=str(exc),
                extensions={"reason": "gpu_unavailable"},
            )
            self._record_failure(job_id, request, detail, exc)
            raise CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id}) from exc
        except MemoryError as exc:
            message = str(exc) or "Chunking operation exhausted available memory"
            detail = ProblemDetail(
                title="Chunking resources exhausted",
                status=503,
                type="https://httpstatuses.com/503",
                detail=message,
                extensions={"retry_after": 60},
            )
            self._record_failure(job_id, request, detail, exc)
            raise CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id}) from exc
        except TimeoutError as exc:
            message = str(exc) or "Chunking operation timed out"
            detail = ProblemDetail(
                title="Chunking resources exhausted",
                status=503,
                type="https://httpstatuses.com/503",
                detail=message,
                extensions={"retry_after": 30},
            )
            self._record_failure(job_id, request, detail, exc)
            raise CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id}) from exc
        except RuntimeError as exc:
            message = str(exc)
            if "GPU semantic checks" in message:
                detail = ProblemDetail(
                    title="GPU unavailable for semantic chunking",
                    status=503,
                    type="https://httpstatuses.com/503",
                    detail=message,
                    extensions={"reason": "gpu_unavailable"},
                )
                self._record_failure(job_id, request, detail, exc)
                raise CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id}) from exc
            self._lifecycle.mark_failed(job_id, reason=message or "Runtime error during chunking", stage="chunk")
            raise

        chunks: list[DocumentChunk] = []
        for index, chunk in enumerate(raw_chunks):
            meta = dict(chunk.meta)
            meta.setdefault("granularity", chunk.granularity)
            meta.setdefault("chunker", chunk.chunker)
            chunks.append(
                DocumentChunk(
                    document_id=request.document_id,
                    chunk_index=index,
                    content=chunk.body,
                    metadata=meta,
                    token_count=meta.get("token_count", 0),
                )
            )
        duration = time.perf_counter() - started
        payload = {"chunks": len(chunks), "strategy": command.strategy}
        payload = {"chunks": len(chunks)}
        self._lifecycle.update_metadata(job_id, payload)
        self._lifecycle.mark_completed(job_id, payload=payload)
        return ChunkingResult(
            job_id=job_id,
            duration_s=duration,
            chunks=tuple(chunks),
            metadata=payload,
        )

    def _extract_text(self, job_id: str, request: ChunkingRequest) -> str:
        candidate = request.text
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        payload = request.options or {}
        raw_text = payload.get("text") if isinstance(payload, Mapping) else None
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise InvalidDocumentError(
                "Chunking requests must include a non-empty 'text' field"
            )
        return raw_text

    def _translate_error(
        self,
        job_id: str,
        command: ChunkCommand,
        exc: Exception,
    ) -> CoordinatorError:
        report = self._errors.translate(exc, command=command, job_id=job_id)
        if report is None:
            self._lifecycle.mark_failed(
                job_id,
                reason=str(exc) or exc.__class__.__name__,
                stage="chunk",
            )
            raise exc
        self._record_failure(job_id, command, report)
        return CoordinatorError(
            report.problem.title,
            context={
                "problem": report.problem,
                "job_id": job_id,
                "severity": report.severity,
                "metric": report.metric,
            },
        )
        payload = request.options or {}
        raw_text = payload.get("text") if isinstance(payload, Mapping) else None
        if not isinstance(raw_text, str) or not raw_text.strip():
            detail = ProblemDetail(
                title="Invalid document payload",
                status=400,
                type="https://httpstatuses.com/400",
                detail="Chunking requests must include a non-empty 'text' field in options",
                instance=f"/v1/chunk/{request.document_id}",
            )
            self._lifecycle.mark_failed(job_id, reason=detail.detail or detail.title, stage="chunk")
            raise CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id})
        return raw_text

    @staticmethod
    def _metadata_without_text(options: Mapping[str, Any] | None) -> Mapping[str, Any]:
        if not isinstance(options, Mapping):
            return {}
        return {key: value for key, value in options.items() if key != "text"}

    def _error(
        self,
        job_id: str,
        request: ChunkingRequest,
        title: str,
        status: int,
        exc: Exception,
        metric: str,
    ) -> CoordinatorError:
        detail = ProblemDetail(
            title=title,
            status=status,
            type="https://medical-kg/errors/chunking",
            detail=str(exc),
            instance=f"/v1/chunk/{request.document_id}",
        )
        profile = None
        if isinstance(request.options, Mapping):
            raw_profile = request.options.get("profile")
            if isinstance(raw_profile, str):
                profile = raw_profile
        record_chunking_failure(profile, metric)
        self._lifecycle.mark_failed(job_id, reason=detail.detail or detail.title, stage="chunk")
        return CoordinatorError(detail.title, context={"problem": detail, "job_id": job_id})

    def _record_failure(
        self,
        job_id: str,
        command: ChunkCommand,
        report: ChunkingErrorReport,
    ) -> None:
        profile = command.options.get("profile")
        if isinstance(profile, str) and profile:
            record_chunking_failure(profile, report.metric or "unknown_error")
        else:
            record_chunking_failure(None, report.metric or "unknown_error")
        self._lifecycle.mark_failed(
            job_id,
            reason=report.problem.detail or report.problem.title,
            stage="chunk",
        )
        request: ChunkingRequest,
        detail: ProblemDetail,
        exc: Exception,
    ) -> None:
        profile = None
        if isinstance(request.options, Mapping):
            raw_profile = request.options.get("profile")
            if isinstance(raw_profile, str):
                profile = raw_profile
        record_chunking_failure(profile, exc.__class__.__name__)
        self._lifecycle.mark_failed(job_id, reason=detail.detail or detail.title, stage="chunk")


__all__ = [
    "ChunkingCoordinator",
    "ChunkingRequest",
    "ChunkingResult",
]
