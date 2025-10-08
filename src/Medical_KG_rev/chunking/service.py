"""High level chunking service bridging configuration and chunkers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section

from .configuration import ChunkerSettings, ChunkingConfig, DEFAULT_CONFIG_PATH
from .factory import ChunkerFactory
from .models import Chunk, Granularity
from .exceptions import (
    ChunkerConfigurationError,
    ChunkingFailedError,
    ChunkingUnavailableError,
    InvalidDocumentError,
    TokenizerMismatchError,
)
from .runtime import ChunkingRuntime, ChunkerSession
from Medical_KG_rev.observability.metrics import set_chunking_circuit_state


class _ChunkingCircuitBreaker:
    """Simple circuit breaker tracking repeated chunking failures."""

    def __init__(
        self,
        *,
        failure_threshold: int,
        base_recovery_seconds: float,
        max_recovery_seconds: float,
    ) -> None:
        self.failure_threshold = max(failure_threshold, 1)
        self.base_recovery_seconds = max(base_recovery_seconds, 1.0)
        self.max_recovery_seconds = max(max_recovery_seconds, self.base_recovery_seconds)
        self._state = "closed"
        self._failure_count = 0
        self._opened_at: float | None = None
        self._open_cycles = 0
        self._recovery_window = self.base_recovery_seconds
        self._update_metrics()

    def guard(self) -> None:
        if self._state != "open":
            return
        assert self._opened_at is not None  # for type checkers
        elapsed = perf_counter() - self._opened_at
        if elapsed >= self._recovery_window:
            self._state = "half_open"
            self._update_metrics()
            return
        remaining = max(self._recovery_window - elapsed, 0.0)
        raise ChunkingUnavailableError(remaining)

    def record_success(self) -> None:
        self._failure_count = 0
        self._open_cycles = 0
        self._opened_at = None
        self._recovery_window = self.base_recovery_seconds
        if self._state != "closed":
            self._state = "closed"
            self._update_metrics()
        else:
            self._update_metrics()

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._state == "half_open":
            self._open_again()
            return
        if self._failure_count >= self.failure_threshold:
            self._open_again()

    @property
    def state(self) -> str:
        return self._state

    def _open_again(self) -> None:
        self._state = "open"
        self._opened_at = perf_counter()
        self._open_cycles += 1
        backoff = self.base_recovery_seconds * (2 ** max(self._open_cycles - 1, 0))
        self._recovery_window = min(backoff, self.max_recovery_seconds)
        self._update_metrics()

    def _update_metrics(self) -> None:
        mapping = {"closed": 0, "open": 1, "half_open": 2}
        set_chunking_circuit_state(mapping.get(self._state, 0))

    @contextmanager
    def attempt(
        self,
        *,
        skip_failures: tuple[type[Exception], ...] = (),
    ) -> "Iterator[None]":
        """Guard a chunking attempt and update circuit state automatically."""

        self.guard()
        try:
            yield
        except skip_failures:
            raise
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()


@dataclass(slots=True)
class ChunkingOptions:
    strategy: str | None = None
    granularity: Granularity | None = None
    params: dict[str, object] | None = None
    enable_multi_granularity: bool | None = None
    auxiliaries: Sequence[ChunkerSettings] | None = None


class ChunkingService:
    """Entry point consumed by other services."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        registry_factory: ChunkerFactory | None = None,
        failure_threshold: int = 5,
        base_recovery_seconds: float = 10.0,
        max_recovery_seconds: float = 120.0,
    ) -> None:
        self.config = ChunkingConfig.load(config_path or DEFAULT_CONFIG_PATH)
        self.factory = registry_factory or ChunkerFactory()
        self.runtime = ChunkingRuntime(factory=self.factory)
        self._session_cache: dict[tuple[object, ...], ChunkerSession] = {}
        self._circuit = _ChunkingCircuitBreaker(
            failure_threshold=failure_threshold,
            base_recovery_seconds=base_recovery_seconds,
            max_recovery_seconds=max_recovery_seconds,
        )

    def chunk_document(
        self,
        document: Document,
        *,
        tenant_id: str,
        source: str | None = None,
        options: ChunkingOptions | None = None,
    ) -> list[Chunk]:
        self._validate_document(document)
        profile = self.config.profile_for_source(source)
        expected_tokenizer = str(profile.primary.params.get("tokenizer", ""))
        requested_tokenizer = ""
        embedding_model = str(profile.primary.params.get("embedding_model", ""))
        if options and options.params:
            requested_tokenizer = str(options.params.get("tokenizer", ""))
            if not embedding_model:
                embedding_model = str(options.params.get("embedding_model", ""))
        if requested_tokenizer and expected_tokenizer and requested_tokenizer != expected_tokenizer:
            raise TokenizerMismatchError(requested_tokenizer, embedding_model)
        allow_multi = (
            profile.enable_multi_granularity
            if options is None or options.enable_multi_granularity is None
            else options.enable_multi_granularity
        )
        chunker_settings: list[ChunkerSettings]
        if options and options.strategy:
            primary = ChunkerSettings(
                strategy=options.strategy,
                granularity=options.granularity,
                params=dict(options.params or {}),
            )
            auxiliaries = list(options.auxiliaries or [])
            chunker_settings = [primary, *auxiliaries]
        else:
            chunker_settings = [profile.primary, *profile.auxiliaries]
        skip_failures = (
            ChunkerConfigurationError,
            InvalidDocumentError,
            ChunkingUnavailableError,
        )
        with self._circuit.attempt(skip_failures=skip_failures):
            plan_key = self._plan_key(chunker_settings, allow_multi)
            session = self._session_cache.get(plan_key)
            if session is None:
                session = self.runtime.create_session(
                    chunker_settings,
                    allow_experimental=True,
                    enable_multi_granularity=allow_multi,
                )
                self._session_cache[plan_key] = session
            try:
                chunks = session.chunk(document, tenant_id=tenant_id)
            except (ChunkerConfigurationError, InvalidDocumentError, ChunkingUnavailableError):
                raise
            except Exception as exc:  # pragma: no cover - defensive fallback
                raise ChunkingFailedError("Chunking process failed", detail=str(exc)) from exc
        return chunks

    def chunk_text(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
        *,
        options: ChunkingOptions | None = None,
    ) -> list[Chunk]:
        if not isinstance(text, str) or not text.strip():
            raise InvalidDocumentError("Text payload must be a non-empty string")
        document = self._document_from_text(document_id, text)
        return self.chunk_document(document, tenant_id=tenant_id, source=None, options=options)

    def list_strategies(self) -> list[str]:
        return sorted(self.factory.registry.list_chunkers(include_experimental=True).keys())

    def clear_session_cache(self) -> None:
        self._session_cache.clear()

    def _document_from_text(self, document_id: str, text: str) -> Document:
        block = Block(
            id=f"{document_id}:block:0",
            type=BlockType.PARAGRAPH,
            text=text,
            spans=[],
            metadata={},
        )
        section = Section(id=f"{document_id}:section:0", title="Document", blocks=[block])
        return Document(id=document_id, source="ad-hoc", title="Document", sections=[section])

    def _validate_document(self, document: Document) -> None:
        if not document.sections:
            raise InvalidDocumentError("Document contains no sections to chunk")
        if not any(section.blocks for section in document.sections):
            raise InvalidDocumentError("Document contains no blocks to chunk")

    @classmethod
    def _plan_key(
        cls, settings: Sequence[ChunkerSettings], allow_multi: bool
    ) -> tuple[object, ...]:
        return (
            allow_multi,
            tuple(
                (
                    setting.strategy,
                    setting.granularity,
                    cls._freeze_value(setting.params),
                )
                for setting in settings
            ),
        )

    @classmethod
    def _freeze_value(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return tuple(
                (key, cls._freeze_value(subvalue))
                for key, subvalue in sorted(value.items(), key=lambda item: item[0])
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(cls._freeze_value(item) for item in value))
        return value
