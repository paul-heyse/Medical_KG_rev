"""State persistence helpers with retry semantics."""

from __future__ import annotations

from typing import Any, Protocol

import structlog
from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential

from .serialization import dumps_orjson, encode_base64

logger = structlog.get_logger(__name__)


class SupportsMetadataUpdate(Protocol):
    def update_metadata(self, job_id: str, payload: dict[str, Any]) -> None: ...


class StatePersistenceError(RuntimeError):
    """Raised when the pipeline state snapshot cannot be persisted."""


class PipelineStatePersister:
    """Persist pipeline state snapshots to a metadata store with retries."""

    def __init__(
        self,
        *,
        metadata_store: SupportsMetadataUpdate,
        max_attempts: int = 3,
        wait_initial: float = 0.5,
        wait_max: float = 5.0,
    ) -> None:
        self._metadata_store = metadata_store
        self._retry = Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=wait_initial, max=wait_max),
            reraise=True,
        )

    def persist(self, job_id: str, *, stage: str, payload: dict[str, Any]) -> str:
        """Persist the serialised payload and return the encoded snapshot."""

        snapshot = encode_base64(dumps_orjson(payload))

        def _attempt() -> None:
            self._metadata_store.update_metadata(
                job_id,
                {
                    f"state.{stage}.snapshot": snapshot,
                },
            )

        try:
            self._retry(_attempt)
        except RetryError as exc:  # pragma: no cover - defensive
            logger.error(
                "pipeline.state.persistence_failed",
                job_id=job_id,
                stage=stage,
                attempts=exc.last_attempt.attempt_number if exc.last_attempt else 0,
                error=str(exc.last_attempt.exception() if exc.last_attempt else exc),
            )
            raise StatePersistenceError("Failed to persist pipeline state") from exc

        logger.debug(
            "pipeline.state.persisted",
            job_id=job_id,
            stage=stage,
        )
        return snapshot

    def persist_state(self, job_id: str, *, stage: str, state: Any) -> str:
        """Helper to persist using a PipelineState instance."""

        payload = getattr(state, "serialise", lambda: state)()
        return self.persist(job_id, stage=stage, payload=payload)
