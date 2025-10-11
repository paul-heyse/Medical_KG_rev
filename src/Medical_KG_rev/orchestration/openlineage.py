"""OpenLineage integration for orchestration pipeline."""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

from Medical_KG_rev.orchestration.ledger import JobLedgerEntry
from Medical_KG_rev.utils.logging import get_logger

logger = get_logger(__name__)

# Optional OpenLineage client
try:
    from openlineage.client import OpenLineageClient  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenLineageClient = None  # type: ignore


class RunState(str, Enum):
    """OpenLineage run state."""

    START = "START"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ABORT = "ABORT"
    FAIL = "FAIL"


class OpenLineageEmitter:
    """OpenLineage event emitter for orchestration pipeline."""

    def __init__(self, client: OpenLineageClient | None = None) -> None:
        """Initialize the OpenLineage emitter."""
        self.client = client
        self.logger = logger
        self.enabled = client is not None

    def emit_run_start(self, run_id: str, job_name: str, metadata: dict[str, Any] | None = None) -> None:
        """Emit run start event."""
        if not self.enabled:
            self.logger.debug("OpenLineage not enabled, skipping run start event")
            return

        try:
            # Create run start event
            event = self._create_run_event(
                run_id=run_id,
                job_name=job_name,
                state=RunState.START,
                metadata=metadata or {},
            )

            # Emit event
            self.client.emit(event)
            self.logger.info(f"Emitted OpenLineage run start event: {run_id}")

        except Exception as exc:
            self.logger.error(f"Failed to emit run start event: {exc}")

    def emit_run_complete(self, run_id: str, job_name: str, metadata: dict[str, Any] | None = None) -> None:
        """Emit run complete event."""
        if not self.enabled:
            self.logger.debug("OpenLineage not enabled, skipping run complete event")
            return

        try:
            # Create run complete event
            event = self._create_run_event(
                run_id=run_id,
                job_name=job_name,
                state=RunState.COMPLETE,
                metadata=metadata or {},
            )

            # Emit event
            self.client.emit(event)
            self.logger.info(f"Emitted OpenLineage run complete event: {run_id}")

        except Exception as exc:
            self.logger.error(f"Failed to emit run complete event: {exc}")

    def emit_run_fail(self, run_id: str, job_name: str, error: str, metadata: dict[str, Any] | None = None) -> None:
        """Emit run fail event."""
        if not self.enabled:
            self.logger.debug("OpenLineage not enabled, skipping run fail event")
            return

        try:
            # Create run fail event
            event = self._create_run_event(
                run_id=run_id,
                job_name=job_name,
                state=RunState.FAIL,
                metadata={
                    **(metadata or {}),
                    "error": error,
                },
            )

            # Emit event
            self.client.emit(event)
            self.logger.info(f"Emitted OpenLineage run fail event: {run_id}")

        except Exception as exc:
            self.logger.error(f"Failed to emit run fail event: {exc}")

    def emit_dataset_event(self, dataset_name: str, event_type: str, metadata: dict[str, Any] | None = None) -> None:
        """Emit dataset event."""
        if not self.enabled:
            self.logger.debug("OpenLineage not enabled, skipping dataset event")
            return

        try:
            # Create dataset event
            event = self._create_dataset_event(
                dataset_name=dataset_name,
                event_type=event_type,
                metadata=metadata or {},
            )

            # Emit event
            self.client.emit(event)
            self.logger.info(f"Emitted OpenLineage dataset event: {dataset_name}")

        except Exception as exc:
            self.logger.error(f"Failed to emit dataset event: {exc}")

    def _create_run_event(self, run_id: str, job_name: str, state: RunState, metadata: dict[str, Any]) -> Any:
        """Create OpenLineage run event."""
        # Mock implementation since we don't have the actual OpenLineage client
        return {
            "eventType": "RUN_STATE_CHANGE",
            "run": {
                "runId": run_id,
                "job": {"name": job_name},
                "state": state.value,
            },
            "metadata": metadata,
        }

    def _create_dataset_event(self, dataset_name: str, event_type: str, metadata: dict[str, Any]) -> Any:
        """Create OpenLineage dataset event."""
        # Mock implementation
        return {
            "eventType": "DATASET_EVENT",
            "dataset": {
                "name": dataset_name,
                "namespace": "medical_kg",
            },
            "eventType": event_type,
            "metadata": metadata,
        }


class OpenLineageIntegration:
    """OpenLineage integration for orchestration pipeline."""

    def __init__(self, emitter: OpenLineageEmitter | None = None) -> None:
        """Initialize the OpenLineage integration."""
        self.emitter = emitter or OpenLineageEmitter()
        self.logger = logger

    def track_job_start(self, job_entry: JobLedgerEntry) -> None:
        """Track job start."""
        try:
            self.emitter.emit_run_start(
                run_id=job_entry.job_id,
                job_name=job_entry.job_type,
                metadata={
                    "tenant_id": job_entry.tenant_id,
                    "start_time": job_entry.start_time,
                    "parameters": job_entry.parameters,
                },
            )
        except Exception as exc:
            self.logger.error(f"Failed to track job start: {exc}")

    def track_job_complete(self, job_entry: JobLedgerEntry) -> None:
        """Track job completion."""
        try:
            self.emitter.emit_run_complete(
                run_id=job_entry.job_id,
                job_name=job_entry.job_type,
                metadata={
                    "tenant_id": job_entry.tenant_id,
                    "end_time": job_entry.end_time,
                    "duration": job_entry.duration,
                    "status": job_entry.status,
                },
            )
        except Exception as exc:
            self.logger.error(f"Failed to track job completion: {exc}")

    def track_job_failure(self, job_entry: JobLedgerEntry, error: str) -> None:
        """Track job failure."""
        try:
            self.emitter.emit_run_fail(
                run_id=job_entry.job_id,
                job_name=job_entry.job_type,
                error=error,
                metadata={
                    "tenant_id": job_entry.tenant_id,
                    "end_time": job_entry.end_time,
                    "duration": job_entry.duration,
                    "status": job_entry.status,
                },
            )
        except Exception as exc:
            self.logger.error(f"Failed to track job failure: {exc}")

    def track_dataset_access(self, dataset_name: str, operation: str, metadata: dict[str, Any] | None = None) -> None:
        """Track dataset access."""
        try:
            self.emitter.emit_dataset_event(
                dataset_name=dataset_name,
                event_type=operation,
                metadata=metadata or {},
            )
        except Exception as exc:
            self.logger.error(f"Failed to track dataset access: {exc}")

    def is_enabled(self) -> bool:
        """Check if OpenLineage is enabled."""
        return self.emitter.enabled


def create_openlineage_client() -> OpenLineageClient | None:
    """Create OpenLineage client if available."""
    if OpenLineageClient is None:
        return None

    try:
        # Get configuration from environment
        endpoint = os.getenv("OPENLINEAGE_ENDPOINT")
        if not endpoint:
            logger.warning("OpenLineage endpoint not configured")
            return None

        # Create client
        client = OpenLineageClient(endpoint)
        logger.info("OpenLineage client created successfully")
        return client

    except Exception as exc:
        logger.error(f"Failed to create OpenLineage client: {exc}")
        return None


def create_openlineage_emitter() -> OpenLineageEmitter:
    """Create OpenLineage emitter."""
    client = create_openlineage_client()
    return OpenLineageEmitter(client)


def create_openlineage_integration() -> OpenLineageIntegration:
    """Create OpenLineage integration."""
    emitter = create_openlineage_emitter()
    return OpenLineageIntegration(emitter)
