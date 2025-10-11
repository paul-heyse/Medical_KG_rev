"""Dagster runtime management and execution."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from dagster import (
    AssetExecutionContext,
    Config,
    Definitions,
    RunRequest,
    SkipReason,
    asset,
    job,
    op,
    schedule,
    sensor,
)

from Medical_KG_rev.adapters.plugins.bootstrap import get_plugin_manager
from Medical_KG_rev.adapters.plugins.manager import AdapterPluginManager
from Medical_KG_rev.adapters.plugins.models import AdapterRequest
from Medical_KG_rev.config.settings import get_settings
from Medical_KG_rev.orchestration.dagster.configuration import (
    DagsterConfig,
    DagsterConfigurationManager,
)
from Medical_KG_rev.orchestration.dagster.stages import (
    create_default_pipeline_resource,
)
from Medical_KG_rev.orchestration.dagster.types import PIPELINE_STATE_DAGSTER_TYPE
from Medical_KG_rev.orchestration.events import StageEventEmitter
from Medical_KG_rev.orchestration.kafka import KafkaClient
from Medical_KG_rev.orchestration.ledger import JobLedger, JobLedgerError

logger = logging.getLogger(__name__)


class DagsterRuntimeManager:
    """Manages Dagster runtime operations."""

    def __init__(self, config: DagsterConfig) -> None:
        """Initialize the runtime manager."""
        self.config = config
        self.config_manager = DagsterConfigurationManager(config)
        self.settings = get_settings()
        self.plugin_manager = get_plugin_manager()
        self.event_emitter = StageEventEmitter()
        self.kafka_client = KafkaClient()
        self.job_ledger = JobLedger()

    def create_definitions(self) -> Definitions:
        """Create Dagster definitions."""
        # Create pipeline resources
        pipeline_resources = self.config_manager.create_pipeline_resources()

        # Create job configurations
        job_configs = self.config_manager.create_job_configs()

        # Create asset configurations
        asset_configs = self.config_manager.create_asset_configs()

        # Create definitions
        definitions = Definitions(
            assets=self._create_assets(asset_configs),
            jobs=self._create_jobs(job_configs),
            resources=pipeline_resources,
            schedules=self._create_schedules(),
            sensors=self._create_sensors(),
        )

        return definitions

    def _create_assets(self, asset_configs: dict[str, Any]) -> list[Any]:
        """Create Dagster assets."""
        assets = []

        for name, config in asset_configs.items():
            asset_def = self._create_asset(name, config)
            assets.append(asset_def)

        return assets

    def _create_asset(self, name: str, config: Any) -> Any:
        """Create a single Dagster asset."""
        @asset(
            name=name,
            description=config.description,
            metadata=config.metadata,
        )
        def asset_fn(context: AssetExecutionContext) -> Any:
            """Asset function."""
            logger.info(f"Executing asset: {name}")

            # Mock asset execution
            return {"asset": name, "status": "completed"}

        return asset_fn

    def _create_jobs(self, job_configs: dict[str, Any]) -> list[Any]:
        """Create Dagster jobs."""
        jobs = []

        for name, config in job_configs.items():
            job_def = self._create_job(name, config)
            jobs.append(job_def)

        return jobs

    def _create_job(self, name: str, config: Any) -> Any:
        """Create a single Dagster job."""
        @op(
            name=f"{name}_op",
            config_schema=config.config_schema,
        )
        def job_op(context: Any) -> Any:
            """Job operation."""
            logger.info(f"Executing job: {name}")

            # Mock job execution
            return {"job": name, "status": "completed"}

        @job(
            name=name,
            description=config.description,
            resource_defs=config.resource_defs,
        )
        def job_def() -> None:
            """Job definition."""
            job_op()

        return job_def

    def _create_schedules(self) -> list[Any]:
        """Create Dagster schedules."""
        schedules = []

        # Create a default schedule
        @schedule(
            job="ingestion_job",
            cron_schedule="0 */6 * * *",  # Every 6 hours
        )
        def ingestion_schedule(context: Any) -> RunRequest:
            """Schedule for ingestion job."""
            return RunRequest(
                run_key=f"ingestion-{context.scheduled_execution_time}",
                tags={"scheduled": "true"},
            )

        schedules.append(ingestion_schedule)
        return schedules

    def _create_sensors(self) -> list[Any]:
        """Create Dagster sensors."""
        sensors = []

        # Create a default sensor
        @sensor(
            job="embedding_job",
        )
        def embedding_sensor(context: Any) -> RunRequest | SkipReason:
            """Sensor for embedding job."""
            # Check if there are pending embeddings
            if self._has_pending_embeddings():
                return RunRequest(
                    run_key=f"embedding-{int(time.time())}",
                    tags={"sensor": "true"},
                )
            else:
                return SkipReason("No pending embeddings")

        sensors.append(embedding_sensor)
        return sensors

    def _has_pending_embeddings(self) -> bool:
        """Check if there are pending embeddings."""
        # Mock implementation
        return True

    def execute_asset(self, asset_name: str, context: AssetExecutionContext) -> Any:
        """Execute a specific asset."""
        try:
            logger.info(f"Executing asset: {asset_name}")

            # Get asset configuration
            asset_config = self.config_manager.get_asset_config(asset_name)
            if not asset_config:
                raise ValueError(f"Asset configuration not found: {asset_name}")

            # Execute asset logic
            result = self._execute_asset_logic(asset_name, asset_config, context)

            # Record in job ledger
            self.job_ledger.record_execution(
                job_id=f"asset-{asset_name}",
                status="completed",
                result=result,
            )

            return result

        except Exception as exc:
            logger.error(f"Asset execution failed: {asset_name}, error: {exc}")

            # Record failure in job ledger
            self.job_ledger.record_execution(
                job_id=f"asset-{asset_name}",
                status="failed",
                error=str(exc),
            )

            raise exc

    def _execute_asset_logic(self, asset_name: str, config: Any, context: AssetExecutionContext) -> Any:
        """Execute asset-specific logic."""
        if asset_name == "document_asset":
            return self._execute_document_asset(config, context)
        elif asset_name == "embedding_asset":
            return self._execute_embedding_asset(config, context)
        else:
            return {"asset": asset_name, "status": "completed"}

    def _execute_document_asset(self, config: Any, context: AssetExecutionContext) -> Any:
        """Execute document asset logic."""
        # Mock implementation
        return {
            "asset": "document_asset",
            "status": "completed",
            "documents_processed": 10,
        }

    def _execute_embedding_asset(self, config: Any, context: AssetExecutionContext) -> Any:
        """Execute embedding asset logic."""
        # Mock implementation
        return {
            "asset": "embedding_asset",
            "status": "completed",
            "embeddings_generated": 100,
        }

    def execute_job(self, job_name: str, context: Any) -> Any:
        """Execute a specific job."""
        try:
            logger.info(f"Executing job: {job_name}")

            # Get job configuration
            job_config = self.config_manager.get_job_config(job_name)
            if not job_config:
                raise ValueError(f"Job configuration not found: {job_name}")

            # Execute job logic
            result = self._execute_job_logic(job_name, job_config, context)

            # Record in job ledger
            self.job_ledger.record_execution(
                job_id=f"job-{job_name}",
                status="completed",
                result=result,
            )

            return result

        except Exception as exc:
            logger.error(f"Job execution failed: {job_name}, error: {exc}")

            # Record failure in job ledger
            self.job_ledger.record_execution(
                job_id=f"job-{job_name}",
                status="failed",
                error=str(exc),
            )

            raise exc

    def _execute_job_logic(self, job_name: str, config: Any, context: Any) -> Any:
        """Execute job-specific logic."""
        if job_name == "ingestion_job":
            return self._execute_ingestion_job(config, context)
        elif job_name == "embedding_job":
            return self._execute_embedding_job(config, context)
        else:
            return {"job": job_name, "status": "completed"}

    def _execute_ingestion_job(self, config: Any, context: Any) -> Any:
        """Execute ingestion job logic."""
        # Mock implementation
        return {
            "job": "ingestion_job",
            "status": "completed",
            "documents_ingested": 5,
        }

    def _execute_embedding_job(self, config: Any, context: Any) -> Any:
        """Execute embedding job logic."""
        # Mock implementation
        return {
            "job": "embedding_job",
            "status": "completed",
            "embeddings_generated": 50,
        }

    def get_runtime_status(self) -> dict[str, Any]:
        """Get runtime status."""
        return {
            "status": "running",
            "config_valid": self.config_manager.validate_configuration(),
            "plugin_manager_available": self.plugin_manager is not None,
            "event_emitter_available": self.event_emitter is not None,
            "kafka_client_available": self.kafka_client is not None,
            "job_ledger_available": self.job_ledger is not None,
        }

    def shutdown(self) -> None:
        """Shutdown the runtime manager."""
        logger.info("Shutting down Dagster runtime manager")

        # Cleanup resources
        if self.kafka_client:
            self.kafka_client.close()

        if self.job_ledger:
            self.job_ledger.close()

        logger.info("Dagster runtime manager shutdown complete")
