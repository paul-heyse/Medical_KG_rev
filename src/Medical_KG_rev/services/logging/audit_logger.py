"""Audit logging for service operations.

This module provides comprehensive audit logging for all service operations
including authentication, authorization, data access, and system changes.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

import structlog

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)


class AuditEvent(BaseModel):
    """Audit event model."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str
    service_name: str
    operation: str
    user_id: str | None = None
    client_ip: str | None = None
    success: bool
    details: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """Audit logger for service operations."""

    def __init__(
        self,
        service_name: str,
        log_file: str | None = None,
        enable_console: bool = True,
        enable_structured: bool = True,
    ):
        """Initialize audit logger."""
        self.service_name = service_name
        self.log_file = log_file
        self.enable_console = enable_console
        self.enable_structured = enable_structured

        # Configure structured logging
        if self.enable_structured:
            self._configure_structured_logging()

        # Configure file logging
        if self.log_file:
            self._configure_file_logging()

        # Configure console logging
        if self.enable_console:
            self._configure_console_logging()

    def _configure_structured_logging(self) -> None:
        """Configure structured logging."""
        try:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer(),
                ],
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

            self.structured_logger = structlog.get_logger(self.service_name)

        except Exception as e:
            logger.error(f"Failed to configure structured logging: {e}")
            self.structured_logger = None

    def _configure_file_logging(self) -> None:
        """Configure file logging."""
        try:
            # Create log directory if it doesn't exist
            log_path = Path(self.log_file) if self.log_file else Path("audit.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Configure file handler
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            # Add handler to logger
            audit_logger = logging.getLogger(f"audit.{self.service_name}")
            audit_logger.addHandler(file_handler)
            audit_logger.setLevel(logging.INFO)

            self.file_logger = audit_logger

        except Exception as e:
            logger.error(f"Failed to configure file logging: {e}")
            self.file_logger = None

    def _configure_console_logging(self) -> None:
        """Configure console logging."""
        try:
            # Configure console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)

            # Add handler to logger
            audit_logger = logging.getLogger(f"audit.{self.service_name}")
            audit_logger.addHandler(console_handler)
            audit_logger.setLevel(logging.INFO)

            self.console_logger = audit_logger

        except Exception as e:
            logger.error(f"Failed to configure console logging: {e}")
            self.console_logger = None

    async def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        try:
            # Log to structured logger
            if self.structured_logger:
                await self._log_structured(event)

            # Log to file
            if self.file_logger:
                await self._log_file(event)

            # Log to console
            if self.console_logger:
                await self._log_console(event)

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

    async def _log_structured(self, event: AuditEvent) -> None:
        """Log event to structured logger."""
        try:
            event_data = event.model_dump()

            if event.success:
                self.structured_logger.info("audit_event", **event_data)
            else:
                self.structured_logger.warning("audit_event", **event_data)

        except Exception as e:
            logger.error(f"Failed to log structured event: {e}")

    async def _log_file(self, event: AuditEvent) -> None:
        """Log event to file."""
        try:
            event_data = event.model_dump()
            event_json = json.dumps(event_data, default=str)

            if event.success:
                self.file_logger.info(event_json)
            else:
                self.file_logger.warning(event_json)

        except Exception as e:
            logger.error(f"Failed to log file event: {e}")

    async def _log_console(self, event: AuditEvent) -> None:
        """Log event to console."""
        try:
            event_data = event.model_dump()
            event_json = json.dumps(event_data, default=str)

            if event.success:
                self.console_logger.info(event_json)
            else:
                self.console_logger.warning(event_json)

        except Exception as e:
            logger.error(f"Failed to log console event: {e}")

    async def log_authentication(
        self,
        user_id: str,
        client_ip: str,
        success: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log authentication event."""
        event = AuditEvent(
            event_type="authentication",
            service_name=self.service_name,
            operation="authenticate",
            user_id=user_id,
            client_ip=client_ip,
            success=success,
            details=details or {},
        )

        await self.log_event(event)

    async def log_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log authorization event."""
        event = AuditEvent(
            event_type="authorization",
            service_name=self.service_name,
            operation="authorize",
            user_id=user_id,
            success=success,
            details={
                "resource": resource,
                "action": action,
                **(details or {}),
            },
        )

        await self.log_event(event)

    async def log_data_access(
        self,
        user_id: str,
        data_type: str,
        operation: str,
        success: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log data access event."""
        event = AuditEvent(
            event_type="data_access",
            service_name=self.service_name,
            operation=operation,
            user_id=user_id,
            success=success,
            details={
                "data_type": data_type,
                **(details or {}),
            },
        )

        await self.log_event(event)

    async def log_system_change(
        self,
        user_id: str,
        change_type: str,
        operation: str,
        success: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log system change event."""
        event = AuditEvent(
            event_type="system_change",
            service_name=self.service_name,
            operation=operation,
            user_id=user_id,
            success=success,
            details={
                "change_type": change_type,
                **(details or {}),
            },
        )

        await self.log_event(event)

    async def log_service_operation(
        self,
        operation: str,
        success: bool,
        details: dict[str, Any] | None = None,
        user_id: str | None = None,
        client_ip: str | None = None,
    ) -> None:
        """Log service operation event."""
        event = AuditEvent(
            event_type="service_operation",
            service_name=self.service_name,
            operation=operation,
            user_id=user_id,
            client_ip=client_ip,
            success=success,
            details=details or {},
        )

        await self.log_event(event)

    async def log_error(
        self,
        error_type: str,
        error_message: str,
        operation: str,
        details: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        """Log error event."""
        event = AuditEvent(
            event_type="error",
            service_name=self.service_name,
            operation=operation,
            user_id=user_id,
            success=False,
            details={
                "error_type": error_type,
                "error_message": error_message,
                **(details or {}),
            },
        )

        await self.log_event(event)


class ServiceAuditManager:
    """Manages audit logging for multiple services."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize service audit manager."""
        self.config = config or {}
        self.audit_loggers: dict[str, AuditLogger] = {}
        self.enabled_services = self.config.get("enabled_services", [])

    def get_audit_logger(self, service_name: str) -> AuditLogger:
        """Get audit logger for a service."""
        if service_name not in self.audit_loggers:
            # Create audit logger for service
            log_file = self.config.get("log_file", f"logs/audit_{service_name}.log")
            enable_console = self.config.get("enable_console", True)
            enable_structured = self.config.get("enable_structured", True)

            self.audit_loggers[service_name] = AuditLogger(
                service_name=service_name,
                log_file=log_file,
                enable_console=enable_console,
                enable_structured=enable_structured,
            )

        return self.audit_loggers[service_name]

    async def log_service_event(
        self,
        service_name: str,
        event_type: str,
        operation: str,
        success: bool,
        details: dict[str, Any] | None = None,
        user_id: str | None = None,
        client_ip: str | None = None,
    ) -> None:
        """Log event for a service."""
        if service_name not in self.enabled_services:
            return

        audit_logger = self.get_audit_logger(service_name)

        if event_type == "authentication":
            await audit_logger.log_authentication(
                user_id=user_id or "unknown",
                client_ip=client_ip or "unknown",
                success=success,
                details=details,
            )
        elif event_type == "authorization":
            await audit_logger.log_authorization(
                user_id=user_id or "unknown",
                resource=details.get("resource", "unknown") if details else "unknown",
                action=details.get("action", "unknown") if details else "unknown",
                success=success,
                details=details,
            )
        elif event_type == "data_access":
            await audit_logger.log_data_access(
                user_id=user_id or "unknown",
                data_type=details.get("data_type", "unknown") if details else "unknown",
                operation=operation,
                success=success,
                details=details,
            )
        elif event_type == "system_change":
            await audit_logger.log_system_change(
                user_id=user_id or "unknown",
                change_type=details.get("change_type", "unknown") if details else "unknown",
                operation=operation,
                success=success,
                details=details,
            )
        else:
            await audit_logger.log_service_operation(
                operation=operation,
                success=success,
                details=details,
                user_id=user_id,
                client_ip=client_ip,
            )

    async def log_service_error(
        self,
        service_name: str,
        error_type: str,
        error_message: str,
        operation: str,
        details: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        """Log error for a service."""
        if service_name not in self.enabled_services:
            return

        audit_logger = self.get_audit_logger(service_name)
        await audit_logger.log_error(
            error_type=error_type,
            error_message=error_message,
            operation=operation,
            details=details,
            user_id=user_id,
        )


# Global audit manager instance
audit_manager = ServiceAuditManager()


async def log_service_operation(
    service_name: str,
    operation: str,
    success: bool,
    details: dict[str, Any] | None = None,
    user_id: str | None = None,
    client_ip: str | None = None,
) -> None:
    """Log service operation."""
    await audit_manager.log_service_event(
        service_name=service_name,
        event_type="service_operation",
        operation=operation,
        success=success,
        details=details,
        user_id=user_id,
        client_ip=client_ip,
    )


async def log_service_error(
    service_name: str,
    error_type: str,
    error_message: str,
    operation: str,
    details: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> None:
    """Log service error."""
    await audit_manager.log_service_error(
        service_name=service_name,
        error_type=error_type,
        error_message=error_message,
        operation=operation,
        details=details,
        user_id=user_id,
    )
