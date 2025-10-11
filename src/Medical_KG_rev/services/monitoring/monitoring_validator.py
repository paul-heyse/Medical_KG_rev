"""Monitoring validation for torch isolation architecture.

This module provides comprehensive validation of monitoring systems
for the torch isolation architecture, including metrics, dashboards,
alerting rules, and service health checks.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import aiohttp


class MonitoringComponent(Enum):
    """Enumeration of monitoring components."""

    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    ALERTMANAGER = "alertmanager"
    GPU_METRICS_EXPORTER = "gpu_metrics_exporter"
    CUSTOM_METRICS_ADAPTER = "custom_metrics_adapter"
    SERVICE_HEALTH_CHECKS = "service_health_checks"


class ValidationStatus(Enum):
    """Enumeration of validation statuses."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"


@dataclass
class ValidationResult:
    """Result of a monitoring validation check."""

    component: MonitoringComponent
    check_name: str
    status: ValidationStatus
    message: str
    details: dict[str, Any] | None = None
    execution_time: float = 0.0


@dataclass
class MonitoringConfig:
    """Configuration for monitoring validation."""

    prometheus_url: str = "http://localhost:9090"
    grafana_url: str = "http://localhost:3000"
    alertmanager_url: str = "http://localhost:9093"
    gpu_metrics_exporter_url: str = "http://localhost:8080"
    custom_metrics_adapter_url: str = "http://localhost:8081"
    service_urls: dict[str, str] | None = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        if self.service_urls is None:
            self.service_urls = {
                "gpu-management": "http://localhost:50051",
                "embedding-service": "http://localhost:50052",
                "reranking-service": "http://localhost:50053",
                "docling-vlm-service": "http://localhost:50054",
            }


class MonitoringValidator:
    """Validator for monitoring systems in torch isolation architecture."""

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.session: aiohttp.ClientSession | None = None
        self.results: list[ValidationResult] = []

    async def __aenter__(self) -> "MonitoringValidator":
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def validate_all(self) -> list[ValidationResult]:
        """Validate all monitoring components."""
        self.results = []

        # Validate Prometheus
        await self._validate_prometheus()

        # Validate Grafana
        await self._validate_grafana()

        # Validate Alertmanager
        await self._validate_alertmanager()

        # Validate GPU metrics exporter
        await self._validate_gpu_metrics_exporter()

        # Validate custom metrics adapter
        await self._validate_custom_metrics_adapter()

        # Validate service health checks
        await self._validate_service_health_checks()

        # Validate monitoring integration
        await self._validate_monitoring_integration()

        return self.results

    async def _validate_prometheus(self) -> None:
        """Validate Prometheus configuration and metrics."""
        start_time = time.time()

        try:
            # Check Prometheus health
            health_url = f"{self.config.prometheus_url}/api/v1/status/config"
            self._check_session()
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    config_data = await response.json()
                    self._add_result(
                        MonitoringComponent.PROMETHEUS,
                        "prometheus_health",
                        ValidationStatus.PASS,
                        "Prometheus is healthy and accessible",
                        {"config_loaded": True},
                    )
                else:
                    self._add_result(
                        MonitoringComponent.PROMETHEUS,
                        "prometheus_health",
                        ValidationStatus.FAIL,
                        f"Prometheus health check failed with status {response.status}",
                    )

            # Check for required metrics
            required_metrics = [
                "gpu_service_calls_total",
                "gpu_service_call_duration_seconds",
                "gpu_service_errors_total",
                "gpu_memory_usage_mb",
                "gpu_service_health_status",
                "circuit_breaker_state",
            ]

            for metric in required_metrics:
                await self._check_metric_exists(metric)

            # Check Prometheus configuration
            await self._check_prometheus_config()

        except Exception as e:
            self._add_result(
                MonitoringComponent.PROMETHEUS,
                "prometheus_validation",
                ValidationStatus.FAIL,
                f"Prometheus validation failed: {e!s}",
            )

        execution_time = time.time() - start_time
        self.results[-1].execution_time = execution_time

    async def _validate_grafana(self) -> None:
        """Validate Grafana configuration and dashboards."""
        start_time = time.time()

        try:
            # Check Grafana health
            health_url = f"{self.config.grafana_url}/api/health"
            self._check_session()
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    self._add_result(
                        MonitoringComponent.GRAFANA,
                        "grafana_health",
                        ValidationStatus.PASS,
                        "Grafana is healthy and accessible",
                        health_data,
                    )
                else:
                    self._add_result(
                        MonitoringComponent.GRAFANA,
                        "grafana_health",
                        ValidationStatus.FAIL,
                        f"Grafana health check failed with status {response.status}",
                    )

            # Check for required dashboards
            required_dashboards = [
                "gpu-services-dashboard",
                "service-architecture-dashboard",
                "auto-scaling-dashboard",
                "gpu-optimization-dashboard",
                "cache-monitoring-dashboard",
            ]

            for dashboard in required_dashboards:
                await self._check_dashboard_exists(dashboard)

            # Check Grafana data sources
            await self._check_grafana_data_sources()

        except Exception as e:
            self._add_result(
                MonitoringComponent.GRAFANA,
                "grafana_validation",
                ValidationStatus.FAIL,
                f"Grafana validation failed: {e!s}",
            )

        execution_time = time.time() - start_time
        self.results[-1].execution_time = execution_time

    async def _validate_alertmanager(self) -> None:
        """Validate Alertmanager configuration and rules."""
        start_time = time.time()

        try:
            # Check Alertmanager health
            health_url = f"{self.config.alertmanager_url}/api/v1/status"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    status_data = await response.json()
                    self._add_result(
                        MonitoringComponent.ALERTMANAGER,
                        "alertmanager_health",
                        ValidationStatus.PASS,
                        "Alertmanager is healthy and accessible",
                        status_data,
                    )
                else:
                    self._add_result(
                        MonitoringComponent.ALERTMANAGER,
                        "alertmanager_health",
                        ValidationStatus.FAIL,
                        f"Alertmanager health check failed with status {response.status}",
                    )

            # Check for required alert rules
            required_alerts = [
                "GPUServiceFailures",
                "CircuitBreakerOpen",
                "HighGPUServiceLatency",
                "HighGPUMemoryUsage",
            ]

            for alert in required_alerts:
                await self._check_alert_rule_exists(alert)

            # Check Alertmanager configuration
            await self._check_alertmanager_config()

        except Exception as e:
            self._add_result(
                MonitoringComponent.ALERTMANAGER,
                "alertmanager_validation",
                ValidationStatus.FAIL,
                f"Alertmanager validation failed: {e!s}",
            )

        execution_time = time.time() - start_time
        self.results[-1].execution_time = execution_time

    async def _validate_gpu_metrics_exporter(self) -> None:
        """Validate GPU metrics exporter."""
        start_time = time.time()

        try:
            # Check GPU metrics exporter health
            health_url = f"{self.config.gpu_metrics_exporter_url}/health"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    self._add_result(
                        MonitoringComponent.GPU_METRICS_EXPORTER,
                        "gpu_metrics_exporter_health",
                        ValidationStatus.PASS,
                        "GPU metrics exporter is healthy and accessible",
                        health_data,
                    )
                else:
                    self._add_result(
                        MonitoringComponent.GPU_METRICS_EXPORTER,
                        "gpu_metrics_exporter_health",
                        ValidationStatus.FAIL,
                        f"GPU metrics exporter health check failed with status {response.status}",
                    )

            # Check for required GPU metrics
            required_gpu_metrics = [
                "gpu_utilization_percentage",
                "gpu_memory_usage_mb",
                "gpu_temperature_celsius",
                "gpu_power_usage_watts",
            ]

            for metric in required_gpu_metrics:
                await self._check_gpu_metric_exists(metric)

        except Exception as e:
            self._add_result(
                MonitoringComponent.GPU_METRICS_EXPORTER,
                "gpu_metrics_exporter_validation",
                ValidationStatus.FAIL,
                f"GPU metrics exporter validation failed: {e!s}",
            )

        execution_time = time.time() - start_time
        self.results[-1].execution_time = execution_time

    async def _validate_custom_metrics_adapter(self) -> None:
        """Validate custom metrics adapter."""
        start_time = time.time()

        try:
            # Check custom metrics adapter health
            health_url = f"{self.config.custom_metrics_adapter_url}/health"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    self._add_result(
                        MonitoringComponent.CUSTOM_METRICS_ADAPTER,
                        "custom_metrics_adapter_health",
                        ValidationStatus.PASS,
                        "Custom metrics adapter is healthy and accessible",
                        health_data,
                    )
                else:
                    self._add_result(
                        MonitoringComponent.CUSTOM_METRICS_ADAPTER,
                        "custom_metrics_adapter_health",
                        ValidationStatus.FAIL,
                        f"Custom metrics adapter health check failed with status {response.status}",
                    )

            # Check for required custom metrics
            required_custom_metrics = ["gpu_utilization_percentage", "gpu_memory_usage_mb"]

            for metric in required_custom_metrics:
                await self._check_custom_metric_exists(metric)

        except Exception as e:
            self._add_result(
                MonitoringComponent.CUSTOM_METRICS_ADAPTER,
                "custom_metrics_adapter_validation",
                ValidationStatus.FAIL,
                f"Custom metrics adapter validation failed: {e!s}",
            )

        execution_time = time.time() - start_time
        self.results[-1].execution_time = execution_time

    async def _validate_service_health_checks(self) -> None:
        """Validate service health checks."""
        start_time = time.time()

        try:
            for service_name, service_url in self.config.service_urls.items():
                await self._check_service_health(service_name, service_url)

        except Exception as e:
            self._add_result(
                MonitoringComponent.SERVICE_HEALTH_CHECKS,
                "service_health_checks_validation",
                ValidationStatus.FAIL,
                f"Service health checks validation failed: {e!s}",
            )

        execution_time = time.time() - start_time
        self.results[-1].execution_time = execution_time

    async def _validate_monitoring_integration(self) -> None:
        """Validate monitoring integration."""
        start_time = time.time()

        try:
            # Check Prometheus-Grafana integration
            await self._check_prometheus_grafana_integration()

            # Check Prometheus-Alertmanager integration
            await self._check_prometheus_alertmanager_integration()

            # Check GPU metrics integration
            await self._check_gpu_metrics_integration()

            # Check custom metrics integration
            await self._check_custom_metrics_integration()

        except Exception as e:
            self._add_result(
                MonitoringComponent.PROMETHEUS,
                "monitoring_integration_validation",
                ValidationStatus.FAIL,
                f"Monitoring integration validation failed: {e!s}",
            )

        execution_time = time.time() - start_time
        self.results[-1].execution_time = execution_time

    async def _check_metric_exists(self, metric_name: str) -> None:
        """Check if a metric exists in Prometheus."""
        try:
            query_url = f"{self.config.prometheus_url}/api/v1/query"
            params = {"query": metric_name}

            async with self.session.get(query_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success" and data.get("data", {}).get("result"):
                        self._add_result(
                            MonitoringComponent.PROMETHEUS,
                            f"metric_{metric_name}",
                            ValidationStatus.PASS,
                            f"Metric {metric_name} exists and has data",
                        )
                    else:
                        self._add_result(
                            MonitoringComponent.PROMETHEUS,
                            f"metric_{metric_name}",
                            ValidationStatus.WARNING,
                            f"Metric {metric_name} exists but has no data",
                        )
                else:
                    self._add_result(
                        MonitoringComponent.PROMETHEUS,
                        f"metric_{metric_name}",
                        ValidationStatus.FAIL,
                        f"Failed to query metric {metric_name}",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.PROMETHEUS,
                f"metric_{metric_name}",
                ValidationStatus.FAIL,
                f"Error checking metric {metric_name}: {e!s}",
            )

    async def _check_dashboard_exists(self, dashboard_name: str) -> None:
        """Check if a dashboard exists in Grafana."""
        try:
            search_url = f"{self.config.grafana_url}/api/search"
            params = {"query": dashboard_name, "type": "dash-db"}

            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        self._add_result(
                            MonitoringComponent.GRAFANA,
                            f"dashboard_{dashboard_name}",
                            ValidationStatus.PASS,
                            f"Dashboard {dashboard_name} exists",
                        )
                    else:
                        self._add_result(
                            MonitoringComponent.GRAFANA,
                            f"dashboard_{dashboard_name}",
                            ValidationStatus.WARNING,
                            f"Dashboard {dashboard_name} not found",
                        )
                else:
                    self._add_result(
                        MonitoringComponent.GRAFANA,
                        f"dashboard_{dashboard_name}",
                        ValidationStatus.FAIL,
                        f"Failed to search for dashboard {dashboard_name}",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.GRAFANA,
                f"dashboard_{dashboard_name}",
                ValidationStatus.FAIL,
                f"Error checking dashboard {dashboard_name}: {e!s}",
            )

    async def _check_alert_rule_exists(self, alert_name: str) -> None:
        """Check if an alert rule exists in Alertmanager."""
        try:
            rules_url = f"{self.config.alertmanager_url}/api/v1/rules"

            async with self.session.get(rules_url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Check if alert rule exists in the rules data
                    alert_found = False
                    for group in data.get("data", {}).get("groups", []):
                        for rule in group.get("rules", []):
                            if rule.get("name") == alert_name:
                                alert_found = True
                                break
                        if alert_found:
                            break

                    if alert_found:
                        self._add_result(
                            MonitoringComponent.ALERTMANAGER,
                            f"alert_{alert_name}",
                            ValidationStatus.PASS,
                            f"Alert rule {alert_name} exists",
                        )
                    else:
                        self._add_result(
                            MonitoringComponent.ALERTMANAGER,
                            f"alert_{alert_name}",
                            ValidationStatus.WARNING,
                            f"Alert rule {alert_name} not found",
                        )
                else:
                    self._add_result(
                        MonitoringComponent.ALERTMANAGER,
                        f"alert_{alert_name}",
                        ValidationStatus.FAIL,
                        "Failed to get alert rules",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.ALERTMANAGER,
                f"alert_{alert_name}",
                ValidationStatus.FAIL,
                f"Error checking alert rule {alert_name}: {e!s}",
            )

    async def _check_service_health(self, service_name: str, service_url: str) -> None:
        """Check health of a specific service."""
        try:
            health_url = f"{service_url}/health"

            async with self.session.get(health_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    self._add_result(
                        MonitoringComponent.SERVICE_HEALTH_CHECKS,
                        f"service_{service_name}_health",
                        ValidationStatus.PASS,
                        f"Service {service_name} is healthy",
                        health_data,
                    )
                else:
                    self._add_result(
                        MonitoringComponent.SERVICE_HEALTH_CHECKS,
                        f"service_{service_name}_health",
                        ValidationStatus.FAIL,
                        f"Service {service_name} health check failed with status {response.status}",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.SERVICE_HEALTH_CHECKS,
                f"service_{service_name}_health",
                ValidationStatus.FAIL,
                f"Error checking service {service_name} health: {e!s}",
            )

    async def _check_prometheus_config(self) -> None:
        """Check Prometheus configuration."""
        try:
            config_url = f"{self.config.prometheus_url}/api/v1/status/config"

            async with self.session.get(config_url) as response:
                if response.status == 200:
                    config_data = await response.json()
                    self._add_result(
                        MonitoringComponent.PROMETHEUS,
                        "prometheus_config",
                        ValidationStatus.PASS,
                        "Prometheus configuration is valid",
                        config_data,
                    )
                else:
                    self._add_result(
                        MonitoringComponent.PROMETHEUS,
                        "prometheus_config",
                        ValidationStatus.FAIL,
                        "Failed to get Prometheus configuration",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.PROMETHEUS,
                "prometheus_config",
                ValidationStatus.FAIL,
                f"Error checking Prometheus configuration: {e!s}",
            )

    async def _check_grafana_data_sources(self) -> None:
        """Check Grafana data sources."""
        try:
            datasources_url = f"{self.config.grafana_url}/api/datasources"

            async with self.session.get(datasources_url) as response:
                if response.status == 200:
                    data = await response.json()
                    self._add_result(
                        MonitoringComponent.GRAFANA,
                        "grafana_data_sources",
                        ValidationStatus.PASS,
                        f"Found {len(data)} data sources",
                        {"data_sources": data},
                    )
                else:
                    self._add_result(
                        MonitoringComponent.GRAFANA,
                        "grafana_data_sources",
                        ValidationStatus.FAIL,
                        "Failed to get Grafana data sources",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.GRAFANA,
                "grafana_data_sources",
                ValidationStatus.FAIL,
                f"Error checking Grafana data sources: {e!s}",
            )

    async def _check_alertmanager_config(self) -> None:
        """Check Alertmanager configuration."""
        try:
            config_url = f"{self.config.alertmanager_url}/api/v1/status"

            async with self.session.get(config_url) as response:
                if response.status == 200:
                    status_data = await response.json()
                    self._add_result(
                        MonitoringComponent.ALERTMANAGER,
                        "alertmanager_config",
                        ValidationStatus.PASS,
                        "Alertmanager configuration is valid",
                        status_data,
                    )
                else:
                    self._add_result(
                        MonitoringComponent.ALERTMANAGER,
                        "alertmanager_config",
                        ValidationStatus.FAIL,
                        "Failed to get Alertmanager configuration",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.ALERTMANAGER,
                "alertmanager_config",
                ValidationStatus.FAIL,
                f"Error checking Alertmanager configuration: {e!s}",
            )

    async def _check_gpu_metric_exists(self, metric_name: str) -> None:
        """Check if a GPU metric exists."""
        try:
            metrics_url = f"{self.config.gpu_metrics_exporter_url}/metrics"

            async with self.session.get(metrics_url) as response:
                if response.status == 200:
                    metrics_text = await response.text()
                    if metric_name in metrics_text:
                        self._add_result(
                            MonitoringComponent.GPU_METRICS_EXPORTER,
                            f"gpu_metric_{metric_name}",
                            ValidationStatus.PASS,
                            f"GPU metric {metric_name} exists",
                        )
                    else:
                        self._add_result(
                            MonitoringComponent.GPU_METRICS_EXPORTER,
                            f"gpu_metric_{metric_name}",
                            ValidationStatus.WARNING,
                            f"GPU metric {metric_name} not found",
                        )
                else:
                    self._add_result(
                        MonitoringComponent.GPU_METRICS_EXPORTER,
                        f"gpu_metric_{metric_name}",
                        ValidationStatus.FAIL,
                        "Failed to get GPU metrics",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.GPU_METRICS_EXPORTER,
                f"gpu_metric_{metric_name}",
                ValidationStatus.FAIL,
                f"Error checking GPU metric {metric_name}: {e!s}",
            )

    async def _check_custom_metric_exists(self, metric_name: str) -> None:
        """Check if a custom metric exists."""
        try:
            metrics_url = f"{self.config.custom_metrics_adapter_url}/metrics"

            async with self.session.get(metrics_url) as response:
                if response.status == 200:
                    metrics_text = await response.text()
                    if metric_name in metrics_text:
                        self._add_result(
                            MonitoringComponent.CUSTOM_METRICS_ADAPTER,
                            f"custom_metric_{metric_name}",
                            ValidationStatus.PASS,
                            f"Custom metric {metric_name} exists",
                        )
                    else:
                        self._add_result(
                            MonitoringComponent.CUSTOM_METRICS_ADAPTER,
                            f"custom_metric_{metric_name}",
                            ValidationStatus.WARNING,
                            f"Custom metric {metric_name} not found",
                        )
                else:
                    self._add_result(
                        MonitoringComponent.CUSTOM_METRICS_ADAPTER,
                        f"custom_metric_{metric_name}",
                        ValidationStatus.FAIL,
                        "Failed to get custom metrics",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.CUSTOM_METRICS_ADAPTER,
                f"custom_metric_{metric_name}",
                ValidationStatus.FAIL,
                f"Error checking custom metric {metric_name}: {e!s}",
            )

    async def _check_prometheus_grafana_integration(self) -> None:
        """Check Prometheus-Grafana integration."""
        try:
            # Check if Grafana can query Prometheus
            query_url = f"{self.config.grafana_url}/api/datasources/proxy/1/api/v1/query"
            params = {"query": "up"}

            async with self.session.get(query_url, params=params) as response:
                if response.status == 200:
                    self._add_result(
                        MonitoringComponent.GRAFANA,
                        "prometheus_grafana_integration",
                        ValidationStatus.PASS,
                        "Prometheus-Grafana integration is working",
                    )
                else:
                    self._add_result(
                        MonitoringComponent.GRAFANA,
                        "prometheus_grafana_integration",
                        ValidationStatus.FAIL,
                        f"Prometheus-Grafana integration failed with status {response.status}",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.GRAFANA,
                "prometheus_grafana_integration",
                ValidationStatus.FAIL,
                f"Error checking Prometheus-Grafana integration: {e!s}",
            )

    async def _check_prometheus_alertmanager_integration(self) -> None:
        """Check Prometheus-Alertmanager integration."""
        try:
            # Check if Prometheus can send alerts to Alertmanager
            alerts_url = f"{self.config.alertmanager_url}/api/v1/alerts"

            async with self.session.get(alerts_url) as response:
                if response.status == 200:
                    alerts_data = await response.json()
                    self._add_result(
                        MonitoringComponent.ALERTMANAGER,
                        "prometheus_alertmanager_integration",
                        ValidationStatus.PASS,
                        "Prometheus-Alertmanager integration is working",
                        {"active_alerts": len(alerts_data)},
                    )
                else:
                    self._add_result(
                        MonitoringComponent.ALERTMANAGER,
                        "prometheus_alertmanager_integration",
                        ValidationStatus.FAIL,
                        f"Prometheus-Alertmanager integration failed with status {response.status}",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.ALERTMANAGER,
                "prometheus_alertmanager_integration",
                ValidationStatus.FAIL,
                f"Error checking Prometheus-Alertmanager integration: {e!s}",
            )

    async def _check_gpu_metrics_integration(self) -> None:
        """Check GPU metrics integration."""
        try:
            # Check if GPU metrics are available in Prometheus
            query_url = f"{self.config.prometheus_url}/api/v1/query"
            params = {"query": "gpu_utilization_percentage"}

            async with self.session.get(query_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success":
                        self._add_result(
                            MonitoringComponent.PROMETHEUS,
                            "gpu_metrics_integration",
                            ValidationStatus.PASS,
                            "GPU metrics integration is working",
                        )
                    else:
                        self._add_result(
                            MonitoringComponent.PROMETHEUS,
                            "gpu_metrics_integration",
                            ValidationStatus.WARNING,
                            "GPU metrics integration has issues",
                        )
                else:
                    self._add_result(
                        MonitoringComponent.PROMETHEUS,
                        "gpu_metrics_integration",
                        ValidationStatus.FAIL,
                        f"GPU metrics integration failed with status {response.status}",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.PROMETHEUS,
                "gpu_metrics_integration",
                ValidationStatus.FAIL,
                f"Error checking GPU metrics integration: {e!s}",
            )

    async def _check_custom_metrics_integration(self) -> None:
        """Check custom metrics integration."""
        try:
            # Check if custom metrics are available in Prometheus
            query_url = f"{self.config.prometheus_url}/api/v1/query"
            params = {"query": "gpu_utilization_percentage"}

            async with self.session.get(query_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success":
                        self._add_result(
                            MonitoringComponent.PROMETHEUS,
                            "custom_metrics_integration",
                            ValidationStatus.PASS,
                            "Custom metrics integration is working",
                        )
                    else:
                        self._add_result(
                            MonitoringComponent.PROMETHEUS,
                            "custom_metrics_integration",
                            ValidationStatus.WARNING,
                            "Custom metrics integration has issues",
                        )
                else:
                    self._add_result(
                        MonitoringComponent.PROMETHEUS,
                        "custom_metrics_integration",
                        ValidationStatus.FAIL,
                        f"Custom metrics integration failed with status {response.status}",
                    )
        except Exception as e:
            self._add_result(
                MonitoringComponent.PROMETHEUS,
                "custom_metrics_integration",
                ValidationStatus.FAIL,
                f"Error checking custom metrics integration: {e!s}",
            )

    def _check_session(self) -> None:
        """Check if session is initialized."""
        if self.session is None:
            raise Exception("Session not initialized")

    def _add_result(
        self,
        component: MonitoringComponent,
        check_name: str,
        status: ValidationStatus,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add a validation result."""
        result = ValidationResult(
            component=component,
            check_name=check_name,
            status=status,
            message=message,
            details=details,
        )
        self.results.append(result)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of validation results."""
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.status == ValidationStatus.PASS])
        failed_checks = len([r for r in self.results if r.status == ValidationStatus.FAIL])
        warning_checks = len([r for r in self.results if r.status == ValidationStatus.WARNING])

        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "warning_checks": warning_checks,
            "success_rate": passed_checks / total_checks if total_checks > 0 else 0.0,
            "results": self.results,
        }
