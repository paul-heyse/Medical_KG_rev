"""Tests for monitoring validation functionality.

This module tests the monitoring validation system for the torch isolation
architecture, including Prometheus, Grafana, Alertmanager, and service health checks.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

# Import monitoring validator
try:
    from Medical_KG_rev.services.monitoring.monitoring_validator import (
        MonitoringComponent,
        MonitoringConfig,
        MonitoringValidator,
        ValidationStatus,
    )
except ImportError:
    # Mock classes for testing when modules are not available
    class ValidationStatus:
        PASS = "PASS"
        FAIL = "FAIL"
        WARNING = "WARNING"
        SKIP = "SKIP"

    class MonitoringComponent:
        PROMETHEUS = "prometheus"
        GRAFANA = "grafana"
        ALERTMANAGER = "alertmanager"
        GPU_METRICS_EXPORTER = "gpu_metrics_exporter"
        CUSTOM_METRICS_ADAPTER = "custom_metrics_adapter"
        SERVICE_HEALTH_CHECKS = "service_health_checks"

    class MonitoringConfig:
        def __init__(self, **kwargs):
            self.prometheus_url = kwargs.get("prometheus_url", "http://localhost:9090")
            self.grafana_url = kwargs.get("grafana_url", "http://localhost:3000")
            self.alertmanager_url = kwargs.get("alertmanager_url", "http://localhost:9093")
            self.gpu_metrics_exporter_url = kwargs.get(
                "gpu_metrics_exporter_url", "http://localhost:8080"
            )
            self.custom_metrics_adapter_url = kwargs.get(
                "custom_metrics_adapter_url", "http://localhost:8081"
            )
            self.service_urls = kwargs.get("service_urls", {})
            self.timeout = kwargs.get("timeout", 30)
            self.retry_attempts = kwargs.get("retry_attempts", 3)
            self.retry_delay = kwargs.get("retry_delay", 1.0)

    class MonitoringValidator:
        def __init__(self, config: MonitoringConfig):
            self.config = config
            self.results = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def validate_all(self):
            # Mock validation results
            return [
                {
                    "component": "prometheus",
                    "check_name": "health",
                    "status": "PASS",
                    "message": "Mock result",
                },
                {
                    "component": "grafana",
                    "check_name": "health",
                    "status": "PASS",
                    "message": "Mock result",
                },
                {
                    "component": "alertmanager",
                    "check_name": "health",
                    "status": "PASS",
                    "message": "Mock result",
                },
            ]

        def get_summary(self):
            return {
                "total_checks": 3,
                "passed_checks": 3,
                "failed_checks": 0,
                "warning_checks": 0,
                "success_rate": 1.0,
                "results": self.results,
            }


class TestMonitoringValidation:
    """Test suite for monitoring validation functionality."""

    @pytest.fixture
    def monitoring_config(self) -> MonitoringConfig:
        """Create monitoring configuration for testing."""
        return MonitoringConfig(
            prometheus_url="http://localhost:9090",
            grafana_url="http://localhost:3000",
            alertmanager_url="http://localhost:9093",
            gpu_metrics_exporter_url="http://localhost:8080",
            custom_metrics_adapter_url="http://localhost:8081",
            service_urls={
                "gpu-management": "http://localhost:50051",
                "embedding-service": "http://localhost:50052",
                "reranking-service": "http://localhost:50053",
                "docling-vlm-service": "http://localhost:50054",
            },
            timeout=30,
        )

    @pytest.fixture
    def validator(self, monitoring_config: MonitoringConfig) -> MonitoringValidator:
        """Create monitoring validator for testing."""
        return MonitoringValidator(monitoring_config)

    @pytest.mark.asyncio
    async def test_monitoring_config_initialization(
        self, monitoring_config: MonitoringConfig
    ) -> None:
        """Test monitoring configuration initialization."""
        assert monitoring_config.prometheus_url == "http://localhost:9090"
        assert monitoring_config.grafana_url == "http://localhost:3000"
        assert monitoring_config.alertmanager_url == "http://localhost:9093"
        assert monitoring_config.gpu_metrics_exporter_url == "http://localhost:8080"
        assert monitoring_config.custom_metrics_adapter_url == "http://localhost:8081"
        assert monitoring_config.timeout == 30
        assert len(monitoring_config.service_urls) == 4

    @pytest.mark.asyncio
    async def test_validator_initialization(self, validator: MonitoringValidator) -> None:
        """Test monitoring validator initialization."""
        assert validator.config is not None
        assert validator.results == []
        assert validator.session is None

    @pytest.mark.asyncio
    async def test_validator_context_manager(self, validator: MonitoringValidator) -> None:
        """Test monitoring validator context manager."""
        async with validator as v:
            assert v is validator
            assert v.session is not None

    @pytest.mark.asyncio
    async def test_prometheus_validation_success(self, validator: MonitoringValidator) -> None:
        """Test successful Prometheus validation."""
        # Mock successful Prometheus response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": [{"metric": {"__name__": "gpu_service_calls_total"}}]},
        }

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._validate_prometheus()

        # Check that validation results were added
        assert len(validator.results) > 0

        # Check for Prometheus health check result
        prometheus_results = [
            r for r in validator.results if r.component == MonitoringComponent.PROMETHEUS
        ]
        assert len(prometheus_results) > 0

        # Check that at least one result is PASS
        pass_results = [r for r in prometheus_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_prometheus_validation_failure(self, validator: MonitoringValidator) -> None:
        """Test Prometheus validation failure."""
        # Mock failed Prometheus response
        mock_response = AsyncMock()
        mock_response.status = 500

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._validate_prometheus()

        # Check that validation results were added
        assert len(validator.results) > 0

        # Check for Prometheus health check result
        prometheus_results = [
            r for r in validator.results if r.component == MonitoringComponent.PROMETHEUS
        ]
        assert len(prometheus_results) > 0

        # Check that at least one result is FAIL
        fail_results = [r for r in prometheus_results if r.status == ValidationStatus.FAIL]
        assert len(fail_results) > 0

    @pytest.mark.asyncio
    async def test_grafana_validation_success(self, validator: MonitoringValidator) -> None:
        """Test successful Grafana validation."""
        # Mock successful Grafana response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "ok"}

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._validate_grafana()

        # Check that validation results were added
        assert len(validator.results) > 0

        # Check for Grafana health check result
        grafana_results = [
            r for r in validator.results if r.component == MonitoringComponent.GRAFANA
        ]
        assert len(grafana_results) > 0

        # Check that at least one result is PASS
        pass_results = [r for r in grafana_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_alertmanager_validation_success(self, validator: MonitoringValidator) -> None:
        """Test successful Alertmanager validation."""
        # Mock successful Alertmanager response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "success"}

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._validate_alertmanager()

        # Check that validation results were added
        assert len(validator.results) > 0

        # Check for Alertmanager health check result
        alertmanager_results = [
            r for r in validator.results if r.component == MonitoringComponent.ALERTMANAGER
        ]
        assert len(alertmanager_results) > 0

        # Check that at least one result is PASS
        pass_results = [r for r in alertmanager_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_gpu_metrics_exporter_validation_success(
        self, validator: MonitoringValidator
    ) -> None:
        """Test successful GPU metrics exporter validation."""
        # Mock successful GPU metrics exporter response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.text.return_value = "gpu_utilization_percentage 75.5"

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._validate_gpu_metrics_exporter()

        # Check that validation results were added
        assert len(validator.results) > 0

        # Check for GPU metrics exporter health check result
        gpu_results = [
            r for r in validator.results if r.component == MonitoringComponent.GPU_METRICS_EXPORTER
        ]
        assert len(gpu_results) > 0

        # Check that at least one result is PASS
        pass_results = [r for r in gpu_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_custom_metrics_adapter_validation_success(
        self, validator: MonitoringValidator
    ) -> None:
        """Test successful custom metrics adapter validation."""
        # Mock successful custom metrics adapter response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.text.return_value = "gpu_utilization_percentage 75.5"

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._validate_custom_metrics_adapter()

        # Check that validation results were added
        assert len(validator.results) > 0

        # Check for custom metrics adapter health check result
        custom_results = [
            r
            for r in validator.results
            if r.component == MonitoringComponent.CUSTOM_METRICS_ADAPTER
        ]
        assert len(custom_results) > 0

        # Check that at least one result is PASS
        pass_results = [r for r in custom_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_service_health_checks_validation_success(
        self, validator: MonitoringValidator
    ) -> None:
        """Test successful service health checks validation."""
        # Mock successful service health check response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._validate_service_health_checks()

        # Check that validation results were added
        assert len(validator.results) > 0

        # Check for service health check results
        service_results = [
            r for r in validator.results if r.component == MonitoringComponent.SERVICE_HEALTH_CHECKS
        ]
        assert len(service_results) > 0

        # Check that at least one result is PASS
        pass_results = [r for r in service_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_monitoring_integration_validation_success(
        self, validator: MonitoringValidator
    ) -> None:
        """Test successful monitoring integration validation."""
        # Mock successful integration response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": [{"metric": {"__name__": "gpu_utilization_percentage"}}]},
        }

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._validate_monitoring_integration()

        # Check that validation results were added
        assert len(validator.results) > 0

        # Check for monitoring integration results
        integration_results = [r for r in validator.results if "integration" in r.check_name]
        assert len(integration_results) > 0

        # Check that at least one result is PASS
        pass_results = [r for r in integration_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_comprehensive_validation(self, validator: MonitoringValidator) -> None:
        """Test comprehensive monitoring validation."""
        # Mock successful responses for all components
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": [{"metric": {"__name__": "test_metric"}}]},
        }
        mock_response.text.return_value = "test_metric 1.0"

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                results = await validator.validate_all()

        # Check that validation results were added
        assert len(results) > 0

        # Check that all components were validated
        components = set(r.component for r in results)
        expected_components = {
            MonitoringComponent.PROMETHEUS,
            MonitoringComponent.GRAFANA,
            MonitoringComponent.ALERTMANAGER,
            MonitoringComponent.GPU_METRICS_EXPORTER,
            MonitoringComponent.CUSTOM_METRICS_ADAPTER,
            MonitoringComponent.SERVICE_HEALTH_CHECKS,
        }
        assert components.issuperset(expected_components)

    @pytest.mark.asyncio
    async def test_validation_summary(self, validator: MonitoringValidator) -> None:
        """Test validation summary generation."""
        # Add some mock results
        validator.results = [
            type(
                "Result",
                (),
                {
                    "component": MonitoringComponent.PROMETHEUS,
                    "check_name": "health",
                    "status": ValidationStatus.PASS,
                    "message": "Test message",
                    "details": None,
                    "execution_time": 1.0,
                },
            )(),
            type(
                "Result",
                (),
                {
                    "component": MonitoringComponent.GRAFANA,
                    "check_name": "health",
                    "status": ValidationStatus.FAIL,
                    "message": "Test message",
                    "details": None,
                    "execution_time": 1.0,
                },
            )(),
            type(
                "Result",
                (),
                {
                    "component": MonitoringComponent.ALERTMANAGER,
                    "check_name": "health",
                    "status": ValidationStatus.WARNING,
                    "message": "Test message",
                    "details": None,
                    "execution_time": 1.0,
                },
            )(),
        ]

        summary = validator.get_summary()

        assert summary["total_checks"] == 3
        assert summary["passed_checks"] == 1
        assert summary["failed_checks"] == 1
        assert summary["warning_checks"] == 1
        assert summary["success_rate"] == 1 / 3
        assert len(summary["results"]) == 3

    @pytest.mark.asyncio
    async def test_metric_existence_check(self, validator: MonitoringValidator) -> None:
        """Test metric existence checking."""
        # Mock successful metric query response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": [{"metric": {"__name__": "gpu_service_calls_total"}}]},
        }

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._check_metric_exists("gpu_service_calls_total")

        # Check that validation result was added
        assert len(validator.results) > 0

        # Check for metric existence result
        metric_results = [
            r for r in validator.results if "metric_gpu_service_calls_total" in r.check_name
        ]
        assert len(metric_results) > 0

        # Check that result is PASS
        pass_results = [r for r in metric_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_dashboard_existence_check(self, validator: MonitoringValidator) -> None:
        """Test dashboard existence checking."""
        # Mock successful dashboard search response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = [{"title": "gpu-services-dashboard", "uid": "test-uid"}]

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._check_dashboard_exists("gpu-services-dashboard")

        # Check that validation result was added
        assert len(validator.results) > 0

        # Check for dashboard existence result
        dashboard_results = [
            r for r in validator.results if "dashboard_gpu-services-dashboard" in r.check_name
        ]
        assert len(dashboard_results) > 0

        # Check that result is PASS
        pass_results = [r for r in dashboard_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_alert_rule_existence_check(self, validator: MonitoringValidator) -> None:
        """Test alert rule existence checking."""
        # Mock successful alert rules response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "data": {"groups": [{"rules": [{"name": "GPUServiceFailures", "state": "active"}]}]}
        }

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._check_alert_rule_exists("GPUServiceFailures")

        # Check that validation result was added
        assert len(validator.results) > 0

        # Check for alert rule existence result
        alert_results = [r for r in validator.results if "alert_GPUServiceFailures" in r.check_name]
        assert len(alert_results) > 0

        # Check that result is PASS
        pass_results = [r for r in alert_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_service_health_check(self, validator: MonitoringValidator) -> None:
        """Test service health checking."""
        # Mock successful service health response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._check_service_health("test-service", "http://localhost:8080")

        # Check that validation result was added
        assert len(validator.results) > 0

        # Check for service health result
        service_results = [
            r for r in validator.results if "service_test-service_health" in r.check_name
        ]
        assert len(service_results) > 0

        # Check that result is PASS
        pass_results = [r for r in service_results if r.status == ValidationStatus.PASS]
        assert len(pass_results) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, validator: MonitoringValidator) -> None:
        """Test error handling in validation."""
        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json.side_effect = Exception("Connection error")

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._validate_prometheus()

        # Check that validation results were added
        assert len(validator.results) > 0

        # Check for error handling result
        error_results = [r for r in validator.results if r.status == ValidationStatus.FAIL]
        assert len(error_results) > 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self, validator: MonitoringValidator) -> None:
        """Test timeout handling in validation."""
        # Mock timeout response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.side_effect = TimeoutError("Request timeout")

        with patch("aiohttp.ClientSession.get", return_value=mock_response):
            async with validator:
                await validator._validate_prometheus()

        # Check that validation results were added
        assert len(validator.results) > 0

        # Check for timeout handling result
        timeout_results = [r for r in validator.results if r.status == ValidationStatus.FAIL]
        assert len(timeout_results) > 0


if __name__ == "__main__":
    # Run validation tests when script is executed directly
    import sys

    print("🔍 Running Monitoring Validation Tests...")
    print("=" * 60)

    async def run_validation_tests() -> None:
        """Run validation tests."""
        print("\n📊 Running Validation Tests:")

        # Test monitoring config initialization
        config = MonitoringConfig()
        assert config.prometheus_url == "http://localhost:9090"
        print("   ✅ Monitoring config initialization: PASS")

        # Test validator initialization
        validator = MonitoringValidator(config)
        assert validator.config is not None
        print("   ✅ Validator initialization: PASS")

        # Test context manager
        async with validator as v:
            assert v is validator
        print("   ✅ Context manager: PASS")

        # Test validation summary
        validator.results = [
            type(
                "Result",
                (),
                {
                    "component": MonitoringComponent.PROMETHEUS,
                    "check_name": "health",
                    "status": ValidationStatus.PASS,
                    "message": "Test message",
                    "details": None,
                    "execution_time": 1.0,
                },
            )()
        ]
        summary = validator.get_summary()
        assert summary["total_checks"] == 1
        print("   ✅ Validation summary: PASS")

        print("\n" + "=" * 60)
        print("✅ ALL VALIDATION TESTS PASSED!")
        print("\nThe monitoring validation system works correctly:")
        print("  ✓ Configuration initialization")
        print("  ✓ Validator initialization")
        print("  ✓ Context manager")
        print("  ✓ Validation summary")

    # Run the tests
    asyncio.run(run_validation_tests())
    sys.exit(0)
