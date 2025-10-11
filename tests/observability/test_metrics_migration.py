"""Unit tests for MetricsMigrationHelper."""

from unittest.mock import Mock, patch

from Medical_KG_rev.config.settings import AppSettings, FeatureFlagSettings
from Medical_KG_rev.observability.metrics_migration import MetricsMigrationHelper


class TestMetricsMigrationHelper:
    """Test cases for MetricsMigrationHelper."""

    def test_init_with_settings(self) -> None:
        """Test initialization with settings."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = True

        helper = MetricsMigrationHelper(settings)

        assert helper.settings is settings
        assert helper.use_domain_registries is True

    def test_init_with_disabled_feature_flag(self) -> None:
        """Test initialization with disabled feature flag."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = False

        helper = MetricsMigrationHelper(settings)

        assert helper.settings is settings
        assert helper.use_domain_registries is False

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_gpu_hardware_metric_with_feature_flag_enabled(self, mock_factory) -> None:
        """Test recording GPU hardware metric with feature flag enabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = True

        mock_gpu_registry = Mock()
        mock_factory.return_value.get_gpu_registry.return_value = mock_gpu_registry

        helper = MetricsMigrationHelper(settings)
        helper.factory = mock_factory.return_value

        helper.record_gpu_hardware_metric("gpu-0", "RTX 4090", "memory_usage", 8192.0)

        mock_gpu_registry.set_memory_usage.assert_called_once_with("gpu-0", "RTX 4090", 8192)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_gpu_service_call_with_feature_flag_enabled(self, mock_factory) -> None:
        """Test recording GPU service call with feature flag enabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = True

        mock_gpu_registry = Mock()
        mock_factory.return_value.get_gpu_registry.return_value = mock_gpu_registry

        helper = MetricsMigrationHelper(settings)
        helper.factory = mock_factory.return_value

        helper.record_gpu_service_call("embedding", "GenerateEmbeddings", "success", 0.5)

        mock_gpu_registry.record_gpu_service_call.assert_called_once_with("embedding", "GenerateEmbeddings", "success")
        mock_gpu_registry.observe_gpu_service_duration.assert_called_once_with("embedding", "GenerateEmbeddings", 0.5)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_external_api_call_with_feature_flag_enabled(self, mock_factory) -> None:
        """Test recording external API call with feature flag enabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = True

        mock_external_api_registry = Mock()
        mock_factory.return_value.get_external_api_registry.return_value = mock_external_api_registry

        helper = MetricsMigrationHelper(settings)
        helper.factory = mock_factory.return_value

        helper.record_external_api_call("REST", "/v1/ingest", "POST", 200, 0.3)

        mock_external_api_registry.record_client_request.assert_called_once_with("REST", "/v1/ingest", "POST", 200)
        mock_external_api_registry.observe_request_duration.assert_called_once_with("REST", "/v1/ingest", 0.3)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_pipeline_stage_with_feature_flag_enabled(self, mock_factory) -> None:
        """Test recording pipeline stage with feature flag enabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = True

        mock_pipeline_registry = Mock()
        mock_factory.return_value.get_pipeline_registry.return_value = mock_pipeline_registry

        helper = MetricsMigrationHelper(settings)
        helper.factory = mock_factory.return_value

        helper.record_pipeline_stage("ingestion", "embedding", "completed", 2.5)

        mock_pipeline_registry.record_stage_execution.assert_called_once_with("ingestion", "embedding", "completed")
        mock_pipeline_registry.observe_stage_duration.assert_called_once_with("ingestion", "embedding", 2.5)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_cache_operation_with_feature_flag_enabled(self, mock_factory) -> None:
        """Test recording cache operation with feature flag enabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = True

        mock_cache_registry = Mock()
        mock_factory.return_value.get_cache_registry.return_value = mock_cache_registry

        helper = MetricsMigrationHelper(settings)
        helper.factory = mock_factory.return_value

        helper.record_cache_operation("redis", "get", "success", 0.001)

        mock_cache_registry.record_cache_operation.assert_called_once_with("redis", "get", "success")
        mock_cache_registry.observe_cache_operation_duration.assert_called_once_with("redis", "get", 0.001)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_reranking_operation_with_feature_flag_enabled(self, mock_factory) -> None:
        """Test recording reranking operation with feature flag enabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = True

        mock_reranking_registry = Mock()
        mock_factory.return_value.get_reranking_registry.return_value = mock_reranking_registry

        helper = MetricsMigrationHelper(settings)
        helper.factory = mock_factory.return_value

        helper.record_reranking_operation("lexical", "completed", 0.5)

        mock_reranking_registry.record_reranking_operation.assert_called_once_with("lexical", "completed")
        mock_reranking_registry.observe_reranking_duration.assert_called_once_with("lexical", 0.5)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_gpu_hardware_metric_with_feature_flag_disabled(self, mock_factory) -> None:
        """Test recording GPU hardware metric with feature flag disabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = False

        helper = MetricsMigrationHelper(settings)

        with patch('Medical_KG_rev.observability.metrics.GPU_MEMORY_USAGE_MB') as mock_gpu_memory:
            helper.record_gpu_hardware_metric("gpu-0", "RTX 4090", "memory_usage", 8192.0)
            mock_gpu_memory.labels.assert_called_once_with(device_id="gpu-0", device_name="RTX 4090")
            mock_gpu_memory.labels.return_value.set.assert_called_once_with(8192.0)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_gpu_service_call_with_feature_flag_disabled(self, mock_factory) -> None:
        """Test recording GPU service call with feature flag disabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = False

        helper = MetricsMigrationHelper(settings)

        with patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALLS_TOTAL') as mock_calls, \
             patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALL_DURATION_SECONDS') as mock_duration:
            helper.record_gpu_service_call("embedding", "GenerateEmbeddings", "success", 0.5)

            mock_calls.labels.assert_called_once_with(service="embedding", method="GenerateEmbeddings", status="success")
            mock_calls.labels.return_value.inc.assert_called_once()
            mock_duration.labels.assert_called_once_with(service="embedding", method="GenerateEmbeddings")
            mock_duration.labels.return_value.observe.assert_called_once_with(0.5)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_external_api_call_with_feature_flag_disabled(self, mock_factory) -> None:
        """Test recording external API call with feature flag disabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = False

        helper = MetricsMigrationHelper(settings)

        with patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALLS_TOTAL') as mock_calls, \
             patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALL_DURATION_SECONDS') as mock_duration:
            helper.record_external_api_call("REST", "/v1/ingest", "POST", 200, 0.3)

            mock_calls.labels.assert_called_once_with(service="gateway", method="REST_/v1/ingest", status="200")
            mock_calls.labels.return_value.inc.assert_called_once()
            mock_duration.labels.assert_called_once_with(service="gateway", method="REST_/v1/ingest")
            mock_duration.labels.return_value.observe.assert_called_once_with(0.3)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_pipeline_stage_with_feature_flag_disabled(self, mock_factory) -> None:
        """Test recording pipeline stage with feature flag disabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = False

        helper = MetricsMigrationHelper(settings)

        with patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALLS_TOTAL') as mock_calls, \
             patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALL_DURATION_SECONDS') as mock_duration:
            helper.record_pipeline_stage("ingestion", "embedding", "completed", 2.5)

            mock_calls.labels.assert_called_once_with(service="pipeline", method="embedding", status="completed")
            mock_calls.labels.return_value.inc.assert_called_once()
            mock_duration.labels.assert_called_once_with(service="pipeline", method="embedding")
            mock_duration.labels.return_value.observe.assert_called_once_with(2.5)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_cache_operation_with_feature_flag_disabled(self, mock_factory) -> None:
        """Test recording cache operation with feature flag disabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = False

        helper = MetricsMigrationHelper(settings)

        with patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALLS_TOTAL') as mock_calls, \
             patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALL_DURATION_SECONDS') as mock_duration:
            helper.record_cache_operation("redis", "get", "success", 0.001)

            mock_calls.labels.assert_called_once_with(service="cache", method="redis_get", status="success")
            mock_calls.labels.return_value.inc.assert_called_once()
            mock_duration.labels.assert_called_once_with(service="cache", method="redis_get")
            mock_duration.labels.return_value.observe.assert_called_once_with(0.001)

    @patch('Medical_KG_rev.observability.metrics_migration.get_metric_registry_factory')
    def test_record_reranking_operation_with_feature_flag_disabled(self, mock_factory) -> None:
        """Test recording reranking operation with feature flag disabled."""
        settings = Mock(spec=AppSettings)
        settings.feature_flags = Mock(spec=FeatureFlagSettings)
        settings.feature_flags.domain_specific_metric_registries = False

        helper = MetricsMigrationHelper(settings)

        with patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALLS_TOTAL') as mock_calls, \
             patch('Medical_KG_rev.observability.metrics.GPU_SERVICE_CALL_DURATION_SECONDS') as mock_duration:
            helper.record_reranking_operation("lexical", "completed", 0.5)

            mock_calls.labels.assert_called_once_with(service="reranking", method="lexical", status="completed")
            mock_calls.labels.return_value.inc.assert_called_once()
            mock_duration.labels.assert_called_once_with(service="reranking", method="lexical")
            mock_duration.labels.return_value.observe.assert_called_once_with(0.5)
