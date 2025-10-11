# Monitoring Validation Guide

This guide provides comprehensive information about monitoring validation for the torch isolation architecture.

## Overview

The monitoring validation system ensures that all monitoring components are properly configured and functioning correctly. This includes:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboards and visualization
- **Alertmanager**: Alerting rules and notifications
- **GPU Metrics Exporter**: GPU-specific metrics collection
- **Custom Metrics Adapter**: Kubernetes custom metrics
- **Service Health Checks**: gRPC service health monitoring

## Components

### Prometheus Validation

Validates Prometheus configuration and metrics collection:

- **Health Check**: Verifies Prometheus is accessible and healthy
- **Configuration**: Validates Prometheus configuration
- **Required Metrics**: Checks for essential metrics:
  - `gpu_service_calls_total`
  - `gpu_service_call_duration_seconds`
  - `gpu_service_errors_total`
  - `gpu_memory_usage_mb`
  - `gpu_service_health_status`
  - `circuit_breaker_state`

### Grafana Validation

Validates Grafana configuration and dashboards:

- **Health Check**: Verifies Grafana is accessible and healthy
- **Data Sources**: Checks configured data sources
- **Required Dashboards**: Validates essential dashboards:
  - `gpu-services-dashboard`
  - `service-architecture-dashboard`
  - `auto-scaling-dashboard`
  - `gpu-optimization-dashboard`
  - `cache-monitoring-dashboard`

### Alertmanager Validation

Validates Alertmanager configuration and alerting rules:

- **Health Check**: Verifies Alertmanager is accessible and healthy
- **Configuration**: Validates Alertmanager configuration
- **Required Alerts**: Checks for essential alert rules:
  - `GPUServiceFailures`
  - `CircuitBreakerOpen`
  - `HighGPUServiceLatency`
  - `HighGPUMemoryUsage`

### GPU Metrics Exporter Validation

Validates GPU metrics collection:

- **Health Check**: Verifies GPU metrics exporter is accessible
- **Required Metrics**: Checks for GPU-specific metrics:
  - `gpu_utilization_percentage`
  - `gpu_memory_usage_mb`
  - `gpu_temperature_celsius`
  - `gpu_power_usage_watts`

### Custom Metrics Adapter Validation

Validates Kubernetes custom metrics:

- **Health Check**: Verifies custom metrics adapter is accessible
- **Required Metrics**: Checks for custom metrics:
  - `gpu_utilization_percentage`
  - `gpu_memory_usage_mb`

### Service Health Checks Validation

Validates gRPC service health:

- **Health Endpoints**: Checks health endpoints for all services:
  - `gpu-management-service`
  - `embedding-service`
  - `reranking-service`
  - `docling-vlm-service`

## Usage

### Command Line Interface

The monitoring validation can be run using the CLI script:

```bash
# Validate all monitoring systems
python scripts/validate_monitoring.py validate-all

# Validate specific components
python scripts/validate_monitoring.py validate-prometheus
python scripts/validate_monitoring.py validate-grafana
python scripts/validate_monitoring.py validate-services

# Generate validation report
python scripts/validate_monitoring.py generate-report results.json
```

### Programmatic Usage

```python
from Medical_KG_rev.services.monitoring.monitoring_validator import (
    MonitoringValidator, MonitoringConfig
)

# Create configuration
config = MonitoringConfig(
    prometheus_url="http://localhost:9090",
    grafana_url="http://localhost:3000",
    alertmanager_url="http://localhost:9093"
)

# Run validation
async with MonitoringValidator(config) as validator:
    results = await validator.validate_all()
    summary = validator.get_summary()

    print(f"Total checks: {summary['total_checks']}")
    print(f"Passed: {summary['passed_checks']}")
    print(f"Failed: {summary['failed_checks']}")
```

## Configuration

### MonitoringConfig Parameters

- `prometheus_url`: Prometheus server URL (default: `http://localhost:9090`)
- `grafana_url`: Grafana server URL (default: `http://localhost:3000`)
- `alertmanager_url`: Alertmanager server URL (default: `http://localhost:9093`)
- `gpu_metrics_exporter_url`: GPU metrics exporter URL (default: `http://localhost:8080`)
- `custom_metrics_adapter_url`: Custom metrics adapter URL (default: `http://localhost:8081`)
- `service_urls`: Dictionary of service names to URLs
- `timeout`: Request timeout in seconds (default: 30)
- `retry_attempts`: Number of retry attempts (default: 3)
- `retry_delay`: Delay between retries in seconds (default: 1.0)

### Service URLs Configuration

```python
service_urls = {
    "gpu-management": "http://localhost:50051",
    "embedding-service": "http://localhost:50052",
    "reranking-service": "http://localhost:50053",
    "docling-vlm-service": "http://localhost:50054"
}
```

## Validation Results

### ValidationStatus Values

- `PASS`: Validation check passed successfully
- `FAIL`: Validation check failed
- `WARNING`: Validation check passed with warnings
- `SKIP`: Validation check was skipped

### ValidationResult Structure

```python
@dataclass
class ValidationResult:
    component: MonitoringComponent
    check_name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
```

### Summary Structure

```python
{
    "total_checks": int,
    "passed_checks": int,
    "failed_checks": int,
    "warning_checks": int,
    "success_rate": float,
    "results": List[ValidationResult]
}
```

## Error Handling

The monitoring validation system includes comprehensive error handling:

- **Connection Errors**: Handles network connectivity issues
- **Timeout Errors**: Manages request timeouts
- **HTTP Errors**: Processes HTTP status codes
- **JSON Parsing Errors**: Handles malformed JSON responses
- **Service Unavailable**: Manages service downtime

## Integration Testing

The monitoring validation system includes integration tests:

```python
# Run integration tests
python -m pytest tests/validation/test_monitoring_validation.py -v
```

### Test Coverage

- Configuration initialization
- Validator initialization
- Context manager functionality
- Component-specific validation
- Error handling
- Timeout handling
- Summary generation

## Troubleshooting

### Common Issues

1. **Service Unavailable**: Ensure all monitoring services are running
2. **Connection Refused**: Check service URLs and ports
3. **Authentication Errors**: Verify credentials and access tokens
4. **Timeout Errors**: Increase timeout values for slow services
5. **Metric Not Found**: Ensure metrics are being collected

### Debug Mode

Enable verbose output for detailed debugging:

```bash
python scripts/validate_monitoring.py validate-all --verbose
```

### Log Analysis

Check service logs for detailed error information:

```bash
# Prometheus logs
kubectl logs -n monitoring prometheus-server

# Grafana logs
kubectl logs -n monitoring grafana

# Alertmanager logs
kubectl logs -n monitoring alertmanager
```

## Best Practices

### Regular Validation

Run monitoring validation regularly to ensure system health:

```bash
# Daily validation
0 6 * * * /usr/bin/python3 /path/to/scripts/validate_monitoring.py validate-all

# Weekly comprehensive validation
0 2 * * 0 /usr/bin/python3 /path/to/scripts/validate_monitoring.py validate-all --verbose
```

### Alerting Integration

Integrate validation results with alerting systems:

```python
# Check validation results and send alerts
if summary["failed_checks"] > 0:
    send_alert("Monitoring validation failed", summary)
```

### Performance Monitoring

Monitor validation performance:

```python
# Track validation execution time
start_time = time.time()
results = await validator.validate_all()
execution_time = time.time() - start_time

# Log performance metrics
logger.info(f"Validation completed in {execution_time:.2f} seconds")
```

## Security Considerations

### Authentication

Ensure proper authentication for monitoring services:

- Use service accounts for Kubernetes services
- Implement mTLS for service-to-service communication
- Use API keys for external service access

### Network Security

Secure network communication:

- Use HTTPS for external service access
- Implement network policies for Kubernetes
- Use VPN for remote access

### Data Privacy

Protect sensitive monitoring data:

- Encrypt metrics data at rest
- Use secure communication protocols
- Implement data retention policies

## Compliance

### Regulatory Requirements

Ensure monitoring validation meets regulatory requirements:

- **HIPAA**: Healthcare data protection
- **SOC2**: Security and availability
- **GDPR**: Data privacy and protection
- **ISO27001**: Information security management
- **NIST**: Cybersecurity framework

### Audit Trail

Maintain comprehensive audit trails:

```python
# Log validation results
audit_logger.info("Monitoring validation completed", {
    "timestamp": datetime.utcnow(),
    "results": summary,
    "user": current_user
})
```

## Future Enhancements

### Planned Features

- **Real-time Validation**: Continuous monitoring validation
- **Predictive Analytics**: Anomaly detection and prediction
- **Automated Remediation**: Self-healing monitoring systems
- **Multi-cluster Support**: Cross-cluster validation
- **Custom Validators**: Extensible validation framework

### Integration Roadmap

- **CI/CD Integration**: Automated validation in pipelines
- **Kubernetes Operators**: Native Kubernetes integration
- **Cloud Provider Integration**: AWS, GCP, Azure support
- **Third-party Tools**: Integration with external monitoring tools

## Support

### Documentation

- [Architecture Overview](../architecture/overview.md)
- [Operational Runbook](../operational-runbook.md)
- [Developer Guide](../guides/developer_guide.md)

### Community

- [GitHub Issues](https://github.com/your-org/medical-kg/issues)
- [Discord Channel](https://discord.gg/your-channel)
- [Mailing List](mailto:dev@your-org.com)

### Professional Support

For enterprise support and consulting:

- **Email**: <support@your-org.com>
- **Phone**: +1-555-0123
- **Website**: <https://your-org.com/support>
