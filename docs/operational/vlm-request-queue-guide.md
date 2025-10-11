# VLM Request Queue Guide

This guide provides comprehensive information about the VLM request queue system in the Medical KG platform.

## Overview

The VLM Request Queue is a critical component that manages and processes VLM (Vision-Language Model) requests under high load conditions, ensuring efficient request handling, prioritization, and resource management.

## Architecture

### Components

1. **VLMRequestQueue**: Core queue management engine
2. **VLMRequest**: Individual request representation
3. **QueueConfig**: Configuration management
4. **QueueMetrics**: Performance metrics collection

### Key Features

- **Request Prioritization**: Support for multiple priority levels (low, normal, high, critical)
- **Queue Strategies**: FIFO, priority-based, round-robin, and weighted processing
- **Concurrent Processing**: Configurable maximum concurrent request processing
- **Retry Logic**: Automatic retry with exponential backoff
- **Timeout Management**: Request timeout handling and cleanup
- **Performance Monitoring**: Comprehensive metrics and monitoring
- **Health Checks**: Continuous health monitoring and cleanup

## Configuration

### QueueConfig Parameters

```python
@dataclass
class QueueConfig:
    max_queue_size: int = 1000
    max_concurrent_requests: int = 10
    strategy: QueueStrategy = QueueStrategy.PRIORITY
    default_timeout: float = 300.0
    default_priority: RequestPriority = RequestPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 30.0
    cleanup_interval: float = 300.0
    enable_metrics: bool = True
    enable_tracing: bool = True
```

### Queue Strategies

#### FIFO Strategy

- **Description**: First In, First Out processing
- **Use Case**: Simple request processing without prioritization
- **Advantages**: Predictable processing order
- **Disadvantages**: No priority handling

#### Priority Strategy (Default)

- **Description**: Priority-based processing
- **Use Case**: Critical requests need faster processing
- **Advantages**: Important requests processed first
- **Disadvantages**: Starvation of low-priority requests

#### Round Robin Strategy

- **Description**: Equal processing time for all requests
- **Use Case**: Fair resource allocation
- **Advantages**: Prevents starvation
- **Disadvantages**: No priority handling

#### Weighted Strategy

- **Description**: Weighted processing based on request characteristics
- **Use Case**: Complex processing requirements
- **Advantages**: Flexible processing control
- **Disadvantages**: Complex configuration

### Request Priorities

#### Low Priority

- **Use Case**: Background processing, non-urgent requests
- **Processing**: Processed when resources are available
- **Timeout**: Standard timeout

#### Normal Priority (Default)

- **Use Case**: Standard request processing
- **Processing**: Processed in order
- **Timeout**: Standard timeout

#### High Priority

- **Use Case**: Important requests requiring faster processing
- **Processing**: Processed before normal priority
- **Timeout**: Extended timeout

#### Critical Priority

- **Use Case**: Emergency or system-critical requests
- **Processing**: Processed immediately
- **Timeout**: Maximum timeout

## Usage

### Basic Usage

```python
from Medical_KG_rev.services.queuing.vlm_request_queue import (
    VLMRequestQueue,
    QueueConfig,
    RequestPriority
)

# Create configuration
config = QueueConfig(
    max_queue_size=1000,
    max_concurrent_requests=10,
    strategy=QueueStrategy.PRIORITY
)

# Create queue
queue = VLMRequestQueue(config)

# Start queue
await queue.start()

# Submit request
request_id = await queue.submit_request(
    pdf_content=pdf_bytes,
    config=processing_config,
    options=processing_options,
    priority=RequestPriority.HIGH
)

# Check request status
status = await queue.get_request_status(request_id)

# Stop queue
await queue.stop()
```

### CLI Usage

The `manage_vlm_request_queue.py` script provides a comprehensive CLI interface:

```bash
# Show current status
python scripts/manage_vlm_request_queue.py status

# Start/stop queue
python scripts/manage_vlm_request_queue.py start
python scripts/manage_vlm_request_queue.py stop

# Submit request
python scripts/manage_vlm_request_queue.py submit document.pdf \
    --priority high \
    --timeout 600 \
    --max-retries 5

# Check request status
python scripts/manage_vlm_request_queue.py request-status <request-id>

# Cancel request
python scripts/manage_vlm_request_queue.py cancel <request-id>

# Monitor queue in real-time
python scripts/manage_vlm_request_queue.py monitor --duration 300 --interval 5

# Configure queue settings
python scripts/manage_vlm_request_queue.py configure \
    --strategy priority \
    --max-queue-size 2000 \
    --max-concurrent 20 \
    --default-timeout 600

# Export/import configuration
python scripts/manage_vlm_request_queue.py export-config --output config.json
python scripts/manage_vlm_request_queue.py import-config config.json

# Run benchmark
python scripts/manage_vlm_request_queue.py benchmark \
    --duration 600 \
    --request-rate 20 \
    --output results.json
```

## Monitoring

### Metrics

The queue exposes the following Prometheus metrics:

- `vlm_queue_size`: Current queue size
- `vlm_max_queue_size`: Maximum queue size
- `vlm_active_requests`: Number of active requests
- `vlm_max_concurrent_requests`: Maximum concurrent requests
- `vlm_completed_requests_total`: Total completed requests
- `vlm_failed_requests_total`: Total failed requests
- `vlm_cancelled_requests_total`: Total cancelled requests
- `vlm_timeout_requests_total`: Total timeout requests
- `vlm_retry_requests_total`: Total retry requests
- `vlm_average_processing_time_seconds`: Average processing time
- `vlm_queue_utilization`: Queue utilization percentage
- `vlm_error_rate`: Error rate percentage
- `vlm_throughput`: Throughput (requests per minute)

### Grafana Dashboard

The VLM Request Queue dashboard provides:

- **Queue Size**: Current queue size and utilization
- **Active Requests**: Number of currently processing requests
- **Request Status Distribution**: Pie chart of request statuses
- **Request Throughput**: Completed and failed requests per minute
- **Average Processing Time**: Processing time trends
- **Error Rate**: Error rate over time
- **Queue Utilization**: Queue utilization percentage
- **Retry Rate**: Retry rate over time

### Alerts

The following alerts are configured:

- **VLMRequestQueueDown**: Queue service is down
- **VLMRequestQueueFull**: Queue is nearly full (>90%)
- **VLMRequestQueueOverflow**: Queue is at maximum capacity
- **VLMRequestQueueHighErrorRate**: Error rate >10% for 5 minutes
- **VLMRequestQueueLowThroughput**: Throughput <1 req/min for 10 minutes
- **VLMRequestQueueHighProcessingTime**: Processing time >300s for 5 minutes
- **VLMRequestQueueTimeoutRequests**: Timeout requests detected
- **VLMRequestQueueRetryRate**: Retry rate >0.1 retries/sec for 5 minutes

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t medical-kg/vlm-request-queue:latest \
    -f ops/docker/vlm-request-queue/Dockerfile .

# Run container
docker run -d \
    --name vlm-request-queue \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 8082:8082 \
    -v /var/log:/var/log \
    medical-kg/vlm-request-queue:latest
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f ops/k8s/vlm-request-queue.yaml

# Check deployment status
kubectl get pods -n medical-kg -l app=vlm-request-queue

# Check logs
kubectl logs -n medical-kg -l app=vlm-request-queue -f

# Check metrics
kubectl port-forward -n medical-kg svc/vlm-request-queue 8080:8080
curl http://localhost:8080/metrics
```

### Health Checks

The queue provides health check endpoints:

- **Health Check**: `GET /health` - Basic health status
- **Readiness Check**: `GET /ready` - Readiness status
- **Metrics**: `GET /metrics` - Prometheus metrics
- **API**: `GET /api/v1/status` - Queue status API

## Performance Optimization

### Best Practices

1. **Monitor Queue Size**: Keep queue size below 80% capacity
2. **Tune Concurrent Requests**: Adjust based on processing capacity
3. **Set Appropriate Timeouts**: Balance between processing time and resource usage
4. **Use Priority Queues**: Implement priority-based processing for critical requests
5. **Monitor Error Rates**: Keep error rates below 5%
6. **Regular Cleanup**: Enable automatic cleanup of old requests

### Tuning Guidelines

#### Queue Size

- **Small Queue (100-500)**: Development and testing
- **Medium Queue (500-1000)**: Production with moderate load
- **Large Queue (1000-5000)**: High-load production environments

#### Concurrent Requests

- **Low Concurrency (1-5)**: Resource-constrained environments
- **Medium Concurrency (5-15)**: Standard production environments
- **High Concurrency (15-50)**: High-performance environments

#### Timeout Settings

- **Short Timeout (60-180s)**: Fast processing requirements
- **Medium Timeout (180-300s)**: Standard processing
- **Long Timeout (300-600s)**: Complex processing requirements

#### Retry Settings

- **No Retries (0)**: Critical systems where failures should be immediate
- **Low Retries (1-2)**: Standard error handling
- **High Retries (3-5)**: Resilient systems with transient failures

## Troubleshooting

### Common Issues

#### Queue Full

- **Symptoms**: Requests rejected with "queue full" error
- **Causes**: Insufficient processing capacity, high request rate
- **Solutions**:
  - Increase queue size
  - Increase concurrent processing
  - Implement request throttling
  - Add more processing resources

#### High Error Rate

- **Symptoms**: Error rate >10%
- **Causes**: Processing failures, resource constraints
- **Solutions**:
  - Check processing service health
  - Increase timeout values
  - Improve error handling
  - Add more processing resources

#### Slow Processing

- **Symptoms**: High processing times, low throughput
- **Causes**: Resource constraints, inefficient processing
- **Solutions**:
  - Increase concurrent processing
  - Optimize processing algorithms
  - Add more processing resources
  - Implement request prioritization

#### Memory Issues

- **Symptoms**: High memory usage, OOM errors
- **Causes**: Large requests, memory leaks
- **Solutions**:
  - Implement request size limits
  - Add memory monitoring
  - Optimize memory usage
  - Implement request streaming

### Debugging

#### Enable Debug Logging

```python
import logging
logging.getLogger("Medical_KG_rev.services.queuing.vlm_request_queue").setLevel(logging.DEBUG)
```

#### Check Queue Status

```python
# Get queue status
status = queue.get_queue_status()
print(f"Queue size: {status['queue_size']}")
print(f"Active requests: {status['active_requests']}")
print(f"Error rate: {status['metrics']['error_rate']}")
```

#### Monitor Request Processing

```python
# Get request status
status = await queue.get_request_status(request_id)
print(f"Request status: {status['status']}")
print(f"Processing time: {status['processing_time']}")
print(f"Retry count: {status['retry_count']}")
```

#### Review Metrics History

```python
# Get metrics history
history = queue.get_metrics_history()
for entry in history[-10:]:  # Last 10 entries
    print(f"Timestamp: {entry['timestamp']}")
    print(f"Throughput: {entry['throughput']}")
    print(f"Error rate: {entry['error_rate']}")
```

## Integration

### VLM Service Integration

The queue integrates with VLM services to provide:

- **Request Management**: Centralized request handling
- **Load Balancing**: Distributed request processing
- **Priority Processing**: Critical request prioritization
- **Error Handling**: Comprehensive error management
- **Performance Monitoring**: Real-time performance tracking

### Service Discovery

The queue uses service discovery to:

- **Locate VLM Services**: Find available VLM processing services
- **Monitor Service Health**: Track service health status
- **Load Balance Requests**: Distribute requests across services
- **Handle Failures**: Route requests to healthy services

### Configuration Management

Configuration is managed through:

- **Environment Variables**: Runtime configuration
- **ConfigMaps**: Kubernetes configuration
- **CLI Parameters**: Command-line configuration
- **JSON Files**: File-based configuration

## Security

### Access Control

- **Service Account**: Dedicated Kubernetes service account
- **RBAC**: Role-based access control for Kubernetes resources
- **Network Policies**: Network isolation for security
- **API Authentication**: Secure API access

### Data Protection

- **Request Encryption**: Encrypt sensitive request data
- **Secure Communication**: All communication encrypted
- **Audit Logging**: Comprehensive audit logging
- **Data Retention**: Automatic data cleanup

## Compliance

### Regulatory Compliance

The queue maintains compliance with:

- **HIPAA**: Healthcare data protection
- **SOC2**: Security and availability
- **GDPR**: Data protection and privacy
- **ISO27001**: Information security management
- **NIST**: Cybersecurity framework

### Audit Requirements

- **Comprehensive Logging**: All operations logged
- **Performance Tracking**: Performance metrics recorded
- **Change Management**: Configuration changes tracked
- **Incident Response**: Incident response procedures

## Maintenance

### Regular Maintenance

1. **Monitor Performance**: Regular performance monitoring
2. **Update Configuration**: Update configuration as needed
3. **Review Logs**: Regular log review and analysis
4. **Update Dependencies**: Keep dependencies updated
5. **Test Queue**: Regular queue functionality testing

### Backup and Recovery

- **Configuration Backup**: Regular configuration backup
- **Metrics Backup**: Historical metrics preservation
- **Recovery Procedures**: Documented recovery procedures
- **Disaster Recovery**: Disaster recovery planning

## Support

### Getting Help

- **Documentation**: Comprehensive documentation available
- **Logs**: Detailed logging for troubleshooting
- **Metrics**: Rich metrics for monitoring
- **Community**: Community support available

### Reporting Issues

When reporting issues, include:

- **Configuration**: Current configuration settings
- **Logs**: Relevant log entries
- **Metrics**: Performance metrics
- **Environment**: Environment details
- **Reproduction Steps**: Steps to reproduce the issue

## Conclusion

The VLM Request Queue is a critical component for managing VLM processing requests under high load conditions. By following this guide, you can effectively configure, monitor, and maintain the queue for optimal performance.

For additional support or questions, refer to the comprehensive documentation or contact the development team.
