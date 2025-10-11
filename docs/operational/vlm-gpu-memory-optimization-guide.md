# VLM GPU Memory Optimization Guide

This guide provides comprehensive information about GPU memory optimization for VLM models in the Medical KG platform.

## Overview

The VLM GPU Memory Optimizer is a critical component that monitors and optimizes GPU memory usage for Vision-Language Models (VLM), ensuring efficient memory utilization and preventing Out-of-Memory (OOM) errors.

## Architecture

### Components

1. **VLMGPUMemoryOptimizer**: Core optimization engine
2. **MemoryConfig**: Configuration management
3. **MemoryMetrics**: GPU metrics collection
4. **MemoryOptimizationResult**: Optimization result tracking

### Key Features

- **Real-time Monitoring**: Continuous GPU memory usage monitoring
- **Adaptive Optimization**: Dynamic batch size adjustment based on memory usage
- **Memory Cleanup**: Automatic garbage collection and memory defragmentation
- **Performance Tracking**: Comprehensive metrics and optimization history
- **Multiple Strategies**: Conservative, balanced, aggressive, and adaptive optimization modes

## Configuration

### MemoryConfig Parameters

```python
@dataclass
class MemoryConfig:
    enabled: bool = True
    strategy: MemoryOptimizationStrategy = MemoryOptimizationStrategy.BALANCED
    monitoring_interval: int = 5  # 5 seconds
    memory_threshold_warning: float = 0.8  # 80% warning threshold
    memory_threshold_critical: float = 0.9  # 90% critical threshold
    memory_threshold_oom: float = 0.95  # 95% OOM threshold
    temperature_threshold: float = 80.0  # 80°C temperature threshold
    optimization_interval: int = 60  # 1 minute
    cleanup_interval: int = 300  # 5 minutes
    max_batch_size: int = 16
    min_batch_size: int = 1
    memory_reserve_mb: int = 1024  # 1GB reserve
    gc_threshold: float = 0.85  # 85% GC threshold
    fragmentation_threshold: float = 0.3  # 30% fragmentation threshold
```

### Optimization Strategies

#### Conservative Strategy

- Prioritizes stability over performance
- Uses 70% of calculated optimal batch size
- Suitable for production environments with strict memory constraints

#### Balanced Strategy (Default)

- Balances performance and memory usage
- Uses calculated optimal batch size
- Suitable for most use cases

#### Aggressive Strategy

- Maximizes performance
- Uses 120% of calculated optimal batch size
- Suitable for development and testing environments

#### Adaptive Strategy

- Adapts based on recent performance metrics
- Adjusts batch size based on GPU utilization
- Suitable for variable workload environments

## Usage

### Basic Usage

```python
from Medical_KG_rev.services.optimization.vlm_gpu_memory_optimizer import (
    VLMGPUMemoryOptimizer,
    MemoryConfig,
    MemoryOptimizationStrategy
)

# Create configuration
config = MemoryConfig(
    strategy=MemoryOptimizationStrategy.BALANCED,
    memory_threshold_warning=0.8,
    memory_threshold_critical=0.9,
    memory_threshold_oom=0.95
)

# Create optimizer
optimizer = VLMGPUMemoryOptimizer(config)

# Start monitoring
await optimizer.start_monitoring()

# Perform optimization
result = await optimizer.optimize_memory()

# Get current batch size
batch_size = optimizer.get_current_batch_size()

# Get memory status
status = optimizer.get_memory_status()
```

### CLI Usage

The `manage_vlm_gpu_memory.py` script provides a comprehensive CLI interface:

```bash
# Show current status
python scripts/manage_vlm_gpu_memory.py status

# Monitor in real-time
python scripts/manage_vlm_gpu_memory.py monitor --duration 300 --interval 5

# Perform immediate optimization
python scripts/manage_vlm_gpu_memory.py optimize

# Set batch size manually
python scripts/manage_vlm_gpu_memory.py set-batch-size 8

# Enable/disable optimization
python scripts/manage_vlm_gpu_memory.py enable
python scripts/manage_vlm_gpu_memory.py disable

# Start/stop background monitoring
python scripts/manage_vlm_gpu_memory.py start-monitoring
python scripts/manage_vlm_gpu_memory.py stop-monitoring

# Configure optimization settings
python scripts/manage_vlm_gpu_memory.py configure \
    --strategy balanced \
    --warning-threshold 0.8 \
    --critical-threshold 0.9 \
    --oom-threshold 0.95 \
    --max-batch-size 16 \
    --min-batch-size 1

# Export/import configuration
python scripts/manage_vlm_gpu_memory.py export-config --output config.json
python scripts/manage_vlm_gpu_memory.py import-config config.json

# Run benchmark
python scripts/manage_vlm_gpu_memory.py benchmark --duration 300 --output results.json
```

## Monitoring

### Metrics

The optimizer exposes the following Prometheus metrics:

- `vlm_memory_usage_percent`: GPU memory usage percentage
- `vlm_memory_total_bytes`: Total GPU memory in bytes
- `vlm_memory_used_bytes`: Used GPU memory in bytes
- `vlm_memory_free_bytes`: Free GPU memory in bytes
- `vlm_gpu_utilization_percent`: GPU utilization percentage
- `vlm_gpu_temperature_celsius`: GPU temperature in Celsius
- `vlm_gpu_power_usage_watts`: GPU power usage in watts
- `vlm_memory_fragmentation_percent`: Memory fragmentation percentage
- `vlm_current_batch_size`: Current batch size
- `vlm_optimal_batch_size`: Optimal batch size
- `vlm_memory_optimizations_total`: Total memory optimizations
- `vlm_memory_optimizations_success_total`: Successful optimizations
- `vlm_memory_cleanups_total`: Total memory cleanups
- `vlm_batch_size_optimizations_total`: Total batch size optimizations
- `vlm_memory_freed_bytes_total`: Total memory freed in bytes

### Grafana Dashboard

The VLM GPU Memory Optimization dashboard provides:

- **GPU Memory Usage**: Real-time memory usage percentage
- **GPU Memory Details**: Total, used, and free memory
- **GPU Utilization**: GPU utilization over time
- **GPU Temperature**: GPU temperature monitoring
- **Batch Size Optimization**: Current vs optimal batch size
- **Memory Optimization Actions**: Optimization rate and success
- **Memory Fragmentation**: Fragmentation percentage
- **Memory Optimization Success Rate**: Success rate percentage
- **Memory Freed Over Time**: Memory freed rate

### Alerts

The following alerts are configured:

- **VLMMemoryOptimizerDown**: Optimizer service is down
- **VLMMemoryUsageHigh**: Memory usage > 90% for 2 minutes
- **VLMMemoryUsageCritical**: Memory usage > 95% for 1 minute
- **VLMMemoryOptimizationFailed**: Optimization failures detected
- **VLMMemoryFragmentationHigh**: Fragmentation > 30% for 5 minutes
- **VLMBatchSizeOptimizationActive**: Batch size optimization active

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t medical-kg/vlm-gpu-memory-optimizer:latest \
    -f ops/docker/vlm-gpu-memory-optimizer/Dockerfile .

# Run container
docker run -d \
    --name vlm-gpu-memory-optimizer \
    --gpus all \
    -p 8080:8080 \
    -p 8081:8081 \
    -v /var/log:/var/log \
    medical-kg/vlm-gpu-memory-optimizer:latest
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f ops/k8s/vlm-gpu-memory-optimizer.yaml

# Check deployment status
kubectl get pods -n medical-kg -l app=vlm-gpu-memory-optimizer

# Check logs
kubectl logs -n medical-kg -l app=vlm-gpu-memory-optimizer -f

# Check metrics
kubectl port-forward -n medical-kg svc/vlm-gpu-memory-optimizer 8080:8080
curl http://localhost:8080/metrics
```

### Health Checks

The optimizer provides health check endpoints:

- **Health Check**: `GET /health` - Basic health status
- **Readiness Check**: `GET /ready` - Readiness status
- **Metrics**: `GET /metrics` - Prometheus metrics

## Performance Optimization

### Best Practices

1. **Monitor Continuously**: Enable background monitoring for production environments
2. **Set Appropriate Thresholds**: Configure thresholds based on your GPU memory capacity
3. **Use Adaptive Strategy**: Use adaptive strategy for variable workloads
4. **Regular Cleanup**: Ensure regular memory cleanup is enabled
5. **Batch Size Limits**: Set appropriate min/max batch size limits

### Tuning Guidelines

#### Memory Thresholds

- **Warning**: 80% - Start monitoring closely
- **Critical**: 90% - Begin aggressive optimization
- **OOM**: 95% - Emergency cleanup and batch size reduction

#### Batch Size Optimization

- **Min Batch Size**: 1 - Minimum for processing
- **Max Batch Size**: 16 - Maximum based on GPU memory
- **Reserve Memory**: 1GB - Reserve for system operations

#### Monitoring Intervals

- **Monitoring**: 5 seconds - Real-time monitoring
- **Optimization**: 60 seconds - Optimization frequency
- **Cleanup**: 300 seconds - Cleanup frequency

## Troubleshooting

### Common Issues

#### High Memory Usage

- **Symptoms**: Memory usage consistently > 90%
- **Causes**: Insufficient batch size optimization, memory leaks
- **Solutions**:
  - Reduce max batch size
  - Increase cleanup frequency
  - Check for memory leaks in VLM models

#### Optimization Failures

- **Symptoms**: High optimization failure rate
- **Causes**: GPU driver issues, insufficient permissions
- **Solutions**:
  - Check GPU driver status
  - Verify NVIDIA permissions
  - Review optimization logs

#### Batch Size Oscillation

- **Symptoms**: Batch size constantly changing
- **Causes**: Unstable memory usage patterns
- **Solutions**:
  - Use conservative strategy
  - Increase optimization interval
  - Stabilize workload patterns

### Debugging

#### Enable Debug Logging

```python
import logging
logging.getLogger("Medical_KG_rev.services.optimization.vlm_gpu_memory_optimizer").setLevel(logging.DEBUG)
```

#### Check GPU Status

```bash
# Check GPU status
nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Check GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv
```

#### Review Optimization History

```python
# Get optimization history
history = optimizer.get_optimization_history()
for entry in history:
    print(f"Action: {entry['action_taken']}, Success: {entry['success']}")
```

## Integration

### VLM Service Integration

The optimizer integrates with VLM services to provide:

- **Dynamic Batch Size**: Automatic batch size adjustment
- **Memory Monitoring**: Real-time memory usage tracking
- **Optimization Triggers**: Automatic optimization based on thresholds
- **Performance Metrics**: Comprehensive performance tracking

### Service Discovery

The optimizer uses service discovery to:

- **Locate VLM Services**: Find VLM services in the cluster
- **Monitor Service Health**: Track service health status
- **Coordinate Optimization**: Coordinate optimization across services

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

### Data Protection

- **No Sensitive Data**: No sensitive data stored or transmitted
- **Encrypted Communication**: All communication encrypted
- **Audit Logging**: Comprehensive audit logging

## Compliance

### Regulatory Compliance

The optimizer maintains compliance with:

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
5. **Test Optimization**: Regular optimization testing

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

The VLM GPU Memory Optimizer is a critical component for ensuring efficient GPU memory usage in VLM processing. By following this guide, you can effectively monitor, optimize, and maintain GPU memory usage for optimal VLM performance.

For additional support or questions, refer to the comprehensive documentation or contact the development team.
