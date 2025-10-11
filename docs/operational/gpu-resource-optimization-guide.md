# GPU Resource Optimization Guide

This guide provides comprehensive information about GPU resource optimization for the Medical KG platform's GPU services.

## Overview

The GPU resource optimization system monitors, allocates, and optimizes GPU resources across all GPU services to ensure optimal performance and efficient resource utilization.

## Architecture Components

### 1. GPU Resource Monitor

The GPU Resource Monitor provides real-time monitoring of GPU resources:

- **GPU Utilization**: Tracks GPU compute utilization percentage
- **Memory Usage**: Monitors GPU memory usage and availability
- **Temperature**: Tracks GPU temperature for thermal management
- **Power Usage**: Monitors GPU power consumption
- **Performance Metrics**: Tracks throughput and latency metrics

### 2. GPU Allocation Manager

The GPU Allocation Manager handles dynamic GPU allocation:

- **Resource Allocation**: Allocates GPUs based on service requirements
- **Memory Management**: Ensures sufficient memory availability
- **Load Balancing**: Distributes load across available GPUs
- **Allocation Tracking**: Maintains allocation state and history

### 3. GPU Memory Optimizer

The GPU Memory Optimizer optimizes memory usage:

- **Memory Cleanup**: Frees unused memory when thresholds are exceeded
- **Memory Defragmentation**: Optimizes memory layout for better utilization
- **Cache Management**: Manages GPU memory caches efficiently
- **Memory Monitoring**: Tracks memory usage patterns and trends

## Configuration

### Optimization Settings

```yaml
optimization:
  interval: 300                    # Optimization interval in seconds
  memory_threshold: 0.85          # Memory usage threshold (85%)
  allocation_timeout: 30          # Allocation timeout in seconds
  deallocation_timeout: 10        # Deallocation timeout in seconds
```

### Service-Specific Settings

```yaml
services:
  embedding_service:
    min_memory_mb: 2048           # Minimum required memory
    max_memory_mb: 8192           # Maximum allowed memory
    preferred_device: null         # Preferred GPU device

  reranking_service:
    min_memory_mb: 1024
    max_memory_mb: 4096
    preferred_device: null

  docling_vlm_service:
    min_memory_mb: 4096
    max_memory_mb: 16384
    preferred_device: null
```

## Deployment

### Prerequisites

- Kubernetes cluster with GPU nodes
- NVIDIA GPU drivers installed
- NVIDIA Container Toolkit configured
- Prometheus monitoring stack

### Deploy GPU Resource Optimizer

```bash
# Deploy the GPU resource optimizer
kubectl apply -f ops/k8s/gpu-resource-optimizer.yaml

# Check deployment status
kubectl get pods -l app=gpu-resource-optimizer -n medical-kg

# View logs
kubectl logs -l app=gpu-resource-optimizer -n medical-kg
```

### Verify Deployment

```bash
# Check if optimizer is running
kubectl get pods -l app=gpu-resource-optimizer -n medical-kg

# Check service status
kubectl get svc gpu-resource-optimizer -n medical-kg

# Check metrics endpoint
kubectl port-forward svc/gpu-resource-optimizer 8000:8000 -n medical-kg
curl http://localhost:8000/metrics
```

## Usage

### Command-Line Interface

The GPU resource optimizer provides a comprehensive CLI:

```bash
# Show current GPU status
python scripts/optimize_gpu_resources.py status

# Optimize memory for a specific service
python scripts/optimize_gpu_resources.py optimize embedding-service

# Allocate GPU for a service
python scripts/optimize_gpu_resources.py allocate embedding-service 4096

# Deallocate GPU from a service
python scripts/optimize_gpu_resources.py deallocate embedding-service 0

# Run continuous optimization loop
python scripts/optimize_gpu_resources.py loop --interval 300

# Export status to JSON
python scripts/optimize_gpu_resources.py export status.json

# Generate comprehensive report
python scripts/optimize_gpu_resources.py report
```

### Programmatic Usage

```python
from src.Medical_KG_rev.services.optimization.gpu_resource_optimizer import GPUResourceOptimizer

# Create optimizer instance
optimizer = GPUResourceOptimizer()

# Get optimization status
status = optimizer.get_optimization_status()

# Optimize memory for a service
results = await optimizer.memory_optimizer.optimize_memory_usage('embedding-service')

# Allocate GPU for a service
device_id = await optimizer.allocation_manager.allocate_gpu(
    'embedding-service',
    required_memory_mb=4096
)

# Start optimization loop
await optimizer.start_optimization_loop(interval=300)
```

## Monitoring

### Grafana Dashboard

The GPU optimization dashboard provides comprehensive monitoring:

- GPU memory utilization trends
- GPU temperature and power usage
- Allocation request rates and duration
- GPU throughput and latency metrics
- Memory usage statistics

### Key Metrics

#### GPU Resource Metrics

- `gpu_utilization_percent`: GPU compute utilization
- `gpu_memory_usage_mb`: GPU memory usage in MB
- `gpu_memory_total_mb`: Total GPU memory in MB
- `gpu_memory_free_mb`: Free GPU memory in MB
- `gpu_temperature_celsius`: GPU temperature
- `gpu_power_usage_watts`: GPU power consumption

#### Performance Metrics

- `gpu_throughput_ops_total`: Total GPU operations
- `gpu_operation_latency_seconds`: GPU operation latency
- `gpu_allocation_requests_total`: Total allocation requests
- `gpu_allocation_duration_seconds`: Allocation duration

### Alerting Rules

```yaml
groups:
- name: gpu-optimization
  rules:
  - alert: HighGPUMemoryUsage
    expr: gpu_memory_usage_mb / gpu_memory_total_mb > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High GPU memory usage detected"

  - alert: GPUTemperatureHigh
    expr: gpu_temperature_celsius > 80
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "GPU temperature is critically high"

  - alert: GPUAllocationFailed
    expr: rate(gpu_allocation_requests_total{status="failed"}[5m]) > 0.1
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "GPU allocation failures detected"
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Available

**Symptoms**: Allocation requests fail with "no suitable device available"

**Solutions**:

- Check GPU availability and status
- Verify memory requirements are reasonable
- Check for resource conflicts

```bash
# Check GPU status
nvidia-smi

# Check allocation status
python scripts/optimize_gpu_resources.py status

# Check service requirements
kubectl describe deployment embedding-service -n medical-kg
```

#### 2. High Memory Usage

**Symptoms**: GPU memory usage consistently high

**Solutions**:

- Run memory optimization
- Check for memory leaks
- Adjust memory thresholds

```bash
# Optimize memory usage
python scripts/optimize_gpu_resources.py optimize embedding-service

# Check memory report
python scripts/optimize_gpu_resources.py report

# Adjust memory threshold
kubectl edit configmap gpu-optimizer-config -n medical-kg
```

#### 3. Allocation Timeouts

**Symptoms**: GPU allocation requests timeout

**Solutions**:

- Increase allocation timeout
- Check GPU availability
- Optimize allocation logic

```bash
# Check allocation status
python scripts/optimize_gpu_resources.py status

# Increase timeout
kubectl edit configmap gpu-optimizer-config -n medical-kg

# Restart optimizer
kubectl rollout restart deployment/gpu-resource-optimizer -n medical-kg
```

### Debugging Commands

```bash
# Check GPU resource status
python scripts/optimize_gpu_resources.py status

# Generate detailed report
python scripts/optimize_gpu_resources.py report

# Export status for analysis
python scripts/optimize_gpu_resources.py export debug-status.json

# Check optimizer logs
kubectl logs -l app=gpu-resource-optimizer -n medical-kg --tail=100

# Check GPU metrics
kubectl port-forward svc/gpu-resource-optimizer 8000:8000 -n medical-kg
curl http://localhost:8000/metrics | grep gpu_
```

## Best Practices

### 1. Resource Planning

Plan GPU resources based on workload requirements:

- **Embedding Service**: 2-8GB memory, moderate compute
- **Reranking Service**: 1-4GB memory, high compute
- **Docling VLM Service**: 4-16GB memory, very high compute
- **GPU Management Service**: 0.5-2GB memory, low compute

### 2. Memory Management

Implement effective memory management:

- Set appropriate memory thresholds (80-85%)
- Monitor memory usage patterns
- Implement memory cleanup procedures
- Use memory pooling where possible

### 3. Allocation Strategy

Use smart allocation strategies:

- Prefer GPUs with highest free memory
- Consider service affinity requirements
- Implement load balancing across GPUs
- Monitor allocation success rates

### 4. Monitoring and Alerting

Implement comprehensive monitoring:

- Track key performance indicators
- Set up appropriate alerting thresholds
- Monitor trends and patterns
- Regular performance reviews

## Performance Optimization

### 1. Memory Optimization

Optimize memory usage for better performance:

```python
# Configure memory optimization
optimizer.memory_optimizer.memory_threshold = 0.80  # 80% threshold
optimizer.memory_optimizer.optimization_interval = 300  # 5 minutes

# Apply memory optimization
results = await optimizer.memory_optimizer.optimize_memory_usage('service-name')
```

### 2. Allocation Optimization

Optimize allocation for faster response times:

```python
# Configure allocation settings
optimizer.allocation_manager.allocation_timeout = 30  # 30 seconds
optimizer.allocation_manager.deallocation_timeout = 10  # 10 seconds

# Allocate with preferences
device_id = await optimizer.allocation_manager.allocate_gpu(
    'service-name',
    required_memory_mb=4096,
    preferred_device_id=0
)
```

### 3. Monitoring Optimization

Optimize monitoring for better performance:

```python
# Configure monitoring intervals
optimizer.monitor.update_interval = 30  # 30 seconds
optimizer.monitor.metrics_retention = 3600  # 1 hour

# Update metrics
optimizer.monitor.update_metrics('service-name')
```

## Security Considerations

### 1. Resource Access Control

Implement proper access controls:

- Restrict GPU access to authorized services
- Use service accounts for GPU operations
- Implement resource quotas and limits
- Monitor resource usage patterns

### 2. Data Protection

Protect sensitive data in GPU memory:

- Clear GPU memory after use
- Implement memory encryption where possible
- Monitor memory access patterns
- Use secure memory allocation

### 3. Network Security

Secure GPU communication:

- Use encrypted communication channels
- Implement network policies
- Monitor network traffic
- Use secure service discovery

## Maintenance

### 1. Regular Updates

Keep GPU optimization components updated:

```bash
# Update GPU resource optimizer
kubectl set image deployment/gpu-resource-optimizer \
  gpu-resource-optimizer=medical-kg/gpu-resource-optimizer:latest -n medical-kg

# Update configuration
kubectl apply -f ops/k8s/gpu-resource-optimizer.yaml
```

### 2. Configuration Management

Manage configuration changes:

```bash
# Update optimization settings
kubectl edit configmap gpu-optimizer-config -n medical-kg

# Apply new configuration
kubectl rollout restart deployment/gpu-resource-optimizer -n medical-kg
```

### 3. Performance Monitoring

Monitor performance regularly:

- Review optimization effectiveness
- Analyze resource utilization trends
- Identify optimization opportunities
- Update optimization strategies

## Conclusion

The GPU resource optimization system provides automated resource management for GPU services, ensuring optimal performance and efficient resource utilization. Regular monitoring, maintenance, and optimization are essential for maintaining optimal GPU resource management.
