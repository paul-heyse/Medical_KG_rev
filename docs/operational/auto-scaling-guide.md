# GPU Services Auto-Scaling Guide

This guide provides comprehensive information about the auto-scaling infrastructure for GPU services in the Medical KG platform.

## Overview

The auto-scaling system automatically adjusts the number of replicas for GPU services based on resource utilization and custom metrics. This ensures optimal performance while maintaining cost efficiency.

## Architecture Components

### 1. Horizontal Pod Autoscaler (HPA)

The HPA monitors resource utilization and custom metrics to automatically scale GPU services:

- **GPU Management Service**: Scales based on CPU and memory utilization
- **Embedding Service**: Scales based on CPU, memory, and GPU utilization
- **Reranking Service**: Scales based on CPU, memory, and GPU utilization
- **Docling VLM Service**: Scales based on CPU, memory, GPU utilization, and GPU memory usage

### 2. Custom Metrics Adapter

The custom metrics adapter exposes GPU-specific metrics to Kubernetes:

- GPU utilization percentage
- GPU memory usage
- GPU temperature
- Service-specific metrics

### 3. GPU Metrics Exporter

The GPU metrics exporter collects and exposes GPU metrics to Prometheus:

- Real-time GPU utilization
- GPU memory usage
- System resource usage
- Service performance metrics

## Configuration

### HPA Configuration

Each service has its own HPA configuration with specific scaling parameters:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: embedding-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: embedding-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: gpu_utilization_percent
      target:
        type: AverageValue
        averageValue: "80"
```

### Scaling Behavior

The HPA uses configurable scaling behavior to prevent rapid scaling:

```yaml
behavior:
  scaleUp:
    stabilizationWindowSeconds: 300
    policies:
    - type: Percent
      value: 100
      periodSeconds: 60
  scaleDown:
    stabilizationWindowSeconds: 600
    policies:
    - type: Percent
      value: 50
      periodSeconds: 60
```

## Deployment

### Prerequisites

- Kubernetes cluster with HPA support
- Prometheus monitoring stack
- NVIDIA GPU nodes with nvidia-ml-py3

### Deploy Auto-Scaling Infrastructure

```bash
# Deploy the complete auto-scaling infrastructure
python scripts/deploy_auto_scaling.py --action deploy

# Check deployment status
python scripts/deploy_auto_scaling.py --action status

# Undeploy if needed
python scripts/deploy_auto_scaling.py --action undeploy
```

### Manual Deployment

```bash
# Create namespace
kubectl create namespace medical-kg

# Deploy GPU metrics exporter
kubectl apply -f ops/k8s/gpu-metrics-exporter.yaml

# Deploy custom metrics adapter
kubectl apply -f ops/k8s/custom-metrics-adapter.yaml

# Deploy HPA configurations
kubectl apply -f ops/k8s/hpa-gpu-services.yaml
```

## Monitoring

### Grafana Dashboard

The auto-scaling dashboard provides comprehensive monitoring:

- GPU utilization by service
- GPU memory usage
- Service replica count
- HPA scaling events
- Service request rate
- Service response time
- CPU and memory usage

### Key Metrics

#### GPU Metrics

- `gpu_utilization_percent`: GPU utilization percentage
- `gpu_memory_usage_mb`: GPU memory usage in MB
- `gpu_memory_total_mb`: Total GPU memory in MB
- `gpu_temperature_celsius`: GPU temperature

#### Service Metrics

- `service_requests_total`: Total service requests
- `service_response_time_seconds`: Service response time
- `cpu_usage_percent`: CPU usage percentage
- `memory_usage_mb`: Memory usage in MB

#### Kubernetes Metrics

- `kube_deployment_status_replicas`: Current replica count
- `kube_horizontalpodautoscaler_status_current_replicas`: HPA current replicas

## Troubleshooting

### Common Issues

#### 1. HPA Not Scaling

**Symptoms**: HPA shows no scaling events despite high resource usage

**Solutions**:

- Check if custom metrics are available
- Verify Prometheus is collecting GPU metrics
- Check HPA configuration for correct metric names

```bash
# Check HPA status
kubectl describe hpa embedding-service-hpa -n medical-kg

# Check custom metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/medical-kg/pods/*/gpu_utilization_percent
```

#### 2. GPU Metrics Not Available

**Symptoms**: GPU metrics show as unavailable or zero

**Solutions**:

- Check if NVIDIA drivers are installed
- Verify nvidia-ml-py3 is available
- Check GPU metrics exporter logs

```bash
# Check GPU metrics exporter logs
kubectl logs -l app=gpu-metrics-exporter -n medical-kg

# Check NVIDIA driver status
nvidia-smi
```

#### 3. Scaling Too Aggressively

**Symptoms**: Services scale up and down rapidly

**Solutions**:

- Increase stabilization windows
- Adjust scaling policies
- Check for metric spikes

```bash
# Update HPA configuration
kubectl edit hpa embedding-service-hpa -n medical-kg
```

### Debugging Commands

```bash
# Check HPA status
kubectl get hpa -n medical-kg

# Describe specific HPA
kubectl describe hpa embedding-service-hpa -n medical-kg

# Check pod status
kubectl get pods -l app=embedding-service -n medical-kg

# Check metrics
kubectl top pods -l app=embedding-service -n medical-kg

# Check custom metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/medical-kg/pods/*/gpu_utilization_percent
```

## Best Practices

### 1. Resource Limits

Set appropriate resource limits to prevent resource contention:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "1000m"
    nvidia.com/gpu: 1
  limits:
    memory: "8Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1
```

### 2. Scaling Thresholds

Use conservative scaling thresholds to prevent over-scaling:

- CPU: 70% utilization
- Memory: 80% utilization
- GPU: 80% utilization

### 3. Stabilization Windows

Use appropriate stabilization windows:

- Scale up: 5 minutes
- Scale down: 10 minutes

### 4. Monitoring

Monitor key metrics to ensure optimal scaling:

- GPU utilization trends
- Service response times
- Scaling events
- Resource usage patterns

## Performance Optimization

### 1. Metric Collection Interval

Optimize metric collection interval for balance between accuracy and performance:

```yaml
# In GPU metrics exporter
interval: 30s  # Default
interval: 15s  # More responsive
interval: 60s  # Less overhead
```

### 2. Scaling Policies

Fine-tune scaling policies based on workload patterns:

```yaml
behavior:
  scaleUp:
    stabilizationWindowSeconds: 300
    policies:
    - type: Percent
      value: 100
      periodSeconds: 60
    - type: Pods
      value: 2
      periodSeconds: 60
```

### 3. Resource Requests

Set appropriate resource requests to ensure proper scheduling:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "1000m"
    nvidia.com/gpu: 1
```

## Security Considerations

### 1. RBAC Permissions

Ensure proper RBAC permissions for HPA and metrics access:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: custom-metrics-adapter
rules:
- apiGroups: [""]
  resources: ["services", "endpoints", "pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["custom.metrics.k8s.io"]
  resources: ["*"]
  verbs: ["*"]
```

### 2. Network Security

Use network policies to secure metrics endpoints:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gpu-metrics-policy
spec:
  podSelector:
    matchLabels:
      app: gpu-metrics-exporter
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
```

## Maintenance

### 1. Regular Updates

Keep auto-scaling components updated:

```bash
# Update GPU metrics exporter
kubectl set image deployment/gpu-metrics-exporter gpu-metrics-exporter=medical-kg/gpu-metrics-exporter:latest -n medical-kg

# Update custom metrics adapter
kubectl set image deployment/custom-metrics-adapter custom-metrics-adapter=medical-kg/custom-metrics-adapter:latest -n medical-kg
```

### 2. Configuration Changes

Update HPA configurations as needed:

```bash
# Edit HPA configuration
kubectl edit hpa embedding-service-hpa -n medical-kg

# Apply new configuration
kubectl apply -f ops/k8s/hpa-gpu-services.yaml
```

### 3. Monitoring Maintenance

Regularly review and update monitoring configurations:

- Update Grafana dashboards
- Adjust alert thresholds
- Review scaling policies
- Analyze performance trends

## Conclusion

The auto-scaling system provides automated resource management for GPU services, ensuring optimal performance and cost efficiency. Regular monitoring and maintenance are essential for maintaining optimal scaling behavior.
