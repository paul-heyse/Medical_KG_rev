# Torch-Free Architecture

## Overview

The Medical_KG_rev system has been architected to be completely torch-free in the main API gateway, with all GPU-intensive operations moved to dedicated gRPC services running in Docker containers. This architecture provides significant benefits in terms of scalability, resource isolation, and deployment simplicity.

## Architecture Benefits

### 1. **Independent Scaling**

- GPU services can be scaled independently based on workload demands
- Main gateway scales based on API request volume
- No resource contention between API and GPU operations

### 2. **Resource Isolation**

- GPU memory and compute resources are isolated to dedicated containers
- Main gateway uses minimal resources without GPU dependencies
- Clear separation of concerns between API and AI processing

### 3. **Simplified Deployment**

- Main gateway can be deployed on any infrastructure without GPU requirements
- GPU services can be deployed on specialized GPU-enabled nodes
- Easier CI/CD pipelines without GPU dependencies

### 4. **Fail-Fast Behavior**

- GPU service failures don't crash the main gateway
- Circuit breaker patterns prevent cascading failures
- Graceful degradation when GPU services are unavailable

## Service Architecture

### Main Gateway (Torch-Free)

- **FastAPI** application with no PyTorch dependencies
- Handles API requests, authentication, and orchestration
- Communicates with GPU services via gRPC
- Implements circuit breaker patterns for resilience

### GPU Services (Torch-Enabled)

- **Docling VLM Service**: PDF processing and document understanding
- **Embedding Service**: Vector embeddings generation (SPLADE, Qwen3)
- **Reranking Service**: Search result reranking
- **GPU Management Service**: GPU resource allocation and monitoring

## Communication Patterns

### gRPC Communication

- All inter-service communication uses gRPC for performance and type safety
- Protocol buffers define service contracts
- Streaming support for large document processing

### Circuit Breaker Pattern

- Prevents cascading failures when GPU services are unavailable
- Configurable failure thresholds and recovery timeouts
- Automatic retry with exponential backoff

### Service Discovery

- Dynamic service discovery for GPU services
- Health checks and automatic failover
- Load balancing across multiple service instances

## Security

### mTLS Authentication

- Mutual TLS authentication between services
- Certificate-based service identity verification
- Encrypted communication channels

### Service Isolation

- Network policies isolate GPU services
- Resource quotas prevent resource exhaustion
- Audit logging for all service operations

## Monitoring and Observability

### Metrics

- Prometheus metrics for all services
- GPU utilization and memory usage tracking
- Service call latency and error rates
- Circuit breaker state monitoring

### Tracing

- OpenTelemetry distributed tracing
- End-to-end request tracing across services
- Performance bottleneck identification

### Logging

- Structured logging with correlation IDs
- Audit logs for security compliance
- Error tracking and alerting

## Deployment

### Docker Containers

- Main gateway: Lightweight container without GPU dependencies
- GPU services: Specialized containers with CUDA support
- Sidecar containers for monitoring and logging

### Kubernetes

- Separate namespaces for gateway and GPU services
- GPU node scheduling for AI workloads
- Horizontal Pod Autoscaling based on metrics

### Configuration Management

- Environment-specific configuration
- Secret management for certificates and API keys
- ConfigMap and Secret resources

## Migration Guide

### From Torch-Dependent to Torch-Free

1. **Identify Torch Usage**: Scan codebase for torch imports and usage
2. **Create gRPC Services**: Implement GPU functionality as separate services
3. **Update Client Code**: Replace direct torch calls with gRPC service calls
4. **Remove Dependencies**: Remove torch from main gateway requirements
5. **Update Tests**: Modify tests to mock gRPC services instead of torch
6. **Deploy Services**: Deploy GPU services and update gateway configuration

### Code Changes Required

```python
# Before (torch-dependent)
import torch
device = torch.cuda.get_device_properties(0)
embeddings = model.encode(texts)

# After (torch-free)
from ..clients.embedding_client import EmbeddingClientManager
client_manager = EmbeddingClientManager()
embeddings = await client_manager.generate_embeddings_batch(texts)
```

## Performance Considerations

### Latency

- gRPC communication adds minimal latency (~1-2ms)
- Batch processing reduces per-request overhead
- Connection pooling improves performance

### Throughput

- Independent scaling increases overall throughput
- GPU services can process multiple requests in parallel
- Circuit breaker prevents overload conditions

### Resource Usage

- Main gateway uses ~50% less memory without torch
- GPU services optimize memory usage for AI workloads
- Better resource utilization across the system

## Troubleshooting

### Common Issues

1. **Service Unavailable**: Check GPU service health and circuit breaker state
2. **High Latency**: Monitor GPU utilization and service scaling
3. **Memory Issues**: Check GPU memory usage and allocation
4. **Certificate Errors**: Verify mTLS certificate configuration

### Debugging Tools

- Service health check endpoints
- Circuit breaker state monitoring
- GPU metrics dashboards
- Distributed tracing visualization

## Future Enhancements

### Planned Improvements

- GPU service auto-scaling based on queue depth
- Advanced caching strategies for repeated operations
- Model warm-up procedures for consistent performance
- Request queuing for high-load scenarios

### Monitoring Enhancements

- Custom metrics for business logic
- Advanced alerting rules
- Performance regression detection
- Capacity planning tools

## Conclusion

The torch-free architecture provides a robust, scalable foundation for the Medical_KG_rev system. By isolating GPU operations in dedicated services, the system achieves better resource utilization, improved reliability, and simplified deployment while maintaining high performance for AI workloads.
