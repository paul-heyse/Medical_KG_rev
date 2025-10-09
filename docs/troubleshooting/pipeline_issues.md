# Pipeline Troubleshooting Guide

This guide provides solutions for common issues encountered when working with the Medical KG pipeline architecture.

## Chunking Coordinator Errors

### ProfileNotFoundError

**Error Message**: "Profile 'biomedical' does not exist"

**Root Cause**: The specified chunking profile is not configured or does not exist.

**Solution**:

1. Check if profile exists in `config/chunking/profiles/`
2. Verify profile name spelling and case sensitivity
3. Ensure profile is properly loaded in configuration

```bash
# Check available profiles
ls config/chunking/profiles/

# Verify profile configuration
cat config/chunking/profiles/biomedical.yaml
```

**Prevention**: Always validate profile names before making chunking requests.

### TokenizerMismatchError

**Error Message**: "Tokenizer mismatch between chunking and embedding models"

**Root Cause**: The chunking tokenizer does not match the embedding model tokenizer.

**Solution**:

1. Ensure chunking profile uses the same tokenizer as the embedding model
2. Update chunking profile to match embedding model configuration
3. Verify tokenizer compatibility in both configurations

```yaml
# config/chunking/profiles/biomedical.yaml
tokenizer:
  model: "bert-base-uncased"  # Must match embedding model

# config/embedding/vllm.yaml
model:
  tokenizer: "bert-base-uncased"  # Must match chunking tokenizer
```

**Prevention**: Use consistent tokenizer configurations across chunking and embedding.

### ChunkingUnavailableError

**Error Message**: "Chunking service is temporarily unavailable"

**Root Cause**: The underlying chunking service (MinerU) is down or experiencing issues.

**Solution**:

1. Check MinerU service health and status
2. Verify GPU availability and memory
3. Check service logs for specific errors
4. Restart MinerU service if necessary

```bash
# Check MinerU service status
docker ps | grep mineru

# Check GPU availability
nvidia-smi

# Check service logs
docker logs mineru-worker

# Restart service if needed
docker restart mineru-worker
```

**Prevention**: Implement health checks and monitoring for MinerU service.

### MineruOutOfMemoryError

**Error Message**: "GPU memory exhausted, retry after cooldown"

**Root Cause**: GPU memory is insufficient for the document size or batch processing.

**Solution**:

1. Reduce document size or split into smaller chunks
2. Decrease batch size in chunking configuration
3. Wait for GPU memory to be freed
4. Consider using CPU-based chunking for large documents

```yaml
# config/chunking/profiles/biomedical.yaml
batch_size: 1  # Reduce from default
max_document_size: 1000000  # Limit document size
```

**Prevention**: Monitor GPU memory usage and implement document size limits.

### MineruGpuUnavailableError

**Error Message**: "GPU is not available for chunking operations"

**Root Cause**: GPU is not available or not properly configured.

**Solution**:

1. Check GPU availability and driver status
2. Verify Docker GPU configuration
3. Restart GPU services if necessary
4. Fall back to CPU-based chunking

```bash
# Check GPU status
nvidia-smi

# Check Docker GPU configuration
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi

# Restart GPU services
sudo systemctl restart nvidia-persistenced
```

**Prevention**: Implement GPU health checks and fallback mechanisms.

## Embedding Errors

### Namespace Access Denied

**Error Message**: "Access denied to namespace 'clinical'"

**Root Cause**: Tenant does not have permission to access the specified namespace.

**Solution**:

1. Check tenant_id in JWT token
2. Verify namespace policy configuration
3. Ensure tenant is included in allowed list
4. Check namespace policy cache

```yaml
# config/embedding/namespaces/clinical.yaml
access_policy:
  allowed_tenants:
    - "tenant1"
    - "tenant2"
  # Add your tenant_id here
```

**Prevention**: Implement proper tenant validation and namespace policy management.

### Model Not Found

**Error Message**: "Embedding model 'bert-base-uncased' not found"

**Root Cause**: The specified embedding model is not registered or available.

**Solution**:

1. Check if model is registered in embedding model registry
2. Verify model configuration and availability
3. Ensure model is properly loaded and accessible
4. Check model registry configuration

```bash
# Check registered models
curl -X GET http://localhost:8000/v1/models

# Verify model configuration
cat config/embedding/vllm.yaml
```

**Prevention**: Implement model validation and registration checks.

### Persistence Failed

**Error Message**: "Failed to persist embeddings to vector store"

**Root Cause**: Vector store is unavailable or experiencing issues.

**Solution**:

1. Check vector store connectivity and health
2. Verify disk space and storage availability
3. Check vector store configuration
4. Restart vector store service if necessary

```bash
# Check vector store status
docker ps | grep vector-store

# Check disk space
df -h

# Check vector store logs
docker logs vector-store
```

**Prevention**: Implement vector store health checks and monitoring.

## Orchestration Failures

### Stage Timeout

**Error Message**: "Stage execution timed out after 300 seconds"

**Root Cause**: Stage execution exceeds the configured timeout limit.

**Solution**:

1. Increase timeout in resilience policy configuration
2. Optimize stage implementation for better performance
3. Check for resource constraints or bottlenecks
4. Implement stage-level timeout handling

```yaml
# config/orchestration/resilience.yaml
timeouts:
  default: 600  # Increase from 300 seconds
  stages:
    chunking: 900  # Specific timeout for chunking stage
```

**Prevention**: Implement proper timeout configuration and stage optimization.

### Retry Exhausted

**Error Message**: "All retry attempts exhausted for stage execution"

**Root Cause**: Stage fails repeatedly and exceeds maximum retry attempts.

**Solution**:

1. Check dead letter queue for failed stages
2. Review stage logs for root cause analysis
3. Fix underlying issue causing stage failures
4. Adjust retry policy if appropriate

```bash
# Check dead letter queue
kubectl logs -l app=orchestration | grep "dead letter"

# Review stage logs
kubectl logs -l app=orchestration | grep "stage failed"
```

**Prevention**: Implement proper error handling and monitoring for stage failures.

### Dependency Missing

**Error Message**: "Required dependency 'chunking-service' is not available"

**Root Cause**: Required service dependency is not running or accessible.

**Solution**:

1. Verify all required services are running
2. Check service discovery and connectivity
3. Ensure proper service configuration
4. Restart missing services

```bash
# Check service status
kubectl get pods -l app=chunking-service
kubectl get pods -l app=embedding-service
kubectl get pods -l app=vector-store

# Check service connectivity
kubectl exec -it chunking-service -- curl http://embedding-service:8000/health
```

**Prevention**: Implement service dependency checks and health monitoring.

## Documentation Lint Failures

### D100 Missing Module Docstring

**Error Message**: "Missing docstring in public module"

**Root Cause**: Module file is missing a module-level docstring.

**Solution**:

1. Add comprehensive module docstring at the top of the file
2. Include module purpose, responsibilities, and usage examples
3. Follow Google-style docstring format

```python
"""One-line summary of module purpose.

This module provides detailed explanation of what the module does, its role
in the larger system, and key design decisions.

Key Responsibilities:
    - Responsibility 1: Be specific about what the module handles
    - Responsibility 2: Include data transformations, external calls, etc.

Example:
    >>> from Medical_KG_rev.gateway.coordinators import ChunkingCoordinator
    >>> coordinator = ChunkingCoordinator(...)
    >>> result = coordinator.execute(request)
"""
```

### D101 Missing Class Docstring

**Error Message**: "Missing docstring in public class"

**Root Cause**: Class definition is missing a class-level docstring.

**Solution**:

1. Add comprehensive class docstring immediately after class definition
2. Include class purpose, attributes, invariants, and usage examples
3. Follow Google-style docstring format

```python
class MyClass:
    """One-line summary of class purpose.

    Detailed explanation of what the class does, why it exists, and how it fits
    into the larger architecture.

    Attributes:
        attribute_name: Description of attribute purpose and valid ranges

    Example:
        >>> instance = MyClass(param1="value", param2=42)
        >>> result = instance.method()
    """
```

### D103 Missing Function Docstring

**Error Message**: "Missing docstring in public function"

**Root Cause**: Function definition is missing a function-level docstring.

**Solution**:

1. Add comprehensive function docstring immediately after function definition
2. Include function purpose, parameters, return values, and examples
3. Follow Google-style docstring format

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """One-line summary of function purpose.

    Detailed explanation of what the function does and how it works.

    Args:
        param1: Description of first parameter with valid values and constraints
        param2: Description of second parameter. Defaults to 10.

    Returns:
        Description of return value and its meaning.

    Example:
        >>> result = my_function("test", 20)
        >>> assert result is True
    """
```

### Section Header Missing

**Error Message**: "Missing required section headers per section_headers.md"

**Root Cause**: File is missing required section headers for code organization.

**Solution**:

1. Add required section headers following the format in `section_headers.md`
2. Organize code into appropriate sections
3. Follow the established section ordering rules

```python
# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# DATA MODELS
# ============================================================================

# ============================================================================
# COORDINATOR IMPLEMENTATION
# ============================================================================

# ============================================================================
# EXPORTS
# ============================================================================
```

## Performance Issues

### High Memory Usage

**Symptoms**: System running out of memory, slow performance

**Root Cause**: Excessive memory usage in pipeline operations

**Solution**:

1. Monitor memory usage across all services
2. Implement memory limits and garbage collection
3. Optimize data structures and algorithms
4. Use streaming processing for large datasets

```bash
# Monitor memory usage
kubectl top pods
docker stats

# Check memory limits
kubectl describe pod <pod-name> | grep -A 5 "Limits"
```

### Slow Response Times

**Symptoms**: High latency, timeouts, poor user experience

**Root Cause**: Performance bottlenecks in pipeline operations

**Solution**:

1. Profile pipeline performance and identify bottlenecks
2. Optimize database queries and external API calls
3. Implement caching for frequently accessed data
4. Use asynchronous processing where appropriate

```bash
# Profile performance
kubectl logs -l app=gateway | grep "duration"
kubectl logs -l app=chunking | grep "processing time"
```

## Monitoring and Debugging

### Health Checks

```bash
# Check overall system health
curl -X GET http://localhost:8000/health

# Check specific service health
curl -X GET http://localhost:8000/v1/health/chunking
curl -X GET http://localhost:8000/v1/health/embedding
```

### Log Analysis

```bash
# View recent logs
kubectl logs -l app=gateway --tail=100

# Filter for specific errors
kubectl logs -l app=chunking | grep "ERROR"

# Follow logs in real-time
kubectl logs -f -l app=embedding
```

### Metrics Monitoring

```bash
# Check Prometheus metrics
curl -X GET http://localhost:9090/api/v1/query?query=chunking_attempts_total

# View Grafana dashboards
open http://localhost:3000
```

## Getting Help

If you encounter issues not covered in this guide:

1. **Check Logs**: Review service logs for detailed error information
2. **Monitor Metrics**: Use Prometheus and Grafana to identify performance issues
3. **Review Documentation**: Consult the comprehensive API documentation
4. **Team Support**: Reach out to the platform team for assistance
5. **Issue Tracking**: Report bugs and issues in the project issue tracker

## References

- [Pipeline Extension Guide](../guides/pipeline_extension_guide.md)
- [Documentation Standards](../contributing/documentation_standards.md)
- [API Documentation](../api/)
- [Architecture Decision Records](../adr/)
