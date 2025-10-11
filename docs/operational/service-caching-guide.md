# Service Caching Guide

This guide provides comprehensive information about the service caching system implemented for GPU services in the Medical KG platform.

## Overview

The service caching system provides transparent caching for gRPC service calls to improve performance and reduce latency. It supports multiple cache backends and provides comprehensive monitoring and management capabilities.

## Architecture

### Components

1. **Cache Manager**: Core caching functionality with support for multiple backends
2. **Service Cache**: Service-specific caching implementations
3. **Cache Integration**: Integration layer for gRPC service clients
4. **Cache Monitoring**: Prometheus metrics and Grafana dashboards

### Cache Backends

- **In-Memory Cache**: Fast, local caching using LRU eviction
- **Redis Cache**: Distributed caching with persistence

## Configuration

### Cache Configuration

```python
from Medical_KG_rev.services.caching.cache_manager import CacheConfig

config = CacheConfig(
    ttl_seconds=3600,           # Time to live in seconds
    max_size_bytes=1024**3,    # Maximum cache size (1GB)
    compression_enabled=True,   # Enable compression
    serialization_format="json"  # Serialization format
)
```

### Service Cache Configuration

```python
from Medical_KG_rev.services.caching.service_cache import ServiceCacheConfig, CachePolicy

config = ServiceCacheConfig(
    cache_type="memory",        # Cache backend type
    redis_url="redis://localhost:6379",  # Redis URL (if using Redis)
    default_policy=CachePolicy(
        enabled=True,
        ttl_seconds=3600,
        key_exclude_params=['pdf_content', 'large_data']
    ),
    policies={
        'generate_embedding': CachePolicy(
            enabled=True,
            ttl_seconds=7200,  # Longer TTL for embeddings
            key_include_params=['text', 'model', 'config']
        ),
        'rerank': CachePolicy(
            enabled=True,
            ttl_seconds=1800,  # Shorter TTL for reranking
            key_include_params=['query', 'documents', 'model', 'config']
        )
    }
)
```

## Usage

### Basic Cache Operations

```python
from Medical_KG_rev.services.caching.cache_manager import create_cache_manager

# Create cache manager
cache_manager = create_cache_manager("memory")

# Set a value
await cache_manager.set("service", "operation", {"param": "value"}, "result")

# Get a value
result = await cache_manager.get("service", "operation", {"param": "value"})

# Delete a value
deleted = await cache_manager.delete("service", "operation", {"param": "value"})

# Get statistics
stats = await cache_manager.get_stats()
```

### Service-Specific Caching

```python
from Medical_KG_rev.services.caching.service_cache import EmbeddingCache, ServiceCacheConfig

# Create embedding cache
cache_config = ServiceCacheConfig()
embedding_cache = EmbeddingCache(cache_config)

# Cache embedding
await embedding_cache.set_embedding("text", "model", embedding_result)

# Get cached embedding
cached_result = await embedding_cache.get_embedding("text", "model")
```

### Cache-Integrated Clients

```python
from Medical_KG_rev.services.caching.cache_integration import (
    CacheIntegratedEmbeddingClient,
    CacheIntegrationConfig
)

# Create cache-integrated client
cache_config = CacheIntegrationConfig(enabled=True, cache_type="memory")
client = CacheIntegratedEmbeddingClient(embedding_client, cache_config)

# Use client (caching is transparent)
result = await client.generate_embedding("text", "model")
```

## Cache Policies

### Policy Configuration

```python
from Medical_KG_rev.services.caching.service_cache import CachePolicy

policy = CachePolicy(
    enabled=True,                    # Enable caching
    ttl_seconds=3600,               # Time to live
    key_include_params=['text'],     # Include only these params in key
    key_exclude_params=['large_data'], # Exclude these params from key
    condition="result.success"        # Condition for caching
)
```

### Policy Conditions

- `result.success`: Cache only successful results
- `result.error_code == 0`: Cache only results with no errors
- Custom conditions can be added as needed

## Monitoring

### Prometheus Metrics

The caching system exposes the following Prometheus metrics:

- `cache_hits_total`: Total cache hits by service and operation
- `cache_misses_total`: Total cache misses by service and operation
- `cache_operations_total`: Total cache operations by service, operation, and result
- `cache_size_bytes`: Cache size in bytes by service
- `cache_ttl_seconds`: Cache TTL distribution by service
- `cache_performance_seconds`: Cache operation performance by service and operation
- `cached_operations_total`: Total cached operations by service and operation

### Grafana Dashboard

A Grafana dashboard is provided at `ops/monitoring/cache-monitoring-dashboard.json` with panels for:

- Cache hit rate
- Cache operations per second
- Cache size utilization
- Cache performance
- Cache TTL distribution
- Cached operations

## Management

### Command-Line Interface

The `scripts/manage_service_cache.py` script provides a CLI for cache management:

```bash
# Display cache statistics
python scripts/manage_service_cache.py stats

# Clear cache
python scripts/manage_service_cache.py clear --confirm

# Monitor cache in real-time
python scripts/manage_service_cache.py monitor --interval 5

# Test cache performance
python scripts/manage_service_cache.py test --operations 1000

# Check cache health
python scripts/manage_service_cache.py health

# Display cache configuration
python scripts/manage_service_cache.py config
```

### Cache Management Operations

```python
# Get cache statistics
stats = await cache_manager.get_stats()

# Clear all cache entries
await cache_manager.clear()

# Close cache manager
await cache_manager.close()
```

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  cache-service:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - cache-data:/data
    environment:
      - REDIS_PASSWORD=redis_password
    command: redis-server --appendonly yes --requirepass redis_password

volumes:
  cache-data:
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cache-service
  template:
    metadata:
      labels:
        app: cache-service
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          volumeMounts:
            - name: redis-data
              mountPath: /data
          resources:
            requests:
              memory: "512Mi"
              cpu: "100m"
            limits:
              memory: "2Gi"
              cpu: "500m"
      volumes:
        - name: redis-data
          persistentVolumeClaim:
            claimName: cache-service-pvc
```

## Performance Optimization

### Cache Key Design

- Use consistent parameter ordering for cache keys
- Exclude large binary data from cache keys
- Include only relevant parameters in cache keys
- Use hashing for complex parameter structures

### TTL Configuration

- Set appropriate TTL values based on data freshness requirements
- Use longer TTLs for stable data (embeddings)
- Use shorter TTLs for dynamic data (reranking results)
- Monitor cache hit rates and adjust TTLs accordingly

### Memory Management

- Configure appropriate cache size limits
- Use LRU eviction for memory-constrained environments
- Monitor memory usage and adjust limits as needed
- Consider compression for large cached values

## Troubleshooting

### Common Issues

1. **Low Cache Hit Rate**
   - Check cache key consistency
   - Verify TTL settings
   - Review cache policy configuration

2. **High Memory Usage**
   - Reduce cache size limits
   - Enable compression
   - Review TTL settings

3. **Cache Performance Issues**
   - Check cache backend configuration
   - Monitor cache operations
   - Review cache key complexity

### Debugging

```python
import logging
logging.getLogger('Medical_KG_rev.services.caching').setLevel(logging.DEBUG)

# Enable debug logging for cache operations
```

### Health Checks

```python
# Check cache health
python scripts/manage_service_cache.py health

# Monitor cache performance
python scripts/manage_service_cache.py monitor
```

## Best Practices

1. **Cache Strategy**
   - Cache expensive operations (VLM processing, embeddings)
   - Use appropriate TTLs for different data types
   - Implement cache invalidation strategies

2. **Key Design**
   - Use consistent parameter ordering
   - Exclude large binary data from keys
   - Include only relevant parameters

3. **Monitoring**
   - Monitor cache hit rates
   - Track cache performance metrics
   - Set up alerts for cache issues

4. **Configuration**
   - Use environment-specific configurations
   - Test cache configurations in staging
   - Document cache policies and TTLs

## Security Considerations

1. **Data Sensitivity**
   - Consider data sensitivity when caching
   - Use appropriate TTLs for sensitive data
   - Implement cache encryption if needed

2. **Access Control**
   - Secure cache backend access
   - Use authentication for Redis
   - Implement proper network security

3. **Data Integrity**
   - Validate cached data
   - Implement cache invalidation strategies
   - Monitor for cache corruption

## Future Enhancements

1. **Advanced Caching**
   - Implement cache warming strategies
   - Add cache prediction algorithms
   - Support for cache hierarchies

2. **Performance Improvements**
   - Optimize cache key generation
   - Implement cache compression algorithms
   - Add cache preloading capabilities

3. **Monitoring Enhancements**
   - Add cache quality metrics
   - Implement cache analytics
   - Add predictive monitoring
