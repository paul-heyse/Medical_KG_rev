# Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for the Medical Knowledge Graph system. It covers common issues, debugging techniques, and solutions for each layer of the system.

## Gateway Layer Issues

### CORS Errors

**Symptoms:**

- Browser console shows CORS errors
- Requests fail with "Access-Control-Allow-Origin" errors

**Solutions:**

```python
# Check CORS configuration in gateway/app.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Debug Steps:**

1. Check browser developer tools for CORS errors
2. Verify allowed origins in CORS configuration
3. Test with curl to isolate browser-specific issues

### Authentication Failures

**Symptoms:**

- 401 Unauthorized responses
- Token validation errors

**Solutions:**

```python
# Verify token validation logic
def validate_token(token: str) -> bool:
    try:
        # Your token validation logic
        return True
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        return False
```

**Debug Steps:**

1. Check token format and expiration
2. Verify authentication middleware configuration
3. Test token validation independently

### Rate Limiting Issues

**Symptoms:**

- 429 Too Many Requests responses
- Requests throttled unexpectedly

**Solutions:**

```python
# Adjust rate limiting configuration
from slowapi import Limiter

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"]
)

# Increase limits for specific endpoints
@app.get("/api/v1/health")
@limiter.limit("1000/minute")
async def health_check():
    return {"status": "healthy"}
```

**Debug Steps:**

1. Check rate limit configuration
2. Monitor request patterns
3. Adjust limits based on usage

## Services Layer Issues

### Model Loading Failures

**Symptoms:**

- Service initialization errors
- "Model not found" exceptions

**Solutions:**

```python
# Check model path and permissions
import os
from pathlib import Path

def validate_model_path(model_path: str) -> bool:
    path = Path(model_path)
    if not path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return False

    if not os.access(path, os.R_OK):
        logger.error(f"No read permission for model path: {model_path}")
        return False

    return True
```

**Debug Steps:**

1. Verify model file exists and is accessible
2. Check file permissions
3. Validate model format and compatibility

### Memory Issues

**Symptoms:**

- Out of memory errors
- Slow performance
- Service crashes

**Solutions:**

```python
# Implement memory monitoring
import psutil
import gc

def monitor_memory():
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        logger.warning(f"High memory usage: {memory.percent}%")
        gc.collect()
        return False
    return True

# Use smaller batch sizes
def process_batch(texts: List[str], batch_size: int = 16):
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        yield process_texts(batch)
```

**Debug Steps:**

1. Monitor memory usage with tools like `htop` or `psutil`
2. Reduce batch sizes
3. Implement garbage collection
4. Use memory profiling tools

### Configuration Errors

**Symptoms:**

- Service startup failures
- Invalid configuration warnings

**Solutions:**

```python
# Validate configuration
from pydantic import BaseModel, ValidationError

def validate_config(config: Dict[str, Any]) -> bool:
    try:
        ServiceConfig(**config)
        return True
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
```

**Debug Steps:**

1. Check configuration file syntax
2. Validate required fields
3. Test configuration loading

## Adapters Layer Issues

### API Rate Limits

**Symptoms:**

- 429 Too Many Requests from external APIs
- Requests throttled

**Solutions:**

```python
# Implement exponential backoff
import time
import random

def exponential_backoff(attempt: int, base_delay: float = 1.0) -> float:
    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
    return min(delay, 60.0)  # Cap at 60 seconds

def retry_with_backoff(func, max_attempts: int = 3):
    for attempt in range(max_attempts):
        try:
            return func()
        except RateLimitError:
            if attempt < max_attempts - 1:
                delay = exponential_backoff(attempt)
                time.sleep(delay)
            else:
                raise
```

**Debug Steps:**

1. Check API rate limit documentation
2. Implement proper backoff strategies
3. Monitor request patterns

### Network Timeouts

**Symptoms:**

- Connection timeout errors
- Slow API responses

**Solutions:**

```python
# Adjust timeout settings
import httpx

client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=10.0,
        read=30.0,
        write=10.0,
        pool=5.0
    )
)

# Implement retry logic
async def fetch_with_retry(url: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = await client.get(url)
            return response
        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
```

**Debug Steps:**

1. Check network connectivity
2. Test API endpoints directly
3. Adjust timeout settings
4. Implement retry logic

### Data Transformation Errors

**Symptoms:**

- Schema validation failures
- Missing required fields

**Solutions:**

```python
# Implement robust data transformation
def transform_data(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return {
            "id": data.get("id", ""),
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "authors": data.get("authors", []),
            "publication_date": data.get("publication_date", ""),
            "doi": data.get("doi", ""),
            "open_access": data.get("open_access", {}).get("is_oa", False)
        }
    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        raise TransformationError(f"Failed to transform data: {e}")
```

**Debug Steps:**

1. Validate input data schema
2. Handle missing fields gracefully
3. Log transformation errors

## Orchestration Layer Issues

### Job Failures

**Symptoms:**

- Jobs stuck in "running" state
- Job execution errors

**Solutions:**

```python
# Implement job monitoring
class JobMonitor:
    def __init__(self):
        self.active_jobs = {}

    def track_job(self, job_id: str, status: str):
        self.active_jobs[job_id] = {
            "status": status,
            "start_time": datetime.now(),
            "last_update": datetime.now()
        }

    def check_stuck_jobs(self):
        now = datetime.now()
        for job_id, job_info in self.active_jobs.items():
            if (now - job_info["last_update"]).seconds > 3600:  # 1 hour
                logger.warning(f"Job {job_id} may be stuck")
                # Implement recovery logic
```

**Debug Steps:**

1. Check job status in ledger
2. Monitor job execution logs
3. Implement job recovery logic

### Event Processing Issues

**Symptoms:**

- Events not processed
- Event queue backlog

**Solutions:**

```python
# Monitor event processing
class EventMonitor:
    def __init__(self):
        self.processed_events = 0
        self.failed_events = 0

    def track_event(self, event_id: str, status: str):
        if status == "processed":
            self.processed_events += 1
        elif status == "failed":
            self.failed_events += 1

    def get_stats(self) -> Dict[str, int]:
        return {
            "processed": self.processed_events,
            "failed": self.failed_events,
            "success_rate": self.processed_events / (self.processed_events + self.failed_events) if (self.processed_events + self.failed_events) > 0 else 0
        }
```

**Debug Steps:**

1. Check event queue status
2. Monitor event processing logs
3. Implement event retry logic

## Knowledge Graph Layer Issues

### Neo4j Connection Issues

**Symptoms:**

- Database connection failures
- Query timeout errors

**Solutions:**

```python
# Implement connection pooling
from neo4j import GraphDatabase

class Neo4jConnectionPool:
    def __init__(self, uri: str, user: str, password: str, max_connections: int = 10):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=3600,
            max_connection_pool_size=max_connections
        )

    def get_session(self):
        return self.driver.session()

    def close(self):
        self.driver.close()
```

**Debug Steps:**

1. Check Neo4j server status
2. Verify connection parameters
3. Test database connectivity
4. Monitor connection pool usage

### Cypher Query Issues

**Symptoms:**

- Query syntax errors
- Performance issues

**Solutions:**

```python
# Implement query validation
def validate_cypher_query(query: str) -> bool:
    try:
        # Basic syntax validation
        if not query.strip().upper().startswith(("MATCH", "CREATE", "MERGE", "DELETE")):
            return False

        # Check for common issues
        if ";" in query:
            logger.warning("Semicolon in query may cause issues")

        return True
    except Exception as e:
        logger.error(f"Query validation failed: {e}")
        return False
```

**Debug Steps:**

1. Validate query syntax
2. Check query performance
3. Use query profiling tools
4. Optimize query structure

## Storage Layer Issues

### Vector Store Issues

**Symptoms:**

- Index creation failures
- Search performance issues

**Solutions:**

```python
# Implement vector store monitoring
class VectorStoreMonitor:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def check_health(self) -> Dict[str, Any]:
        try:
            # Test basic operations
            test_vector = [0.1] * 768
            result = self.vector_store.search(test_vector, k=1)

            return {
                "status": "healthy",
                "index_size": self.vector_store.get_index_size(),
                "search_time": result.get("search_time", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
```

**Debug Steps:**

1. Check vector store configuration
2. Monitor index size and performance
3. Test search operations
4. Implement health checks

### Object Store Issues

**Symptoms:**

- File upload failures
- Storage quota exceeded

**Solutions:**

```python
# Implement storage monitoring
class StorageMonitor:
    def __init__(self, storage_client):
        self.storage_client = storage_client

    def check_quota(self) -> Dict[str, Any]:
        try:
            usage = self.storage_client.get_usage()
            quota = self.storage_client.get_quota()

            return {
                "usage": usage,
                "quota": quota,
                "percentage": (usage / quota) * 100 if quota > 0 else 0
            }
        except Exception as e:
            return {"error": str(e)}
```

**Debug Steps:**

1. Check storage quota and usage
2. Monitor file upload success rates
3. Implement storage cleanup
4. Test storage operations

## Validation Layer Issues

### Schema Validation Failures

**Symptoms:**

- Data validation errors
- Schema mismatch warnings

**Solutions:**

```python
# Implement robust validation
from jsonschema import validate, ValidationError

def validate_data(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    try:
        validate(instance=data, schema=schema)
        return True
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        return False
```

**Debug Steps:**

1. Check data schema compliance
2. Validate required fields
3. Handle validation errors gracefully
4. Update schemas as needed

## Common Debugging Techniques

### Logging

**Enable Debug Logging:**

```python
import logging
import os

# Set debug level
logging.basicConfig(level=logging.DEBUG)

# Enable specific logger
logger = logging.getLogger("Medical_KG_rev")
logger.setLevel(logging.DEBUG)

# Add file handler
file_handler = logging.FileHandler("debug.log")
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
```

### Performance Profiling

**Profile Code Performance:**

```python
import cProfile
import pstats
from io import StringIO

def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        # Print profiling results
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        return result
    return wrapper
```

### Memory Profiling

**Monitor Memory Usage:**

```python
import tracemalloc
import psutil

def monitor_memory():
    # Start memory tracing
    tracemalloc.start()

    # Get current memory usage
    process = psutil.Process()
    memory_info = process.memory_info()

    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

    # Get memory snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
```

## Monitoring and Alerting

### Health Checks

**Implement Health Checks:**

```python
from enum import Enum
from typing import Dict, Any

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class SystemHealth:
    def __init__(self):
        self.components = {}

    def check_component(self, name: str, checker_func) -> HealthStatus:
        try:
            result = checker_func()
            if result:
                self.components[name] = HealthStatus.HEALTHY
                return HealthStatus.HEALTHY
            else:
                self.components[name] = HealthStatus.DEGRADED
                return HealthStatus.DEGRADED
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            self.components[name] = HealthStatus.UNHEALTHY
            return HealthStatus.UNHEALTHY

    def get_overall_health(self) -> HealthStatus:
        if any(status == HealthStatus.UNHEALTHY for status in self.components.values()):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in self.components.values()):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
```

### Metrics Collection

**Collect System Metrics:**

```python
from prometheus_client import Counter, Histogram, Gauge

# System metrics
SYSTEM_ERRORS = Counter('system_errors_total', 'Total system errors', ['component', 'error_type'])
SYSTEM_DURATION = Histogram('system_duration_seconds', 'System operation duration', ['component', 'operation'])
SYSTEM_HEALTH = Gauge('system_health', 'System health status', ['component'])

def track_system_operation(component: str, operation: str):
    """Track system operation metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with SYSTEM_DURATION.labels(component=component, operation=operation).time():
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    SYSTEM_ERRORS.labels(component=component, error_type=type(e).__name__).inc()
                    raise
        return wrapper
    return decorator
```

## Recovery Procedures

### Service Recovery

**Implement Service Recovery:**

```python
class ServiceRecovery:
    def __init__(self, service):
        self.service = service
        self.max_retries = 3
        self.retry_delay = 5

    def recover_service(self):
        """Attempt to recover a failed service."""
        for attempt in range(self.max_retries):
            try:
                self.service.initialize()
                logger.info(f"Service recovered on attempt {attempt + 1}")
                return True
            except Exception as e:
                logger.error(f"Recovery attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        logger.error("Service recovery failed after all attempts")
        return False
```

### Data Recovery

**Implement Data Recovery:**

```python
class DataRecovery:
    def __init__(self, storage_client):
        self.storage_client = storage_client

    def recover_data(self, backup_path: str):
        """Recover data from backup."""
        try:
            # Restore from backup
            self.storage_client.restore_from_backup(backup_path)
            logger.info("Data recovery completed successfully")
            return True
        except Exception as e:
            logger.error(f"Data recovery failed: {e}")
            return False
```

## Best Practices

1. **Proactive Monitoring**: Implement comprehensive monitoring and alerting
2. **Graceful Degradation**: Handle failures gracefully without system-wide impact
3. **Comprehensive Logging**: Log all important events and errors
4. **Health Checks**: Implement health checks for all components
5. **Recovery Procedures**: Have clear recovery procedures for common failures
6. **Performance Monitoring**: Monitor performance metrics and optimize accordingly
7. **Error Handling**: Implement robust error handling with specific error types
8. **Documentation**: Maintain up-to-date troubleshooting documentation
9. **Testing**: Test failure scenarios and recovery procedures
10. **Backup and Recovery**: Implement proper backup and recovery procedures
