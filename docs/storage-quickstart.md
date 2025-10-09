# Storage Quick Start Guide

## Overview

This guide provides a quick start for using the S3-compatible object storage and Redis caching features in Medical_KG_rev.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.12+ environment
- Basic understanding of S3 and Redis concepts

## Local Development Setup

### 1. Start Storage Services

```bash
# Start MinIO (S3-compatible) and Redis
docker-compose up -d minio redis

# Verify services are running
docker-compose ps minio redis
```

### 2. Configure Environment Variables

```bash
# S3/MinIO configuration
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export S3_ENDPOINT_URL=http://localhost:9000
export S3_BUCKET=medical-kg-pdf

# Redis configuration
export REDIS_URL=redis://localhost:6379/0
```

### 3. Create S3 Bucket

```bash
# Create bucket using AWS CLI
aws s3 mb s3://medical-kg-pdf --endpoint-url http://localhost:9000

# Verify bucket creation
aws s3 ls --endpoint-url http://localhost:9000
```

## Basic Usage

### 1. Storage Client Setup

```python
from Medical_KG_rev.storage.clients import create_storage_clients
from Medical_KG_rev.config.settings import ObjectStorageSettings, RedisCacheSettings

# Create storage clients
object_settings = ObjectStorageSettings(
    bucket="medical-kg-pdf",
    endpoint_url="http://localhost:9000",
    access_key_id="minioadmin",
    secret_access_key="minioadmin",
    use_tls=False
)

redis_settings = RedisCacheSettings(
    url="redis://localhost:6379/0",
    use_tls=False
)

clients = create_storage_clients(object_settings, redis_settings)
```

### 2. PDF Storage Operations

```python
# Store a PDF
pdf_data = b"fake pdf content"
asset = await clients.pdf_storage_client.store_pdf(
    tenant_id="tenant-123",
    document_id="doc-456",
    pdf_data=pdf_data,
    content_type="application/pdf"
)

print(f"Stored PDF: {asset.uri}")
print(f"Checksum: {asset.checksum}")
print(f"Size: {asset.size} bytes")

# Retrieve PDF data
pdf_data = await clients.pdf_storage_client.get_pdf_data(
    tenant_id="tenant-123",
    document_id="doc-456",
    checksum=asset.checksum
)

print(f"Retrieved PDF: {len(pdf_data)} bytes")

# Generate presigned URL
presigned_url = clients.pdf_storage_client.get_presigned_url(
    tenant_id="tenant-123",
    document_id="doc-456",
    checksum=asset.checksum,
    expires_in=3600
)

print(f"Presigned URL: {presigned_url}")
```

### 3. Document Artifact Storage

```python
# Store MinerU output
mineru_output = b'{"pages": 10, "figures": 3, "tables": 2}'
uri = await clients.document_storage_client.upload_document_artifact(
    tenant_id="tenant-123",
    document_id="doc-456",
    artifact_type="mineru_output",
    data=mineru_output,
    file_extension="json"
)

print(f"Stored artifact: {uri}")

# Retrieve artifact
artifact_data = await clients.document_storage_client.get_document_artifact(
    tenant_id="tenant-123",
    document_id="doc-456",
    artifact_type="mineru_output",
    checksum="abc123",
    file_extension="json"
)

print(f"Retrieved artifact: {len(artifact_data)} bytes")
```

### 4. Cache Operations

```python
# Store data in cache
await clients.cache_backend.set(
    "test:key",
    {"message": "Hello, World!", "timestamp": "2024-01-01T00:00:00Z"},
    ttl=3600
)

# Retrieve data from cache
cached_data = await clients.cache_backend.get("test:key")
print(f"Cached data: {cached_data}")

# Check if key exists
exists = await clients.cache_backend.exists("test:key")
print(f"Key exists: {exists}")

# Delete key
await clients.cache_backend.delete("test:key")
```

## Pipeline Integration

### 1. PDF Download Stage

```python
from Medical_KG_rev.orchestration.stages.pdf_download import StorageAwarePdfDownloadStage
from Medical_KG_rev.orchestration.stages.types import PipelineState

# Create PDF download stage
stage = StorageAwarePdfDownloadStage(pdf_storage=clients.pdf_storage_client)

# Execute stage
pipeline_state = PipelineState(
    job_id="test-job",
    tenant_id="tenant-123",
    document_id="doc-456",
    metadata={
        "pdf_urls": [
            "https://example.com/paper1.pdf",
            "https://example.com/paper2.pdf"
        ]
    }
)

result = await stage.execute(pipeline_state)
print(f"Downloaded {len(result.metadata['pdf_assets'])} PDFs")
```

### 2. PDF Gate Stage

```python
from Medical_KG_rev.orchestration.stages.pdf_gate import SimplePdfGateStage
from Medical_KG_rev.orchestration.stages.types import PdfGateStatus

# Create PDF gate stage
gate = SimplePdfGateStage()

# Execute gate
pipeline_state = PipelineState(
    job_id="test-job",
    tenant_id="tenant-123",
    document_id="doc-456",
    pdf_gate=PdfGateStatus(ir_ready=True)
)

result = await gate.execute(pipeline_state)
print(f"Gate status: {result.pdf_gate.ir_ready}")
```

## Adapter Integration

### 1. Storage Helper Mixin

```python
from Medical_KG_rev.adapters.mixins.storage_helpers import StorageHelperMixin
from Medical_KG_rev.adapters.base import BaseAdapter

class MyAdapter(BaseAdapter, StorageHelperMixin):
    def __init__(self):
        super().__init__(name="my-adapter")
        # Storage client will be injected via context

    async def fetch_and_upload_pdf(self, context, pdf_url, document_id):
        # Fetch PDF data
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url)
            pdf_data = response.content

        # Upload to storage if available
        storage_uri = await self.upload_pdf_if_available(
            tenant_id=context.tenant_id,
            document_id=document_id,
            pdf_data=pdf_data
        )

        return storage_uri
```

### 2. OpenAlex Adapter with Storage

```python
from Medical_KG_rev.adapters.openalex.adapter import OpenAlexAdapter
from Medical_KG_rev.adapters.base import AdapterContext

# Create adapter with storage
adapter = OpenAlexAdapter()
adapter._pdf_storage = clients.pdf_storage_client

# Fetch and upload PDF
context = AdapterContext(
    tenant_id="tenant-123",
    domain="biomedical",
    correlation_id="corr-456"
)

storage_uri = await adapter.fetch_and_upload_pdf(
    context=context,
    pdf_url="https://example.com/paper.pdf",
    document_id="doc-789"
)

print(f"PDF stored at: {storage_uri}")
```

## MinerU Integration

### 1. MinerU Service with Storage

```python
from Medical_KG_rev.services.mineru.service import MineruProcessor
from Medical_KG_rev.services.mineru.types import MineruRequest

# Create MinerU processor with storage
processor = MineruProcessor(
    pdf_storage=clients.pdf_storage_client
)

# Process PDF from storage
request = MineruRequest(
    storage_uri="s3://medical-kg-pdf/pdf/tenant-123/doc-456/abc123.pdf",
    content=None  # Will be fetched from storage
)

result = await processor.process_async(request)
print(f"Processed PDF: {result.pages} pages, {result.figures} figures")
```

## Configuration Examples

### 1. Development Configuration

```python
# config/settings.py
ENVIRONMENT_DEFAULTS = {
    Environment.DEV: {
        "object_storage": {
            "endpoint_url": "http://minio:9000",
            "bucket": "medical-kg-pdf",
            "use_tls": False,
        },
        "redis_cache": {
            "url": "redis://redis:6379/0",
            "use_tls": False,
        }
    }
}
```

### 2. Production Configuration

```python
# config/settings.py
ENVIRONMENT_DEFAULTS = {
    Environment.PROD: {
        "object_storage": {
            "bucket": "medical-kg-pdf-prod",
            "region": "us-east-1",
            "use_tls": True,
        },
        "redis_cache": {
            "url": "redis://redis-cluster:6379/0",
            "use_tls": True,
            "max_connections": 20,
        }
    }
}
```

## Testing

### 1. Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from Medical_KG_rev.storage.clients import PdfStorageClient

@pytest.mark.asyncio
async def test_pdf_storage():
    # Mock storage client
    mock_store = MagicMock()
    mock_store.put = AsyncMock()

    client = PdfStorageClient(mock_store, settings)

    # Test PDF upload
    pdf_data = b"test pdf content"
    asset = await client.store_pdf(
        tenant_id="test-tenant",
        document_id="test-doc",
        pdf_data=pdf_data
    )

    assert asset.checksum is not None
    assert asset.size == len(pdf_data)
```

### 2. Integration Tests

```python
import pytest
from moto import mock_s3

@pytest.mark.integration
@mock_s3
async def test_s3_integration():
    # Test with real S3 operations
    clients = create_storage_clients(object_settings, redis_settings)

    # Test PDF storage
    pdf_data = b"integration test pdf"
    asset = await clients.pdf_storage_client.store_pdf(
        tenant_id="integration-tenant",
        document_id="integration-doc",
        pdf_data=pdf_data
    )

    # Test retrieval
    retrieved_data = await clients.pdf_storage_client.get_pdf_data(
        "integration-tenant", "integration-doc", asset.checksum
    )

    assert retrieved_data == pdf_data
```

## Monitoring

### 1. Health Checks

```python
async def check_storage_health(clients):
    try:
        # Test S3 connectivity
        await clients.object_store.put("health-check", b"test")
        await clients.object_store.get("health-check")
        await clients.object_store.delete("health-check")

        # Test Redis connectivity
        await clients.cache_backend.set("health-check", "test", ttl=60)
        value = await clients.cache_backend.get("health-check")
        assert value == "test"

        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
```

### 2. Metrics

```python
from prometheus_client import Counter, Histogram

# Storage metrics
pdf_uploads_total = Counter('storage_pdf_upload_total', 'Total PDF uploads')
pdf_upload_duration = Histogram('storage_pdf_upload_duration_seconds', 'PDF upload duration')
cache_hits_total = Counter('storage_cache_hit_total', 'Cache hits')
cache_misses_total = Counter('storage_cache_miss_total', 'Cache misses')

# Usage in code
with pdf_upload_duration.time():
    asset = await client.store_pdf(...)
    pdf_uploads_total.inc()
```

## Troubleshooting

### 1. Common Issues

**S3 Connection Errors**:

```bash
# Check MinIO status
docker-compose ps minio

# Check bucket permissions
aws s3 ls s3://medical-kg-pdf --endpoint-url http://localhost:9000
```

**Redis Connection Errors**:

```bash
# Check Redis status
docker-compose ps redis

# Test Redis connectivity
redis-cli -u "redis://localhost:6379/0" ping
```

### 2. Debug Scripts

```python
#!/usr/bin/env python3
"""Debug storage connectivity."""

import asyncio
from Medical_KG_rev.storage.clients import create_storage_clients
from Medical_KG_rev.config.settings import ObjectStorageSettings, RedisCacheSettings

async def debug_storage():
    # Create clients
    object_settings = ObjectStorageSettings(
        endpoint_url="http://localhost:9000",
        bucket="medical-kg-pdf"
    )
    redis_settings = RedisCacheSettings(url="redis://localhost:6379/0")
    clients = create_storage_clients(object_settings, redis_settings)

    # Test PDF storage
    test_data = b"debug test pdf"
    asset = await clients.pdf_storage_client.store_pdf(
        tenant_id="debug",
        document_id="test",
        pdf_data=test_data
    )
    print(f"Uploaded PDF: {asset.uri}")

    # Test cache
    await clients.cache_backend.set("debug:test", {"status": "ok"})
    cached = await clients.cache_backend.get("debug:test")
    print(f"Cached value: {cached}")

if __name__ == "__main__":
    asyncio.run(debug_storage())
```

## Next Steps

1. **Explore Documentation**: Read the full [Storage Architecture Documentation](storage-architecture.md)
2. **Operational Procedures**: Review the [Operational Runbook](operational-runbook.md)
3. **Integration Examples**: Check the test files for more usage examples
4. **Monitoring Setup**: Configure Prometheus and Grafana for production monitoring
5. **Security Configuration**: Set up proper access controls and encryption

## Support

For questions or issues:

- Check the troubleshooting section above
- Review the operational runbook
- Contact the platform team
- Open an issue in the repository

Happy coding! 🚀
