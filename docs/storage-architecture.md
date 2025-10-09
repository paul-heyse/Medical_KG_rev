# Storage Architecture Documentation

## Overview

The Medical_KG_rev platform integrates S3-compatible object storage and Redis caching to provide durable, scalable storage for pipeline artifacts and metadata. This document describes the architecture, components, and operational procedures.

## Architecture Components

### 1. Object Storage (S3/MinIO)

**Purpose**: Durable storage for PDFs, MinerU artifacts, and other pipeline outputs.

**Key Features**:

- S3-compatible API support (AWS S3, MinIO, etc.)
- Configurable endpoints and credentials
- Automatic checksum generation and validation
- Presigned URL generation for secure access
- Metadata storage for provenance tracking

**Configuration**:

```python
ObjectStorageSettings(
    bucket="medical-kg-pdf",
    region="us-east-1",
    endpoint_url="http://minio:9000",  # Optional for MinIO
    access_key_id="minioadmin",
    secret_access_key="minioadmin",
    use_tls=False,  # For development
    max_file_size=100 * 1024 * 1024,  # 100MB
    key_prefix="pdf"
)
```

**Storage Layout**:

```
bucket/
├── pdf/
│   └── {tenant_id}/
│       └── {document_id}/
│           └── {checksum}.pdf
└── documents/
    └── {tenant_id}/
        └── {document_id}/
            └── {artifact_type}/
                └── {checksum}.{extension}
```

### 2. Redis Cache

**Purpose**: High-performance caching for metadata, checksums, and temporary data.

**Key Features**:

- In-memory data store with configurable TTL
- Connection pooling and failover support
- Key prefixing for multi-tenancy
- Automatic expiration handling

**Configuration**:

```python
RedisCacheSettings(
    url="redis://redis:6379/0",
    password="redis-password",  # Optional
    use_tls=False,  # For development
    db_index=0,
    key_prefix="medical-kg",
    default_ttl=3600,  # 1 hour
    max_connections=10
)
```

**Cache Layout**:

```
{key_prefix}:pdf:{tenant_id}:{document_id}:{checksum} -> PdfAsset
{key_prefix}:metadata:{tenant_id}:{document_id} -> dict
{key_prefix}:checksum:{hash} -> str
```

## Client Architecture

### 1. PdfStorageClient

**Purpose**: Typed client for PDF-specific storage operations.

**Key Methods**:

- `store_pdf()`: Upload PDF with metadata caching
- `get_pdf_asset()`: Retrieve PDF metadata from cache
- `get_pdf_data()`: Download PDF data from storage
- `get_presigned_url()`: Generate secure access URLs
- `delete_pdf()`: Remove PDF and associated metadata

**Usage Example**:

```python
from Medical_KG_rev.storage.clients import create_storage_clients

# Create clients
clients = create_storage_clients(object_settings, redis_settings)

# Store PDF
asset = await clients.pdf_storage_client.store_pdf(
    tenant_id="tenant-123",
    document_id="doc-456",
    pdf_data=pdf_bytes,
    content_type="application/pdf"
)

# Retrieve PDF
pdf_data = await clients.pdf_storage_client.get_pdf_data(
    tenant_id="tenant-123",
    document_id="doc-456",
    checksum=asset.checksum
)
```

### 2. DocumentStorageClient

**Purpose**: General-purpose storage for document artifacts.

**Key Methods**:

- `upload_document_artifact()`: Store artifacts (MinerU output, etc.)
- `get_document_artifact()`: Retrieve stored artifacts

**Usage Example**:

```python
# Store MinerU output
uri = await clients.document_storage_client.upload_document_artifact(
    tenant_id="tenant-123",
    document_id="doc-456",
    artifact_type="mineru_output",
    data=json_bytes,
    file_extension="json"
)

# Retrieve artifact
data = await clients.document_storage_client.get_document_artifact(
    tenant_id="tenant-123",
    document_id="doc-456",
    artifact_type="mineru_output",
    checksum="abc123",
    file_extension="json"
)
```

## Pipeline Integration

### 1. PDF Download Stage

**Purpose**: Download PDFs from external sources and store them persistently.

**Key Features**:

- HTTP retry logic with exponential backoff
- Automatic content type detection
- Error handling and logging
- Storage URI generation

**Configuration**:

```yaml
# config/orchestration/pipelines/pdf-two-phase.yaml
stages:
  - name: pdf-download
    type: pdf-download
    config:
      retry_attempts: 3
      timeout: 30
      max_file_size: 100MB
```

**Input State**:

```python
PipelineState(
    metadata={
        "pdf_urls": [
            "https://example.com/paper1.pdf",
            "https://example.com/paper2.pdf"
        ]
    }
)
```

**Output State**:

```python
PipelineState(
    metadata={
        "pdf_assets": [
            {
                "url": "https://example.com/paper1.pdf",
                "storage_uri": "s3://bucket/pdf/tenant/doc/abc123.pdf",
                "checksum": "abc123",
                "size": 1024,
                "content_type": "application/pdf",
                "error": None
            }
        ]
    }
)
```

### 2. PDF Gate Stage

**Purpose**: Conditional pipeline progression based on PDF readiness.

**Key Features**:

- Checks `pdf_gate.ir_ready` status
- Blocks pipeline until PDF is processed
- Supports both sync and async execution

**Configuration**:

```yaml
stages:
  - name: pdf-gate
    type: pdf-gate
    config:
      gate_name: "pdf-ir-gate"
```

### 3. MinerU Integration

**Purpose**: Process PDFs stored in object storage.

**Key Features**:

- Fetches PDFs from S3 using cached metadata
- Supports both content and storage URI inputs
- Automatic checksum extraction from URIs

**Usage Example**:

```python
# MinerU service with storage support
processor = MineruProcessor(
    pdf_storage=clients.pdf_storage_client
)

# Process PDF from storage
request = MineruRequest(
    storage_uri="s3://bucket/pdf/tenant/doc/abc123.pdf",
    content=None  # Will be fetched from storage
)

result = await processor.process_async(request)
```

## Adapter Integration

### 1. StorageHelperMixin

**Purpose**: Provide storage capabilities to adapters.

**Key Features**:

- Optional PDF upload functionality
- Error handling and logging
- Protocol compliance checking

**Usage Example**:

```python
from Medical_KG_rev.adapters.mixins.storage_helpers import StorageHelperMixin

class MyAdapter(BaseAdapter, StorageHelperMixin):
    async def fetch_and_upload_pdf(self, context, pdf_url, document_id):
        # Fetch PDF data
        pdf_data = await self._fetch_pdf_data(pdf_url)

        # Upload to storage if available
        storage_uri = await self.upload_pdf_if_available(
            tenant_id=context.tenant_id,
            document_id=document_id,
            pdf_data=pdf_data
        )

        return storage_uri
```

## Configuration Management

### 1. Environment-Specific Settings

**Development**:

```python
ENVIRONMENT_DEFAULTS[Environment.DEV] = {
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
```

**Production**:

```python
ENVIRONMENT_DEFAULTS[Environment.PROD] = {
    "object_storage": {
        "use_tls": True,
    },
    "redis_cache": {
        "use_tls": True,
    }
}
```

### 2. Secrets Management

**Environment Variables**:

```bash
# S3/MinIO
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
S3_ENDPOINT_URL=http://minio:9000
S3_BUCKET=medical-kg-pdf

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=redis-password
```

**Vault Integration** (Production):

```python
# Load secrets from Vault
vault_client = hvac.Client(url="https://vault.example.com")
secrets = vault_client.secrets.kv.v2.read_secret_version(path="medical-kg/storage")

object_storage = ObjectStorageSettings(
    access_key_id=secrets["data"]["data"]["s3_access_key"],
    secret_access_key=secrets["data"]["data"]["s3_secret_key"],
    # ...
)
```

## Monitoring and Observability

### 1. Metrics

**Storage Metrics**:

- `storage_pdf_upload_total`: Total PDF uploads
- `storage_pdf_upload_duration_seconds`: Upload duration
- `storage_pdf_upload_size_bytes`: Upload size distribution
- `storage_cache_hit_total`: Cache hit rate
- `storage_cache_miss_total`: Cache miss rate

**Pipeline Metrics**:

- `pipeline_pdf_download_total`: PDF download attempts
- `pipeline_pdf_download_success_total`: Successful downloads
- `pipeline_pdf_download_failure_total`: Failed downloads
- `pipeline_pdf_gate_open_total`: Gate open events
- `pipeline_pdf_gate_closed_total`: Gate closed events

### 2. Logging

**Structured Logging**:

```python
logger.info(
    "storage.pdf_uploaded",
    tenant_id=tenant_id,
    document_id=document_id,
    s3_key=asset.s3_key,
    size=len(pdf_data),
    checksum=asset.checksum,
    duration=upload_time
)
```

**Error Logging**:

```python
logger.warning(
    "storage.pdf_upload_failed",
    tenant_id=tenant_id,
    document_id=document_id,
    error=str(e),
    retry_count=retry_count
)
```

### 3. Tracing

**OpenTelemetry Spans**:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("storage.upload_pdf") as span:
    span.set_attribute("tenant_id", tenant_id)
    span.set_attribute("document_id", document_id)
    span.set_attribute("file_size", len(pdf_data))

    # Upload PDF
    asset = await client.store_pdf(...)

    span.set_attribute("s3_key", asset.s3_key)
    span.set_attribute("checksum", asset.checksum)
```

## Operational Procedures

### 1. Health Checks

**Storage Health Check**:

```python
async def check_storage_health(clients: StorageClients) -> bool:
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
        logger.error("storage.health_check_failed", error=str(e))
        return False
```

### 2. Backup and Recovery

**S3 Backup Strategy**:

- Enable versioning on S3 buckets
- Configure cross-region replication
- Set up lifecycle policies for cost optimization
- Regular backup validation

**Redis Backup Strategy**:

- Enable Redis persistence (RDB + AOF)
- Regular snapshot backups
- Cross-region replication for disaster recovery

### 3. Capacity Planning

**Storage Growth Estimation**:

```python
# Estimate storage growth
estimated_pdfs_per_month = 10000
average_pdf_size = 2 * 1024 * 1024  # 2MB
monthly_growth = estimated_pdfs_per_month * average_pdf_size

# Plan for 6 months retention
total_capacity_needed = monthly_growth * 6
```

**Redis Memory Planning**:

```python
# Estimate Redis memory usage
estimated_metadata_per_pdf = 1024  # 1KB
estimated_pdfs = 100000
total_metadata_size = estimated_pdfs * estimated_metadata_per_pdf

# Plan for 2x overhead
redis_memory_needed = total_metadata_size * 2
```

## Troubleshooting

### 1. Common Issues

**S3 Connection Errors**:

- Check endpoint URL and credentials
- Verify network connectivity
- Check bucket permissions

**Redis Connection Errors**:

- Verify Redis URL and password
- Check Redis server status
- Verify connection pool settings

**PDF Upload Failures**:

- Check file size limits
- Verify S3 bucket permissions
- Check network timeouts

### 2. Debugging Tools

**Storage Debug Script**:

```python
#!/usr/bin/env python3
"""Debug storage connectivity and functionality."""

import asyncio
from Medical_KG_rev.storage.clients import create_storage_clients
from Medical_KG_rev.config.settings import ObjectStorageSettings, RedisCacheSettings

async def debug_storage():
    # Create clients
    object_settings = ObjectStorageSettings()
    redis_settings = RedisCacheSettings()
    clients = create_storage_clients(object_settings, redis_settings)

    # Test PDF storage
    test_data = b"test pdf content"
    asset = await clients.pdf_storage_client.store_pdf(
        tenant_id="debug",
        document_id="test",
        pdf_data=test_data
    )
    print(f"Uploaded PDF: {asset.uri}")

    # Test retrieval
    retrieved = await clients.pdf_storage_client.get_pdf_data(
        "debug", "test", asset.checksum
    )
    print(f"Retrieved PDF: {len(retrieved)} bytes")

    # Test cache
    await clients.cache_backend.set("debug:test", {"status": "ok"})
    cached = await clients.cache_backend.get("debug:test")
    print(f"Cached value: {cached}")

if __name__ == "__main__":
    asyncio.run(debug_storage())
```

**Redis Debug Script**:

```bash
#!/bin/bash
# Debug Redis connectivity

redis-cli -u "$REDIS_URL" ping
redis-cli -u "$REDIS_URL" info memory
redis-cli -u "$REDIS_URL" keys "medical-kg:*" | head -10
```

## Security Considerations

### 1. Access Control

**S3 Bucket Policies**:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::account:role/MedicalKGService"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::medical-kg-pdf/*"
        }
    ]
}
```

**Redis ACL**:

```redis
# Redis 6+ ACL configuration
user medical-kg on >password +@all ~medical-kg:*
```

### 2. Encryption

**S3 Encryption**:

- Enable server-side encryption (SSE-S3 or SSE-KMS)
- Use HTTPS for all S3 operations
- Encrypt data in transit

**Redis Encryption**:

- Enable TLS for Redis connections
- Use strong passwords
- Consider Redis AUTH for additional security

### 3. Audit Logging

**S3 Access Logging**:

- Enable S3 access logging
- Monitor for unusual access patterns
- Set up alerts for failed access attempts

**Redis Monitoring**:

- Monitor Redis slow log
- Track command usage patterns
- Set up alerts for memory usage

## Performance Optimization

### 1. S3 Optimization

**Multipart Uploads**:

```python
# For large files (>100MB)
await client.store_pdf(
    tenant_id=tenant_id,
    document_id=document_id,
    pdf_data=large_pdf_data,
    use_multipart=True  # Future enhancement
)
```

**Connection Pooling**:

```python
# Configure S3 client with connection pooling
s3_client = boto3.client(
    "s3",
    config=Config(
        max_pool_connections=50,
        retries={"max_attempts": 3}
    )
)
```

### 2. Redis Optimization

**Connection Pooling**:

```python
redis_client = Redis(
    max_connections=20,
    retry_on_timeout=True,
    socket_keepalive=True
)
```

**Memory Optimization**:

```redis
# Redis configuration
maxmemory 2gb
maxmemory-policy allkeys-lru
```

### 3. Caching Strategies

**Cache Warming**:

```python
# Pre-load frequently accessed data
async def warm_cache(clients: StorageClients):
    popular_docs = await get_popular_documents()
    for doc in popular_docs:
        await clients.cache_backend.set(
            f"pdf:{doc.tenant_id}:{doc.document_id}",
            doc.metadata,
            ttl=3600
        )
```

**Cache Invalidation**:

```python
# Invalidate cache on document updates
async def invalidate_document_cache(clients: StorageClients, tenant_id: str, document_id: str):
    pattern = f"medical-kg:pdf:{tenant_id}:{document_id}:*"
    await clients.cache_backend.delete_pattern(pattern)
```

## Future Enhancements

### 1. Planned Features

- **CDN Integration**: CloudFront distribution for global PDF access
- **Compression**: Automatic PDF compression for storage optimization
- **Deduplication**: Content-based deduplication to reduce storage costs
- **Archive Tiering**: Automatic migration to cheaper storage classes

### 2. Scalability Improvements

- **Horizontal Scaling**: Multiple Redis instances with clustering
- **Load Balancing**: S3 request distribution across regions
- **Async Processing**: Background tasks for large file operations

### 3. Monitoring Enhancements

- **Custom Dashboards**: Grafana dashboards for storage metrics
- **Alerting**: PagerDuty integration for critical issues
- **Cost Monitoring**: AWS Cost Explorer integration

## Conclusion

The storage architecture provides a robust, scalable foundation for the Medical_KG_rev platform. By combining S3-compatible object storage with Redis caching, the system ensures data durability, performance, and operational efficiency.

Key benefits:

- **Durability**: S3 provides 99.999999999% durability
- **Performance**: Redis caching reduces latency for frequently accessed data
- **Scalability**: Both components scale horizontally
- **Cost-effectiveness**: Lifecycle policies and caching reduce costs
- **Operational simplicity**: Standard APIs and tooling

For questions or issues, refer to the troubleshooting section or contact the platform team.
