# Medical_KG_rev Operational Runbook

## Overview

This runbook provides operational procedures for the Medical_KG_rev platform, focusing on storage, monitoring, and troubleshooting procedures.

## Table of Contents

1. [System Overview](#system-overview)
2. [Monitoring and Alerting](#monitoring-and-alerting)
3. [Storage Operations](#storage-operations)
4. [Pipeline Operations](#pipeline-operations)
5. [Troubleshooting](#troubleshooting)
6. [Emergency Procedures](#emergency-procedures)
7. [Maintenance Procedures](#maintenance-procedures)

## System Overview

### Architecture Components

- **API Gateway**: Multi-protocol API (REST, GraphQL, gRPC, SOAP, AsyncAPI)
- **Orchestration**: Dagster-based pipeline orchestration
- **Storage**: S3-compatible object storage + Redis caching
- **AI Services**: MinerU (PDF processing), vLLM (embeddings)
- **Knowledge Graph**: Neo4j graph database
- **Monitoring**: Prometheus + Grafana + Jaeger

### Key Services

| Service | Port | Purpose | Health Check |
|---------|------|---------|--------------|
| API Gateway | 8000 | Main API endpoint | `/health` |
| Dagster UI | 3000 | Pipeline management | `/health` |
| MinIO | 9000 | Object storage | `/minio/health/live` |
| Redis | 6379 | Caching | `PING` |
| Neo4j | 7474 | Graph database | `/health` |
| Prometheus | 9090 | Metrics collection | `/health` |
| Grafana | 3001 | Dashboards | `/health` |
| Jaeger | 16686 | Distributed tracing | `/health` |

## Monitoring and Alerting

### Key Metrics

#### Storage Metrics

- `storage_pdf_upload_total`: Total PDF uploads
- `storage_pdf_upload_duration_seconds`: Upload duration
- `storage_pdf_upload_size_bytes`: Upload size distribution
- `storage_cache_hit_total`: Cache hit rate
- `storage_cache_miss_total`: Cache miss rate

#### Pipeline Metrics

- `pipeline_pdf_download_total`: PDF download attempts
- `pipeline_pdf_download_success_total`: Successful downloads
- `pipeline_pdf_download_failure_total`: Failed downloads
- `pipeline_pdf_gate_open_total`: Gate open events
- `pipeline_pdf_gate_closed_total`: Gate closed events

#### System Metrics

- `http_requests_total`: API request count
- `http_request_duration_seconds`: API response time
- `dagster_job_runs_total`: Pipeline job runs
- `dagster_job_failures_total`: Pipeline failures

### Alerting Rules

#### Critical Alerts

```yaml
# High error rate
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"

# Storage failures
- alert: StorageFailures
  expr: rate(storage_pdf_upload_failure_total[5m]) > 0.05
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Storage upload failures detected"

# Pipeline failures
- alert: PipelineFailures
  expr: rate(dagster_job_failures_total[10m]) > 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Pipeline failures detected"
```

#### Warning Alerts

```yaml
# High latency
- alert: HighLatency
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High API latency detected"

# Low cache hit rate
- alert: LowCacheHitRate
  expr: rate(storage_cache_hit_total[5m]) / (rate(storage_cache_hit_total[5m]) + rate(storage_cache_miss_total[5m])) < 0.8
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Low cache hit rate detected"
```

### Dashboard URLs

- **Main Dashboard**: <http://grafana:3001/d/main>
- **Storage Dashboard**: <http://grafana:3001/d/storage>
- **Pipeline Dashboard**: <http://grafana:3001/d/pipeline>
- **API Dashboard**: <http://grafana:3001/d/api>

## Storage Operations

### Health Checks

#### S3/MinIO Health Check

```bash
#!/bin/bash
# Check MinIO health
curl -f http://minio:9000/minio/health/live || exit 1

# Check bucket accessibility
aws s3 ls s3://medical-kg-pdf --endpoint-url http://minio:9000 || exit 1

# Test upload/download
echo "test" | aws s3 cp - s3://medical-kg-pdf/health-check --endpoint-url http://minio:9000
aws s3 cp s3://medical-kg-pdf/health-check - --endpoint-url http://minio:9000
aws s3 rm s3://medical-kg-pdf/health-check --endpoint-url http://minio:9000
```

#### Redis Health Check

```bash
#!/bin/bash
# Check Redis connectivity
redis-cli -u "$REDIS_URL" ping || exit 1

# Check memory usage
redis-cli -u "$REDIS_URL" info memory | grep used_memory_human

# Check key count
redis-cli -u "$REDIS_URL" dbsize
```

### Storage Maintenance

#### Cleanup Old Files

```bash
#!/bin/bash
# Clean up files older than 30 days
aws s3 ls s3://medical-kg-pdf --recursive --endpoint-url http://minio:9000 | \
awk '$1 < "'$(date -d '30 days ago' --iso-8601)'" {print $4}' | \
xargs -I {} aws s3 rm s3://medical-kg-pdf/{} --endpoint-url http://minio:9000
```

#### Redis Memory Cleanup

```bash
#!/bin/bash
# Clear expired keys
redis-cli -u "$REDIS_URL" --scan --pattern "medical-kg:*" | \
xargs -I {} redis-cli -u "$REDIS_URL" expire {} 0

# Check memory usage after cleanup
redis-cli -u "$REDIS_URL" info memory | grep used_memory_human
```

### Backup Procedures

#### S3 Backup

```bash
#!/bin/bash
# Create S3 backup
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_BUCKET="medical-kg-pdf-backup"

# Sync current bucket to backup bucket
aws s3 sync s3://medical-kg-pdf s3://$BACKUP_BUCKET/$BACKUP_DATE \
  --endpoint-url http://minio:9000

# Verify backup
aws s3 ls s3://$BACKUP_BUCKET/$BACKUP_DATE --endpoint-url http://minio:9000
```

#### Redis Backup

```bash
#!/bin/bash
# Create Redis backup
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="/backups/redis_$BACKUP_DATE.rdb"

# Trigger Redis save
redis-cli -u "$REDIS_URL" BGSAVE

# Wait for save to complete
while [ "$(redis-cli -u "$REDIS_URL" LASTSAVE)" = "$(redis-cli -u "$REDIS_URL" LASTSAVE)" ]; do
  sleep 1
done

# Copy backup file
cp /var/lib/redis/dump.rdb "$BACKUP_FILE"
```

## Pipeline Operations

### Pipeline Management

#### Start Pipeline

```bash
#!/bin/bash
# Start PDF processing pipeline
dagster job execute --job pdf-two-phase --config config/orchestration/pipelines/pdf-two-phase.yaml
```

#### Monitor Pipeline

```bash
#!/bin/bash
# Check pipeline status
dagster job status --job pdf-two-phase

# View pipeline logs
dagster job logs --job pdf-two-phase --tail
```

#### Stop Pipeline

```bash
#!/bin/bash
# Stop running pipeline
dagster job cancel --job pdf-two-phase
```

### Pipeline Troubleshooting

#### Check Pipeline State

```python
#!/usr/bin/env python3
"""Check pipeline state and diagnose issues."""

import asyncio
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.storage.clients import create_storage_clients
from Medical_KG_rev.config.settings import ObjectStorageSettings, RedisCacheSettings

async def check_pipeline_state(job_id: str):
    # Check ledger state
    ledger = JobLedger()
    state = await ledger.get_job_state(job_id)
    print(f"Job {job_id} state: {state}")

    # Check storage
    object_settings = ObjectStorageSettings()
    redis_settings = RedisCacheSettings()
    clients = create_storage_clients(object_settings, redis_settings)

    # Check if PDF assets exist
    if "pdf_assets" in state.metadata:
        for asset in state.metadata["pdf_assets"]:
            if "storage_uri" in asset:
                print(f"PDF asset: {asset['storage_uri']}")

                # Check if file exists in storage
                try:
                    # Extract checksum from URI
                    checksum = asset["storage_uri"].split("/")[-1].replace(".pdf", "")
                    data = await clients.pdf_storage_client.get_pdf_data(
                        state.tenant_id, state.document_id, checksum
                    )
                    print(f"PDF exists in storage: {len(data)} bytes")
                except Exception as e:
                    print(f"PDF not found in storage: {e}")

if __name__ == "__main__":
    import sys
    job_id = sys.argv[1] if len(sys.argv) > 1 else "test-job"
    asyncio.run(check_pipeline_state(job_id))
```

## Troubleshooting

### Common Issues

#### 1. PDF Upload Failures

**Symptoms**:

- `storage_pdf_upload_failure_total` metric increasing
- Error logs showing "Storage error" or "S3 error"

**Diagnosis**:

```bash
# Check S3 connectivity
aws s3 ls s3://medical-kg-pdf --endpoint-url http://minio:9000

# Check bucket permissions
aws s3api get-bucket-acl --bucket medical-kg-pdf --endpoint-url http://minio:9000

# Check disk space on MinIO
df -h /var/lib/minio
```

**Resolution**:

1. Check MinIO service status: `docker-compose ps minio`
2. Restart MinIO if needed: `docker-compose restart minio`
3. Check bucket permissions and policies
4. Verify network connectivity between services

#### 2. Redis Connection Issues

**Symptoms**:

- `storage_cache_miss_total` metric increasing
- Error logs showing "Redis connection failed"

**Diagnosis**:

```bash
# Check Redis connectivity
redis-cli -u "$REDIS_URL" ping

# Check Redis memory usage
redis-cli -u "$REDIS_URL" info memory

# Check Redis logs
docker-compose logs redis
```

**Resolution**:

1. Check Redis service status: `docker-compose ps redis`
2. Restart Redis if needed: `docker-compose restart redis`
3. Check memory usage and clear expired keys
4. Verify Redis configuration

#### 3. Pipeline Stuck at PDF Gate

**Symptoms**:

- Pipeline jobs stuck in "running" state
- `pipeline_pdf_gate_closed_total` metric increasing
- No progress on PDF processing

**Diagnosis**:

```python
# Check PDF gate status
from Medical_KG_rev.orchestration.ledger import JobLedger

ledger = JobLedger()
state = await ledger.get_job_state("stuck-job-id")
print(f"PDF gate status: {state.pdf_gate}")
```

**Resolution**:

1. Check if PDF download completed successfully
2. Verify PDF exists in storage
3. Check MinerU service health
4. Manually trigger PDF gate if needed

#### 4. High Memory Usage

**Symptoms**:

- System memory usage > 80%
- Redis memory usage high
- Slow response times

**Diagnosis**:

```bash
# Check system memory
free -h

# Check Redis memory
redis-cli -u "$REDIS_URL" info memory

# Check Docker container memory
docker stats
```

**Resolution**:

1. Clear Redis cache: `redis-cli -u "$REDIS_URL" FLUSHDB`
2. Restart high-memory services
3. Check for memory leaks in application code
4. Scale up resources if needed

### Debug Commands

#### Storage Debug

```bash
#!/bin/bash
# Debug storage connectivity
echo "Testing S3 connectivity..."
aws s3 ls s3://medical-kg-pdf --endpoint-url http://minio:9000

echo "Testing Redis connectivity..."
redis-cli -u "$REDIS_URL" ping

echo "Testing PDF upload..."
echo "test" | aws s3 cp - s3://medical-kg-pdf/debug-test --endpoint-url http://minio:9000
aws s3 rm s3://medical-kg-pdf/debug-test --endpoint-url http://minio:9000
```

#### Pipeline Debug

```bash
#!/bin/bash
# Debug pipeline state
dagster job status --job pdf-two-phase

# Check pipeline logs
dagster job logs --job pdf-two-phase --tail 100

# Check specific job
dagster job status --job pdf-two-phase --run-id <run-id>
```

## Emergency Procedures

### Service Outage Response

#### 1. API Gateway Down

1. Check service status: `docker-compose ps gateway`
2. Check logs: `docker-compose logs gateway`
3. Restart service: `docker-compose restart gateway`
4. Check dependencies (Redis, Neo4j, etc.)
5. Notify team if issue persists

#### 2. Storage Service Down

1. Check MinIO status: `docker-compose ps minio`
2. Check disk space: `df -h`
3. Restart MinIO: `docker-compose restart minio`
4. Check Redis status: `docker-compose ps redis`
5. Restart Redis if needed: `docker-compose restart redis`
6. Verify data integrity after restart

#### 3. Pipeline Service Down

1. Check Dagster status: `docker-compose ps dagster`
2. Check logs: `docker-compose logs dagster`
3. Restart Dagster: `docker-compose restart dagster`
4. Check running jobs: `dagster job status`
5. Cancel stuck jobs if needed

### Data Recovery

#### 1. S3 Data Recovery

```bash
#!/bin/bash
# Restore from backup
BACKUP_DATE="20240101_120000"  # Replace with actual backup date
BACKUP_BUCKET="medical-kg-pdf-backup"

# Restore from backup
aws s3 sync s3://$BACKUP_BUCKET/$BACKUP_DATE s3://medical-kg-pdf \
  --endpoint-url http://minio:9000

# Verify restoration
aws s3 ls s3://medical-kg-pdf --endpoint-url http://minio:9000
```

#### 2. Redis Data Recovery

```bash
#!/bin/bash
# Restore Redis from backup
BACKUP_FILE="/backups/redis_20240101_120000.rdb"

# Stop Redis
docker-compose stop redis

# Copy backup file
cp "$BACKUP_FILE" /var/lib/redis/dump.rdb

# Start Redis
docker-compose start redis

# Verify restoration
redis-cli -u "$REDIS_URL" dbsize
```

### Rollback Procedures

#### 1. Application Rollback

```bash
#!/bin/bash
# Rollback to previous version
git checkout previous-version-tag
docker-compose build
docker-compose up -d

# Verify rollback
curl -f http://localhost:8000/health
```

#### 2. Configuration Rollback

```bash
#!/bin/bash
# Rollback configuration changes
git checkout HEAD~1 -- config/
docker-compose restart

# Verify configuration
docker-compose config
```

## Maintenance Procedures

### Regular Maintenance

#### Daily Tasks

- [ ] Check system health dashboards
- [ ] Review error logs
- [ ] Monitor storage usage
- [ ] Check pipeline job status

#### Weekly Tasks

- [ ] Review performance metrics
- [ ] Clean up old log files
- [ ] Check backup status
- [ ] Review security logs

#### Monthly Tasks

- [ ] Update system dependencies
- [ ] Review and update documentation
- [ ] Performance optimization review
- [ ] Security audit

### Scheduled Maintenance

#### Backup Schedule

- **S3 Backups**: Daily at 2 AM UTC
- **Redis Backups**: Daily at 3 AM UTC
- **Configuration Backups**: Weekly on Sundays

#### Cleanup Schedule

- **Old Files**: Daily cleanup of files older than 30 days
- **Expired Cache**: Daily cleanup of expired Redis keys
- **Log Files**: Weekly cleanup of old log files

### Performance Tuning

#### S3 Optimization

```bash
#!/bin/bash
# Optimize S3 performance
aws configure set default.s3.max_concurrent_requests 20
aws configure set default.s3.max_bandwidth 100MB/s
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 16MB
```

#### Redis Optimization

```bash
#!/bin/bash
# Optimize Redis performance
redis-cli -u "$REDIS_URL" CONFIG SET maxmemory-policy allkeys-lru
redis-cli -u "$REDIS_URL" CONFIG SET tcp-keepalive 60
redis-cli -u "$REDIS_URL" CONFIG SET timeout 300
```

## Contact Information

### Team Contacts

- **Platform Team**: <platform@company.com>
- **On-Call Engineer**: +1-555-0123
- **Emergency Contact**: +1-555-9999

### Escalation Procedures

1. **Level 1**: Platform Team (0-15 minutes)
2. **Level 2**: Senior Engineer (15-30 minutes)
3. **Level 3**: Engineering Manager (30-60 minutes)
4. **Level 4**: CTO (60+ minutes)

### External Dependencies

- **AWS Support**: Enterprise Support Plan
- **MinIO Support**: Community Support
- **Redis Support**: Enterprise Support Plan
- **Neo4j Support**: Enterprise Support Plan

## Appendix

### Useful Commands

#### Docker Commands

```bash
# View all containers
docker-compose ps

# View logs
docker-compose logs -f service-name

# Restart service
docker-compose restart service-name

# Scale service
docker-compose up -d --scale service-name=3

# Execute command in container
docker-compose exec service-name command
```

#### Monitoring Commands

```bash
# Check system resources
htop
df -h
free -h

# Check network connectivity
netstat -tulpn
ss -tulpn

# Check process status
ps aux | grep service-name
```

#### Storage Commands

```bash
# S3 operations
aws s3 ls s3://bucket-name --endpoint-url http://minio:9000
aws s3 cp local-file s3://bucket-name/remote-file --endpoint-url http://minio:9000
aws s3 rm s3://bucket-name/file --endpoint-url http://minio:9000

# Redis operations
redis-cli -u "$REDIS_URL" keys "pattern*"
redis-cli -u "$REDIS_URL" get "key"
redis-cli -u "$REDIS_URL" set "key" "value"
redis-cli -u "$REDIS_URL" del "key"
```

### Configuration Files

#### Environment Variables

```bash
# S3/MinIO
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export S3_ENDPOINT_URL=http://minio:9000
export S3_BUCKET=medical-kg-pdf

# Redis
export REDIS_URL=redis://redis:6379/0
export REDIS_PASSWORD=redis-password

# Application
export ENVIRONMENT=dev
export LOG_LEVEL=INFO
export CORRELATION_ID_HEADER=X-Correlation-ID
```

#### Docker Compose Overrides

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  minio:
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio-data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  redis:
    environment:
      REDIS_PASSWORD: redis-password
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"

volumes:
  minio-data:
  redis-data:
```

This runbook should be reviewed and updated regularly to reflect changes in the system architecture and operational procedures.
