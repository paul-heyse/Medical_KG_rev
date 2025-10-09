# Maintenance Procedures

This document provides comprehensive procedures for routine maintenance tasks, troubleshooting, and system optimization for the Medical_KG_rev system.

## Table of Contents

1. [Routine Maintenance Tasks](#routine-maintenance-tasks)
2. [Database Maintenance](#database-maintenance)
3. [System Monitoring](#system-monitoring)
4. [Performance Optimization](#performance-optimization)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Backup and Recovery](#backup-and-recovery)
7. [Security Maintenance](#security-maintenance)
8. [Update Procedures](#update-procedures)
9. [Emergency Procedures](#emergency-procedures)
10. [Maintenance Checklists](#maintenance-checklists)

## Routine Maintenance Tasks

### Daily Tasks

#### System Health Check

```bash
#!/bin/bash
# scripts/maintenance/daily_health_check.sh

set -e

LOG_FILE="/var/log/maintenance/daily_health_check.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "$DATE - Starting daily health check" >> "$LOG_FILE"

# Check system resources
echo "Checking system resources..." >> "$LOG_FILE"
df -h >> "$LOG_FILE"
free -h >> "$LOG_FILE"
top -bn1 | head -20 >> "$LOG_FILE"

# Check service status
echo "Checking service status..." >> "$LOG_FILE"
kubectl get pods -n medical-kg-rev >> "$LOG_FILE"
kubectl get services -n medical-kg-rev >> "$LOG_FILE"

# Check database connections
echo "Checking database connections..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -c "SELECT COUNT(*) FROM pg_stat_activity;" >> "$LOG_FILE"

# Check Redis status
echo "Checking Redis status..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/redis-client -- redis-cli -h redis-primary ping >> "$LOG_FILE"

# Check Neo4j status
echo "Checking Neo4j status..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/neo4j-client -- cypher-shell -u neo4j -p password -a bolt://neo4j-primary:7687 "RETURN 1;" >> "$LOG_FILE"

# Check Qdrant status
echo "Checking Qdrant status..." >> "$LOG_FILE"
curl -s http://qdrant-primary:6333/health >> "$LOG_FILE"

# Check API endpoints
echo "Checking API endpoints..." >> "$LOG_FILE"
curl -s http://api-gateway:8000/health >> "$LOG_FILE"

echo "$DATE - Daily health check completed" >> "$LOG_FILE"
```

#### Log Rotation

```bash
#!/bin/bash
# scripts/maintenance/log_rotation.sh

set -e

LOG_DIR="/var/log/medical-kg-rev"
RETENTION_DAYS=30

echo "Starting log rotation..."

# Rotate application logs
find "$LOG_DIR" -name "*.log" -type f -mtime +$RETENTION_DAYS -delete

# Compress old logs
find "$LOG_DIR" -name "*.log" -type f -mtime +7 -exec gzip {} \;

# Clean up compressed logs older than retention period
find "$LOG_DIR" -name "*.log.gz" -type f -mtime +$RETENTION_DAYS -delete

# Rotate Kubernetes logs
kubectl logs --since=24h -n medical-kg-rev > "$LOG_DIR/k8s_logs_$(date +%Y%m%d).log"

echo "Log rotation completed"
```

#### Database Cleanup

```bash
#!/bin/bash
# scripts/maintenance/database_cleanup.sh

set -e

echo "Starting database cleanup..."

# Clean up old processing jobs
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
DELETE FROM ingestion_jobs
WHERE status = 'completed'
AND created_at < NOW() - INTERVAL '30 days';
"

# Clean up old audit logs
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
DELETE FROM audit_logs
WHERE created_at < NOW() - INTERVAL '90 days';
"

# Clean up old search queries
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
DELETE FROM search_queries
WHERE created_at < NOW() - INTERVAL '7 days';
"

# Vacuum and analyze tables
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
VACUUM ANALYZE documents;
VACUUM ANALYZE entities;
VACUUM ANALYZE extractions;
VACUUM ANALYZE claims;
"

echo "Database cleanup completed"
```

### Weekly Tasks

#### System Backup

```bash
#!/bin/bash
# scripts/maintenance/weekly_backup.sh

set -e

BACKUP_DIR="/backups/weekly"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/maintenance/backup.log"

echo "$(date) - Starting weekly backup" >> "$LOG_FILE"

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup PostgreSQL
echo "Backing up PostgreSQL..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- pg_dump -h postgresql-primary -U postgres -d medical_kg_rev > "$BACKUP_DIR/$DATE/postgresql_backup.sql"

# Backup Neo4j
echo "Backing up Neo4j..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/neo4j-client -- neo4j-admin dump --database=neo4j --to=/tmp/neo4j_backup.dump
kubectl cp medical-kg-rev/neo4j-client-pod:/tmp/neo4j_backup.dump "$BACKUP_DIR/$DATE/neo4j_backup.dump"

# Backup Redis
echo "Backing up Redis..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/redis-client -- redis-cli -h redis-primary --rdb /tmp/redis_backup.rdb
kubectl cp medical-kg-rev/redis-client-pod:/tmp/redis_backup.rdb "$BACKUP_DIR/$DATE/redis_backup.rdb"

# Backup Qdrant
echo "Backing up Qdrant..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/qdrant-client -- curl -X POST http://qdrant-primary:6333/collections/backup -d '{"name": "backup_'$DATE'"}'

# Compress backup
echo "Compressing backup..." >> "$LOG_FILE"
tar -czf "$BACKUP_DIR/$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"
rm -rf "$BACKUP_DIR/$DATE"

# Upload to cloud storage (if configured)
if [ -n "$CLOUD_STORAGE_URL" ]; then
    echo "Uploading to cloud storage..." >> "$LOG_FILE"
    aws s3 cp "$BACKUP_DIR/$DATE.tar.gz" "$CLOUD_STORAGE_URL/weekly/"
fi

# Clean up old backups (keep 4 weeks)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +28 -delete

echo "$(date) - Weekly backup completed" >> "$LOG_FILE"
```

#### Performance Analysis

```bash
#!/bin/bash
# scripts/maintenance/performance_analysis.sh

set -e

LOG_FILE="/var/log/maintenance/performance_analysis.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "$DATE - Starting performance analysis" >> "$LOG_FILE"

# Database performance
echo "Analyzing database performance..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;
" >> "$LOG_FILE"

# Slow query analysis
echo "Analyzing slow queries..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
SELECT
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
" >> "$LOG_FILE"

# Index usage analysis
echo "Analyzing index usage..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
SELECT
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_tup_read DESC;
" >> "$LOG_FILE"

# Redis memory usage
echo "Analyzing Redis memory usage..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/redis-client -- redis-cli -h redis-primary info memory >> "$LOG_FILE"

# Neo4j performance
echo "Analyzing Neo4j performance..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/neo4j-client -- cypher-shell -u neo4j -p password -a bolt://neo4j-primary:7687 "
CALL dbms.listQueries() YIELD query, elapsedTimeMillis
ORDER BY elapsedTimeMillis DESC
LIMIT 10;
" >> "$LOG_FILE"

echo "$DATE - Performance analysis completed" >> "$LOG_FILE"
```

### Monthly Tasks

#### Security Audit

```bash
#!/bin/bash
# scripts/maintenance/security_audit.sh

set -e

LOG_FILE="/var/log/maintenance/security_audit.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "$DATE - Starting security audit" >> "$LOG_FILE"

# Check for security updates
echo "Checking for security updates..." >> "$LOG_FILE"
apt list --upgradable | grep -i security >> "$LOG_FILE"

# Check SSL certificate expiration
echo "Checking SSL certificates..." >> "$LOG_FILE"
kubectl get secrets -n medical-kg-rev -o json | jq -r '.items[] | select(.type=="kubernetes.io/tls") | .metadata.name' | while read secret; do
    kubectl get secret "$secret" -n medical-kg-rev -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -noout -dates >> "$LOG_FILE"
done

# Check for failed login attempts
echo "Checking failed login attempts..." >> "$LOG_FILE"
kubectl logs -n medical-kg-rev deployment/api-gateway --since=720h | grep -i "failed\|error\|unauthorized" | wc -l >> "$LOG_FILE"

# Check for suspicious activity
echo "Checking for suspicious activity..." >> "$LOG_FILE"
kubectl logs -n medical-kg-rev deployment/api-gateway --since=720h | grep -i "sql injection\|xss\|csrf" >> "$LOG_FILE"

# Check user permissions
echo "Checking user permissions..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
SELECT
    usename,
    usesuper,
    usecreatedb,
    usebypassrls
FROM pg_user;
" >> "$LOG_FILE"

# Check database access logs
echo "Checking database access logs..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
SELECT
    usename,
    application_name,
    client_addr,
    state,
    COUNT(*)
FROM pg_stat_activity
GROUP BY usename, application_name, client_addr, state;
" >> "$LOG_FILE"

echo "$DATE - Security audit completed" >> "$LOG_FILE"
```

#### Capacity Planning

```bash
#!/bin/bash
# scripts/maintenance/capacity_planning.sh

set -e

LOG_FILE="/var/log/maintenance/capacity_planning.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "$DATE - Starting capacity planning analysis" >> "$LOG_FILE"

# Database size analysis
echo "Analyzing database sizes..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
" >> "$LOG_FILE"

# Growth rate analysis
echo "Analyzing growth rates..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
SELECT
    'documents' as table_name,
    COUNT(*) as total_count,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days') as last_30_days,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days') as last_7_days
FROM documents
UNION ALL
SELECT
    'entities' as table_name,
    COUNT(*) as total_count,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days') as last_30_days,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days') as last_7_days
FROM entities;
" >> "$LOG_FILE"

# Storage usage
echo "Analyzing storage usage..." >> "$LOG_FILE"
kubectl get pv -o json | jq -r '.items[] | "\(.metadata.name): \(.spec.capacity.storage)"' >> "$LOG_FILE"

# Memory usage trends
echo "Analyzing memory usage trends..." >> "$LOG_FILE"
kubectl top pods -n medical-kg-rev >> "$LOG_FILE"

# CPU usage trends
echo "Analyzing CPU usage trends..." >> "$LOG_FILE"
kubectl top nodes >> "$LOG_FILE"

echo "$DATE - Capacity planning analysis completed" >> "$LOG_FILE"
```

## Database Maintenance

### PostgreSQL Maintenance

#### Regular Maintenance Tasks

```sql
-- Daily maintenance script
-- Analyze tables for query optimization
ANALYZE documents;
ANALYZE entities;
ANALYZE extractions;
ANALYZE claims;

-- Check for long-running queries
SELECT
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
AND state = 'active';

-- Check for locks
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

#### Index Maintenance

```sql
-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Check for unused indexes
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0;

-- Reindex frequently used indexes
REINDEX INDEX CONCURRENTLY idx_documents_title;
REINDEX INDEX CONCURRENTLY idx_entities_name;
REINDEX INDEX CONCURRENTLY idx_extractions_document_id;
```

#### Table Maintenance

```sql
-- Vacuum and analyze tables
VACUUM ANALYZE documents;
VACUUM ANALYZE entities;
VACUUM ANALYZE extractions;
VACUUM ANALYZE claims;

-- Check table bloat
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Neo4j Maintenance

#### Graph Database Maintenance

```cypher
// Check database size
CALL db.info();

// Check node and relationship counts
MATCH (n) RETURN labels(n) as label, count(n) as count ORDER BY count DESC;

// Check relationship counts
MATCH ()-[r]->() RETURN type(r) as relationship_type, count(r) as count ORDER BY count DESC;

// Check for orphaned nodes
MATCH (n) WHERE NOT (n)--() RETURN count(n) as orphaned_nodes;

// Check for duplicate relationships
MATCH (a)-[r1]->(b), (a)-[r2]->(b)
WHERE r1 <> r2 AND type(r1) = type(r2)
RETURN count(*) as duplicate_relationships;

// Analyze database
CALL db.analyze();

// Check for long-running queries
CALL dbms.listQueries() YIELD query, elapsedTimeMillis
WHERE elapsedTimeMillis > 30000
RETURN query, elapsedTimeMillis
ORDER BY elapsedTimeMillis DESC;
```

#### Index Maintenance

```cypher
// List all indexes
SHOW INDEXES;

// Check index usage
CALL db.indexes() YIELD name, state, populationPercent, type
RETURN name, state, populationPercent, type;

// Create indexes for frequently queried properties
CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id);
CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type);

// Drop unused indexes
DROP INDEX entity_unused_index IF EXISTS;
```

### Redis Maintenance

#### Redis Maintenance Tasks

```bash
#!/bin/bash
# Redis maintenance script

# Check Redis memory usage
redis-cli -h redis-primary info memory

# Check Redis keyspace
redis-cli -h redis-primary info keyspace

# Check for expired keys
redis-cli -h redis-primary --scan --pattern "*" | wc -l

# Clean up expired keys
redis-cli -h redis-primary --scan --pattern "*" | xargs redis-cli -h redis-primary del

# Check Redis configuration
redis-cli -h redis-primary config get "*"

# Monitor Redis commands
redis-cli -h redis-primary monitor
```

#### Cache Optimization

```python
# Redis cache optimization script
import redis
import json
from datetime import datetime, timedelta

class RedisMaintenance:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)

    def analyze_cache_usage(self):
        """Analyze cache usage patterns."""
        info = self.redis_client.info()

        print(f"Memory usage: {info['used_memory_human']}")
        print(f"Memory peak: {info['used_memory_peak_human']}")
        print(f"Keyspace hits: {info['keyspace_hits']}")
        print(f"Keyspace misses: {info['keyspace_misses']}")

        hit_rate = info['keyspace_hits'] / (info['keyspace_hits'] + info['keyspace_misses'])
        print(f"Hit rate: {hit_rate:.2%}")

    def cleanup_expired_keys(self):
        """Clean up expired keys."""
        keys = self.redis_client.keys("*")
        expired_count = 0

        for key in keys:
            ttl = self.redis_client.ttl(key)
            if ttl == -1:  # No expiration set
                continue
            elif ttl == -2:  # Key expired
                expired_count += 1

        print(f"Found {expired_count} expired keys")
        return expired_count

    def optimize_memory(self):
        """Optimize Redis memory usage."""
        # Set maxmemory policy
        self.redis_client.config_set('maxmemory-policy', 'allkeys-lru')

        # Enable memory optimization
        self.redis_client.config_set('hash-max-ziplist-entries', 512)
        self.redis_client.config_set('hash-max-ziplist-value', 64)

        print("Memory optimization configured")

    def backup_cache(self, backup_file: str):
        """Backup cache data."""
        keys = self.redis_client.keys("*")
        backup_data = {}

        for key in keys:
            value = self.redis_client.get(key)
            if value:
                backup_data[key] = value.decode('utf-8')

        with open(backup_file, 'w') as f:
            json.dump(backup_data, f)

        print(f"Cache backed up to {backup_file}")

# Usage
redis_maintenance = RedisMaintenance("redis://redis-primary:6379")
redis_maintenance.analyze_cache_usage()
redis_maintenance.cleanup_expired_keys()
redis_maintenance.optimize_memory()
```

## System Monitoring

### Health Monitoring

#### Comprehensive Health Check

```python
import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class HealthMonitor:
    """System health monitoring."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_status = {}

    async def check_all_services(self) -> Dict[str, Any]:
        """Check health of all services."""
        tasks = [
            self.check_database(),
            self.check_redis(),
            self.check_neo4j(),
            self.check_qdrant(),
            self.check_api_gateway(),
            self.check_document_service(),
            self.check_search_service()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "services": {}
        }

        service_names = [
            "database", "redis", "neo4j", "qdrant",
            "api_gateway", "document_service", "search_service"
        ]

        for i, result in enumerate(results):
            service_name = service_names[i]
            if isinstance(result, Exception):
                health_status["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(result)
                }
                health_status["overall_status"] = "unhealthy"
            else:
                health_status["services"][service_name] = result

        return health_status

    async def check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database health."""
        try:
            # Implementation would check database connection
            return {
                "status": "healthy",
                "response_time_ms": 50,
                "connections": 10,
                "max_connections": 100
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            raise

    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            # Implementation would check Redis connection
            return {
                "status": "healthy",
                "response_time_ms": 5,
                "memory_usage": "256MB",
                "connected_clients": 5
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            raise

    async def check_neo4j(self) -> Dict[str, Any]:
        """Check Neo4j health."""
        try:
            # Implementation would check Neo4j connection
            return {
                "status": "healthy",
                "response_time_ms": 100,
                "node_count": 1000000,
                "relationship_count": 5000000
            }
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            raise

    async def check_qdrant(self) -> Dict[str, Any]:
        """Check Qdrant health."""
        try:
            # Implementation would check Qdrant connection
            return {
                "status": "healthy",
                "response_time_ms": 200,
                "collection_count": 10,
                "vector_count": 500000
            }
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            raise

    async def check_api_gateway(self) -> Dict[str, Any]:
        """Check API gateway health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['api_gateway_url']}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return {
                            "status": "healthy",
                            "response_time_ms": 100,
                            "version": "1.0.0"
                        }
                    else:
                        raise Exception(f"HTTP {response.status}")
        except Exception as e:
            logger.error(f"API gateway health check failed: {e}")
            raise

    async def check_document_service(self) -> Dict[str, Any]:
        """Check document service health."""
        try:
            # Implementation would check document service
            return {
                "status": "healthy",
                "response_time_ms": 150,
                "queue_size": 5,
                "processing_rate": 10
            }
        except Exception as e:
            logger.error(f"Document service health check failed: {e}")
            raise

    async def check_search_service(self) -> Dict[str, Any]:
        """Check search service health."""
        try:
            # Implementation would check search service
            return {
                "status": "healthy",
                "response_time_ms": 300,
                "index_size": "2GB",
                "search_latency_p95": 450
            }
        except Exception as e:
            logger.error(f"Search service health check failed: {e}")
            raise
```

### Performance Monitoring

#### Performance Metrics Collection

```python
import time
import psutil
import asyncio
from typing import Dict, Any
from datetime import datetime

class PerformanceMonitor:
    """Performance monitoring and metrics collection."""

    def __init__(self):
        self.metrics = {}

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=1),
                "load_average": psutil.getloadavg(),
                "cpu_count": psutil.cpu_count()
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "used_percent": psutil.virtual_memory().percent,
                "swap_total": psutil.swap_memory().total,
                "swap_used": psutil.swap_memory().used
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "usage_percent": psutil.disk_usage('/').percent
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
                "packets_sent": psutil.net_io_counters().packets_sent,
                "packets_recv": psutil.net_io_counters().packets_recv
            }
        }

        return metrics

    async def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "api": {
                "requests_per_second": await self.get_requests_per_second(),
                "average_response_time": await self.get_average_response_time(),
                "error_rate": await self.get_error_rate()
            },
            "database": {
                "connections": await self.get_database_connections(),
                "query_time": await self.get_average_query_time(),
                "slow_queries": await self.get_slow_queries_count()
            },
            "cache": {
                "hit_rate": await self.get_cache_hit_rate(),
                "memory_usage": await self.get_cache_memory_usage(),
                "evictions": await self.get_cache_evictions()
            },
            "search": {
                "index_size": await self.get_search_index_size(),
                "search_latency": await self.get_search_latency(),
                "queries_per_second": await self.get_search_qps()
            }
        }

        return metrics

    async def get_requests_per_second(self) -> float:
        """Get API requests per second."""
        # Implementation would query metrics
        return 150.0

    async def get_average_response_time(self) -> float:
        """Get average API response time."""
        # Implementation would query metrics
        return 250.0

    async def get_error_rate(self) -> float:
        """Get API error rate."""
        # Implementation would query metrics
        return 0.02

    async def get_database_connections(self) -> int:
        """Get current database connections."""
        # Implementation would query database
        return 25

    async def get_average_query_time(self) -> float:
        """Get average database query time."""
        # Implementation would query database
        return 50.0

    async def get_slow_queries_count(self) -> int:
        """Get count of slow queries."""
        # Implementation would query database
        return 5

    async def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        # Implementation would query Redis
        return 0.85

    async def get_cache_memory_usage(self) -> int:
        """Get cache memory usage."""
        # Implementation would query Redis
        return 256 * 1024 * 1024  # 256MB

    async def get_cache_evictions(self) -> int:
        """Get cache evictions count."""
        # Implementation would query Redis
        return 100

    async def get_search_index_size(self) -> int:
        """Get search index size."""
        # Implementation would query Qdrant
        return 2 * 1024 * 1024 * 1024  # 2GB

    async def get_search_latency(self) -> float:
        """Get search latency."""
        # Implementation would query metrics
        return 300.0

    async def get_search_qps(self) -> float:
        """Get search queries per second."""
        # Implementation would query metrics
        return 50.0
```

## Performance Optimization

### Database Optimization

#### Query Optimization

```sql
-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS)
SELECT d.title, e.name, e.type
FROM documents d
JOIN document_entities de ON d.id = de.document_id
JOIN entities e ON de.entity_id = e.id
WHERE d.source = 'pubmed'
AND d.created_at >= '2024-01-01'
ORDER BY d.created_at DESC
LIMIT 100;

-- Check for missing indexes
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
AND n_distinct > 100
ORDER BY n_distinct DESC;

-- Create composite indexes for common queries
CREATE INDEX CONCURRENTLY idx_documents_source_created_at
ON documents (source, created_at DESC);

CREATE INDEX CONCURRENTLY idx_entities_name_type
ON entities (name, type);

-- Optimize table statistics
ANALYZE documents;
ANALYZE entities;
ANALYZE extractions;
```

#### Connection Pool Optimization

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import asyncio

class DatabaseOptimizer:
    """Database optimization utilities."""

    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )

    async def optimize_connections(self):
        """Optimize database connections."""
        # Check current connection usage
        with self.engine.connect() as conn:
            result = conn.execute("""
                SELECT
                    state,
                    COUNT(*) as count
                FROM pg_stat_activity
                WHERE datname = 'medical_kg_rev'
                GROUP BY state
            """)

            for row in result:
                print(f"{row.state}: {row.count} connections")

    async def analyze_slow_queries(self):
        """Analyze slow queries."""
        with self.engine.connect() as conn:
            result = conn.execute("""
                SELECT
                    query,
                    calls,
                    total_time,
                    mean_time,
                    rows
                FROM pg_stat_statements
                ORDER BY total_time DESC
                LIMIT 10
            """)

            print("Slow queries:")
            for row in result:
                print(f"Query: {row.query[:100]}...")
                print(f"Calls: {row.calls}, Total time: {row.total_time}ms")
                print(f"Mean time: {row.mean_time}ms, Rows: {row.rows}")
                print("-" * 50)

    async def optimize_indexes(self):
        """Optimize database indexes."""
        with self.engine.connect() as conn:
            # Check for unused indexes
            result = conn.execute("""
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0
            """)

            print("Unused indexes:")
            for row in result:
                print(f"{row.schemaname}.{row.tablename}.{row.indexname}")

            # Check for duplicate indexes
            result = conn.execute("""
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """)

            print("\nAll indexes:")
            for row in result:
                print(f"{row.schemaname}.{row.tablename}.{row.indexname}")
                print(f"  {row.indexdef}")
```

### Cache Optimization

#### Redis Cache Optimization

```python
import redis
import json
from typing import Dict, Any, List
import time

class CacheOptimizer:
    """Redis cache optimization utilities."""

    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)

    def analyze_cache_patterns(self) -> Dict[str, Any]:
        """Analyze cache usage patterns."""
        info = self.redis_client.info()

        analysis = {
            "memory_usage": {
                "used": info['used_memory_human'],
                "peak": info['used_memory_peak_human'],
                "fragmentation_ratio": info['mem_fragmentation_ratio']
            },
            "keyspace": {
                "hits": info['keyspace_hits'],
                "misses": info['keyspace_misses'],
                "hit_rate": info['keyspace_hits'] / (info['keyspace_hits'] + info['keyspace_misses'])
            },
            "operations": {
                "total_commands": info['total_commands_processed'],
                "instantaneous_ops": info['instantaneous_ops_per_sec']
            }
        }

        return analysis

    def optimize_memory_usage(self):
        """Optimize Redis memory usage."""
        # Set memory policy
        self.redis_client.config_set('maxmemory-policy', 'allkeys-lru')

        # Optimize hash settings
        self.redis_client.config_set('hash-max-ziplist-entries', 512)
        self.redis_client.config_set('hash-max-ziplist-value', 64)

        # Optimize list settings
        self.redis_client.config_set('list-max-ziplist-size', 512)

        # Optimize set settings
        self.redis_client.config_set('set-max-intset-entries', 512)

        print("Redis memory optimization configured")

    def cleanup_expired_keys(self) -> int:
        """Clean up expired keys."""
        keys = self.redis_client.keys("*")
        expired_count = 0

        for key in keys:
            ttl = self.redis_client.ttl(key)
            if ttl == -2:  # Key expired
                expired_count += 1

        if expired_count > 0:
            print(f"Found {expired_count} expired keys")

        return expired_count

    def analyze_key_patterns(self) -> Dict[str, int]:
        """Analyze key patterns and usage."""
        keys = self.redis_client.keys("*")
        patterns = {}

        for key in keys:
            # Extract pattern (e.g., "user:123" -> "user:*")
            if ':' in key:
                pattern = key.split(':')[0] + ':*'
            else:
                pattern = key

            patterns[pattern] = patterns.get(pattern, 0) + 1

        return patterns

    def optimize_cache_strategy(self):
        """Optimize cache strategy based on usage patterns."""
        patterns = self.analyze_key_patterns()

        print("Key patterns:")
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count} keys")

        # Set appropriate TTL for different patterns
        for pattern in patterns:
            if pattern.startswith('session:'):
                # Session keys should expire quickly
                self.redis_client.config_set('timeout', 3600)
            elif pattern.startswith('cache:'):
                # Cache keys should have longer TTL
                self.redis_client.config_set('timeout', 86400)
```

## Troubleshooting Guide

### Common Issues

#### Database Connection Issues

```bash
#!/bin/bash
# Database troubleshooting script

echo "=== Database Connection Troubleshooting ==="

# Check PostgreSQL status
echo "1. Checking PostgreSQL status..."
kubectl get pods -n medical-kg-rev -l app=postgresql

# Check PostgreSQL logs
echo "2. Checking PostgreSQL logs..."
kubectl logs -n medical-kg-rev deployment/postgresql-primary --tail=50

# Check database connections
echo "3. Checking database connections..."
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -c "
SELECT
    state,
    COUNT(*) as count
FROM pg_stat_activity
GROUP BY state;
"

# Check for locks
echo "4. Checking for database locks..."
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -c "
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
"

# Check database size
echo "5. Checking database size..."
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -c "
SELECT
    pg_size_pretty(pg_database_size('medical_kg_rev')) as database_size;
"
```

#### API Performance Issues

```bash
#!/bin/bash
# API performance troubleshooting script

echo "=== API Performance Troubleshooting ==="

# Check API gateway status
echo "1. Checking API gateway status..."
kubectl get pods -n medical-kg-rev -l app=api-gateway

# Check API gateway logs
echo "2. Checking API gateway logs..."
kubectl logs -n medical-kg-rev deployment/api-gateway --tail=50

# Check API response times
echo "3. Checking API response times..."
curl -w "@curl-format.txt" -o /dev/null -s "http://api-gateway:8000/health"

# Check API metrics
echo "4. Checking API metrics..."
curl -s "http://api-gateway:8000/metrics" | grep -E "(http_requests_total|http_request_duration_seconds)"

# Check for slow queries
echo "5. Checking for slow queries..."
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -c "
SELECT
    query,
    calls,
    total_time,
    mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 5;
"
```

#### Search Performance Issues

```bash
#!/bin/bash
# Search performance troubleshooting script

echo "=== Search Performance Troubleshooting ==="

# Check search service status
echo "1. Checking search service status..."
kubectl get pods -n medical-kg-rev -l app=search-service

# Check search service logs
echo "2. Checking search service logs..."
kubectl logs -n medical-kg-rev deployment/search-service --tail=50

# Check Qdrant status
echo "3. Checking Qdrant status..."
kubectl get pods -n medical-kg-rev -l app=qdrant

# Check Qdrant collections
echo "4. Checking Qdrant collections..."
curl -s "http://qdrant-primary:6333/collections" | jq '.'

# Check search index size
echo "5. Checking search index size..."
curl -s "http://qdrant-primary:6333/collections/documents" | jq '.result.config.params.vectors.size'

# Test search performance
echo "6. Testing search performance..."
time curl -s "http://api-gateway:8000/v1/search/documents?q=diabetes&limit=10"
```

### Error Analysis

#### Log Analysis

```python
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import subprocess

class LogAnalyzer:
    """Log analysis utilities."""

    def __init__(self, namespace: str = "medical-kg-rev"):
        self.namespace = namespace

    def analyze_errors(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze errors in logs."""
        since_time = datetime.now() - timedelta(hours=hours)

        # Get logs from all services
        services = [
            "api-gateway",
            "document-service",
            "search-service",
            "kg-service"
        ]

        error_summary = {
            "total_errors": 0,
            "error_types": {},
            "services": {},
            "time_range": f"Last {hours} hours"
        }

        for service in services:
            try:
                logs = self.get_service_logs(service, since_time)
                errors = self.extract_errors(logs)

                error_summary["services"][service] = {
                    "error_count": len(errors),
                    "errors": errors
                }

                error_summary["total_errors"] += len(errors)

                # Categorize errors
                for error in errors:
                    error_type = self.categorize_error(error)
                    error_summary["error_types"][error_type] = \
                        error_summary["error_types"].get(error_type, 0) + 1

            except Exception as e:
                print(f"Error analyzing {service}: {e}")

        return error_summary

    def get_service_logs(self, service: str, since_time: datetime) -> List[str]:
        """Get logs from a specific service."""
        cmd = [
            "kubectl", "logs", "-n", self.namespace,
            f"deployment/{service}",
            f"--since={since_time.isoformat()}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.split('\n')

    def extract_errors(self, logs: List[str]) -> List[Dict[str, Any]]:
        """Extract error messages from logs."""
        errors = []

        error_patterns = [
            r'ERROR',
            r'Exception',
            r'Traceback',
            r'Failed',
            r'Error:',
            r'panic'
        ]

        for line in logs:
            for pattern in error_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    errors.append({
                        "timestamp": self.extract_timestamp(line),
                        "message": line.strip(),
                        "level": "ERROR"
                    })
                    break

        return errors

    def extract_timestamp(self, log_line: str) -> str:
        """Extract timestamp from log line."""
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
        match = re.search(timestamp_pattern, log_line)
        return match.group(0) if match else ""

    def categorize_error(self, error: Dict[str, Any]) -> str:
        """Categorize error type."""
        message = error["message"].lower()

        if "database" in message or "connection" in message:
            return "Database Error"
        elif "timeout" in message:
            return "Timeout Error"
        elif "memory" in message or "oom" in message:
            return "Memory Error"
        elif "permission" in message or "unauthorized" in message:
            return "Permission Error"
        elif "validation" in message:
            return "Validation Error"
        else:
            return "Other Error"

    def analyze_performance_issues(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance issues from logs."""
        since_time = datetime.now() - timedelta(hours=hours)

        performance_issues = {
            "slow_queries": [],
            "high_memory_usage": [],
            "timeout_errors": [],
            "connection_issues": []
        }

        # Get API gateway logs
        logs = self.get_service_logs("api-gateway", since_time)

        for line in logs:
            # Check for slow queries
            if "slow query" in line.lower():
                performance_issues["slow_queries"].append(line.strip())

            # Check for high memory usage
            if "memory" in line.lower() and ("high" in line.lower() or "critical" in line.lower()):
                performance_issues["high_memory_usage"].append(line.strip())

            # Check for timeout errors
            if "timeout" in line.lower():
                performance_issues["timeout_errors"].append(line.strip())

            # Check for connection issues
            if "connection" in line.lower() and ("failed" in line.lower() or "error" in line.lower()):
                performance_issues["connection_issues"].append(line.strip())

        return performance_issues

# Usage
log_analyzer = LogAnalyzer()
error_summary = log_analyzer.analyze_errors(24)
performance_issues = log_analyzer.analyze_performance_issues(24)
```

## Backup and Recovery

### Backup Procedures

#### Automated Backup Script

```bash
#!/bin/bash
# Comprehensive backup script

set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/maintenance/backup.log"

echo "$(date) - Starting backup procedure" >> "$LOG_FILE"

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup PostgreSQL
echo "Backing up PostgreSQL..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- pg_dump \
    -h postgresql-primary -U postgres -d medical_kg_rev \
    --format=custom --compress=9 \
    > "$BACKUP_DIR/$DATE/postgresql_backup.dump"

# Backup Neo4j
echo "Backing up Neo4j..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/neo4j-client -- neo4j-admin dump \
    --database=neo4j --to=/tmp/neo4j_backup.dump
kubectl cp medical-kg-rev/neo4j-client-pod:/tmp/neo4j_backup.dump \
    "$BACKUP_DIR/$DATE/neo4j_backup.dump"

# Backup Redis
echo "Backing up Redis..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/redis-client -- redis-cli \
    -h redis-primary --rdb /tmp/redis_backup.rdb
kubectl cp medical-kg-rev/redis-client-pod:/tmp/redis_backup.rdb \
    "$BACKUP_DIR/$DATE/redis_backup.rdb"

# Backup Qdrant
echo "Backing up Qdrant..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/qdrant-client -- curl \
    -X POST "http://qdrant-primary:6333/collections/backup" \
    -d "{\"name\": \"backup_$DATE\"}"

# Backup configuration files
echo "Backing up configuration..." >> "$LOG_FILE"
kubectl get configmaps -n medical-kg-rev -o yaml > "$BACKUP_DIR/$DATE/configmaps.yaml"
kubectl get secrets -n medical-kg-rev -o yaml > "$BACKUP_DIR/$DATE/secrets.yaml"

# Backup Kubernetes manifests
echo "Backing up Kubernetes manifests..." >> "$LOG_FILE"
kubectl get all -n medical-kg-rev -o yaml > "$BACKUP_DIR/$DATE/kubernetes_manifests.yaml"

# Compress backup
echo "Compressing backup..." >> "$LOG_FILE"
tar -czf "$BACKUP_DIR/$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"
rm -rf "$BACKUP_DIR/$DATE"

# Upload to cloud storage
if [ -n "$CLOUD_STORAGE_URL" ]; then
    echo "Uploading to cloud storage..." >> "$LOG_FILE"
    aws s3 cp "$BACKUP_DIR/$DATE.tar.gz" "$CLOUD_STORAGE_URL/backups/"
fi

# Clean up old backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

echo "$(date) - Backup procedure completed" >> "$LOG_FILE"
```

### Recovery Procedures

#### Database Recovery

```bash
#!/bin/bash
# Database recovery script

set -e

BACKUP_FILE="$1"
RECOVERY_DIR="/tmp/recovery"
LOG_FILE="/var/log/maintenance/recovery.log"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

echo "$(date) - Starting recovery procedure" >> "$LOG_FILE"

# Extract backup
echo "Extracting backup..." >> "$LOG_FILE"
mkdir -p "$RECOVERY_DIR"
tar -xzf "$BACKUP_FILE" -C "$RECOVERY_DIR"

# Stop services
echo "Stopping services..." >> "$LOG_FILE"
kubectl scale deployment --replicas=0 -n medical-kg-rev --all

# Wait for pods to terminate
kubectl wait --for=delete pod -l app=api-gateway -n medical-kg-rev --timeout=60s

# Restore PostgreSQL
echo "Restoring PostgreSQL..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- pg_restore \
    -h postgresql-primary -U postgres -d medical_kg_rev \
    --clean --if-exists \
    < "$RECOVERY_DIR/postgresql_backup.dump"

# Restore Neo4j
echo "Restoring Neo4j..." >> "$LOG_FILE"
kubectl cp "$RECOVERY_DIR/neo4j_backup.dump" \
    medical-kg-rev/neo4j-client-pod:/tmp/neo4j_backup.dump
kubectl exec -n medical-kg-rev deployment/neo4j-client -- neo4j-admin restore \
    --from=/tmp/neo4j_backup.dump --database=neo4j --force

# Restore Redis
echo "Restoring Redis..." >> "$LOG_FILE"
kubectl cp "$RECOVERY_DIR/redis_backup.rdb" \
    medical-kg-rev/redis-client-pod:/tmp/redis_backup.rdb
kubectl exec -n medical-kg-rev deployment/redis-client -- redis-cli \
    -h redis-primary --rdb /tmp/redis_backup.rdb

# Restart services
echo "Restarting services..." >> "$LOG_FILE"
kubectl scale deployment --replicas=3 -n medical-kg-rev --all

# Wait for services to be ready
kubectl wait --for=condition=ready pod -l app=api-gateway -n medical-kg-rev --timeout=300s

# Clean up
rm -rf "$RECOVERY_DIR"

echo "$(date) - Recovery procedure completed" >> "$LOG_FILE"
```

## Security Maintenance

### Security Updates

#### Security Patch Management

```bash
#!/bin/bash
# Security patch management script

set -e

LOG_FILE="/var/log/maintenance/security_patches.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "$DATE - Starting security patch management" >> "$LOG_FILE"

# Check for security updates
echo "Checking for security updates..." >> "$LOG_FILE"
apt list --upgradable | grep -i security >> "$LOG_FILE"

# Check Docker image vulnerabilities
echo "Checking Docker image vulnerabilities..." >> "$LOG_FILE"
for image in $(kubectl get pods -n medical-kg-rev -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u); do
    echo "Scanning image: $image" >> "$LOG_FILE"
    trivy image --severity HIGH,CRITICAL "$image" >> "$LOG_FILE"
done

# Check Kubernetes security
echo "Checking Kubernetes security..." >> "$LOG_FILE"
kubectl auth can-i --list --namespace=medical-kg-rev >> "$LOG_FILE"

# Check for exposed secrets
echo "Checking for exposed secrets..." >> "$LOG_FILE"
kubectl get secrets -n medical-kg-rev -o json | jq -r '.items[] | select(.data) | .metadata.name' >> "$LOG_FILE"

# Check SSL certificate expiration
echo "Checking SSL certificates..." >> "$LOG_FILE"
kubectl get secrets -n medical-kg-rev -o json | jq -r '.items[] | select(.type=="kubernetes.io/tls") | .metadata.name' | while read secret; do
    kubectl get secret "$secret" -n medical-kg-rev -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -noout -dates >> "$LOG_FILE"
done

# Check for security misconfigurations
echo "Checking for security misconfigurations..." >> "$LOG_FILE"
kubectl get pods -n medical-kg-rev -o json | jq -r '.items[] | select(.spec.securityContext.runAsUser == 0) | .metadata.name' >> "$LOG_FILE"

echo "$DATE - Security patch management completed" >> "$LOG_FILE"
```

### Access Control

#### User Access Review

```bash
#!/bin/bash
# User access review script

set -e

LOG_FILE="/var/log/maintenance/access_review.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "$DATE - Starting user access review" >> "$LOG_FILE"

# Review Kubernetes RBAC
echo "Reviewing Kubernetes RBAC..." >> "$LOG_FILE"
kubectl get roles -n medical-kg-rev >> "$LOG_FILE"
kubectl get rolebindings -n medical-kg-rev >> "$LOG_FILE"
kubectl get clusterroles >> "$LOG_FILE"
kubectl get clusterrolebindings >> "$LOG_FILE"

# Review database users
echo "Reviewing database users..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -c "
SELECT
    usename,
    usesuper,
    usecreatedb,
    usebypassrls,
    valuntil
FROM pg_user;
" >> "$LOG_FILE"

# Review API users
echo "Reviewing API users..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
SELECT
    username,
    email,
    roles,
    last_login,
    created_at
FROM users
ORDER BY last_login DESC;
" >> "$LOG_FILE"

# Review inactive users
echo "Reviewing inactive users..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -d medical_kg_rev -c "
SELECT
    username,
    email,
    last_login,
    created_at
FROM users
WHERE last_login < NOW() - INTERVAL '90 days'
ORDER BY last_login DESC;
" >> "$LOG_FILE"

echo "$DATE - User access review completed" >> "$LOG_FILE"
```

## Update Procedures

### Application Updates

#### Rolling Update Procedure

```bash
#!/bin/bash
# Rolling update procedure

set -e

NEW_VERSION="$1"
LOG_FILE="/var/log/maintenance/rolling_update.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new_version>"
    exit 1
fi

echo "$DATE - Starting rolling update to version $NEW_VERSION" >> "$LOG_FILE"

# Backup current state
echo "Creating backup..." >> "$LOG_FILE"
kubectl get all -n medical-kg-rev -o yaml > "/backups/pre_update_$DATE.yaml"

# Update API gateway
echo "Updating API gateway..." >> "$LOG_FILE"
kubectl set image deployment/api-gateway -n medical-kg-rev \
    api-gateway=medical-kg-rev/api-gateway:$NEW_VERSION

# Wait for rollout
kubectl rollout status deployment/api-gateway -n medical-kg-rev --timeout=300s

# Update document service
echo "Updating document service..." >> "$LOG_FILE"
kubectl set image deployment/document-service -n medical-kg-rev \
    document-service=medical-kg-rev/document-service:$NEW_VERSION

# Wait for rollout
kubectl rollout status deployment/document-service -n medical-kg-rev --timeout=300s

# Update search service
echo "Updating search service..." >> "$LOG_FILE"
kubectl set image deployment/search-service -n medical-kg-rev \
    search-service=medical-kg-rev/search-service:$NEW_VERSION

# Wait for rollout
kubectl rollout status deployment/search-service -n medical-kg-rev --timeout=300s

# Update KG service
echo "Updating KG service..." >> "$LOG_FILE"
kubectl set image deployment/kg-service -n medical-kg-rev \
    kg-service=medical-kg-rev/kg-service:$NEW_VERSION

# Wait for rollout
kubectl rollout status deployment/kg-service -n medical-kg-rev --timeout=300s

# Verify update
echo "Verifying update..." >> "$LOG_FILE"
kubectl get pods -n medical-kg-rev -o wide >> "$LOG_FILE"

# Health check
echo "Performing health check..." >> "$LOG_FILE"
curl -f http://api-gateway:8000/health >> "$LOG_FILE"

echo "$DATE - Rolling update completed" >> "$LOG_FILE"
```

### Database Updates

#### Database Migration Procedure

```bash
#!/bin/bash
# Database migration procedure

set -e

MIGRATION_VERSION="$1"
LOG_FILE="/var/log/maintenance/database_migration.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

if [ -z "$MIGRATION_VERSION" ]; then
    echo "Usage: $0 <migration_version>"
    exit 1
fi

echo "$DATE - Starting database migration to version $MIGRATION_VERSION" >> "$LOG_FILE"

# Backup database
echo "Creating database backup..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- pg_dump \
    -h postgresql-primary -U postgres -d medical_kg_rev \
    > "/backups/pre_migration_$DATE.sql"

# Run migration
echo "Running database migration..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql \
    -h postgresql-primary -U postgres -d medical_kg_rev \
    -f "/migrations/$MIGRATION_VERSION.sql"

# Verify migration
echo "Verifying migration..." >> "$LOG_FILE"
kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql \
    -h postgresql-primary -U postgres -d medical_kg_rev \
    -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;" >> "$LOG_FILE"

# Update application
echo "Updating application..." >> "$LOG_FILE"
kubectl set image deployment/api-gateway -n medical-kg-rev \
    api-gateway=medical-kg-rev/api-gateway:$MIGRATION_VERSION

# Wait for rollout
kubectl rollout status deployment/api-gateway -n medical-kg-rev --timeout=300s

# Health check
echo "Performing health check..." >> "$LOG_FILE"
curl -f http://api-gateway:8000/health >> "$LOG_FILE"

echo "$DATE - Database migration completed" >> "$LOG_FILE"
```

## Emergency Procedures

### Incident Response

#### Emergency Response Script

```bash
#!/bin/bash
# Emergency response script

set -e

INCIDENT_TYPE="$1"
LOG_FILE="/var/log/maintenance/emergency_response.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "$DATE - Emergency response initiated for: $INCIDENT_TYPE" >> "$LOG_FILE"

case "$INCIDENT_TYPE" in
    "database_down")
        echo "Database down - initiating recovery..." >> "$LOG_FILE"

        # Check database status
        kubectl get pods -n medical-kg-rev -l app=postgresql >> "$LOG_FILE"

        # Restart database
        kubectl delete pod -n medical-kg-rev -l app=postgresql
        kubectl wait --for=condition=ready pod -l app=postgresql -n medical-kg-rev --timeout=300s

        # Verify recovery
        kubectl exec -n medical-kg-rev deployment/postgresql-client -- psql -h postgresql-primary -U postgres -c "SELECT 1;" >> "$LOG_FILE"
        ;;

    "api_down")
        echo "API down - initiating recovery..." >> "$LOG_FILE"

        # Check API status
        kubectl get pods -n medical-kg-rev -l app=api-gateway >> "$LOG_FILE"

        # Restart API
        kubectl delete pod -n medical-kg-rev -l app=api-gateway
        kubectl wait --for=condition=ready pod -l app=api-gateway -n medical-kg-rev --timeout=300s

        # Verify recovery
        curl -f http://api-gateway:8000/health >> "$LOG_FILE"
        ;;

    "high_memory")
        echo "High memory usage - initiating cleanup..." >> "$LOG_FILE"

        # Check memory usage
        kubectl top pods -n medical-kg-rev >> "$LOG_FILE"

        # Restart high memory pods
        kubectl get pods -n medical-kg-rev -o json | jq -r '.items[] | select(.status.containerStatuses[0].restartCount > 3) | .metadata.name' | while read pod; do
            kubectl delete pod "$pod" -n medical-kg-rev
        done

        # Clear Redis cache
        kubectl exec -n medical-kg-rev deployment/redis-client -- redis-cli -h redis-primary FLUSHDB
        ;;

    "security_breach")
        echo "Security breach detected - initiating lockdown..." >> "$LOG_FILE"

        # Block suspicious IPs
        kubectl get pods -n medical-kg-rev -o json | jq -r '.items[] | select(.spec.hostNetwork == true) | .metadata.name' | while read pod; do
            kubectl exec "$pod" -n medical-kg-rev -- iptables -A INPUT -s 0.0.0.0/0 -j DROP
        done

        # Disable external access
        kubectl patch ingress api-gateway-ingress -n medical-kg-rev -p '{"spec":{"rules":[]}}'

        # Alert security team
        echo "SECURITY BREACH DETECTED" | mail -s "Security Alert" security@medical-kg-rev.com
        ;;

    *)
        echo "Unknown incident type: $INCIDENT_TYPE" >> "$LOG_FILE"
        exit 1
        ;;
esac

echo "$DATE - Emergency response completed" >> "$LOG_FILE"
```

## Maintenance Checklists

### Daily Checklist

- [ ] System health check completed
- [ ] Database connections verified
- [ ] API endpoints responding
- [ ] Search functionality working
- [ ] Log rotation completed
- [ ] Backup status verified
- [ ] Security alerts reviewed
- [ ] Performance metrics checked

### Weekly Checklist

- [ ] Full system backup completed
- [ ] Performance analysis completed
- [ ] Security audit performed
- [ ] Database maintenance completed
- [ ] Cache optimization performed
- [ ] Log analysis completed
- [ ] Capacity planning reviewed
- [ ] Update procedures tested

### Monthly Checklist

- [ ] Security patches applied
- [ ] User access review completed
- [ ] Disaster recovery tested
- [ ] Performance optimization completed
- [ ] Documentation updated
- [ ] Compliance audit performed
- [ ] Capacity planning updated
- [ ] Maintenance procedures reviewed

### Quarterly Checklist

- [ ] Security assessment completed
- [ ] Disaster recovery plan tested
- [ ] Business continuity plan reviewed
- [ ] Performance benchmarks updated
- [ ] Maintenance procedures updated
- [ ] Training materials updated
- [ ] Compliance requirements reviewed
- [ ] System architecture reviewed

This comprehensive maintenance procedures document provides all the necessary procedures for maintaining the Medical_KG_rev system effectively. It covers routine maintenance, database maintenance, system monitoring, performance optimization, troubleshooting, backup and recovery, security maintenance, update procedures, emergency procedures, and maintenance checklists.
