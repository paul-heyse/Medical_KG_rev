# Rollback Procedures

This document provides comprehensive rollback procedures for the Medical_KG_rev system, including automated and manual rollback strategies, validation steps, and recovery procedures for different deployment scenarios.

## Overview

Rollback procedures are essential for maintaining system reliability and minimizing downtime when deployments fail or cause issues. This guide covers various rollback strategies, from simple application rollbacks to complex database and infrastructure rollbacks.

## Rollback Strategies

### Docling VLM Rollback (Failover to MinerU)

Use this runbook when the Docling Gemma3 pipeline experiences a sustained outage or quality regression.

1. **Validate Alert Context**
   - Confirm alerts from `config/monitoring/alerts.yml` for `DoclingVLMProcessingSlow`, `DoclingVLMSuccessRateDrop`, or `DoclingModelUnavailable` are firing.
   - Review Grafana dashboard `docling-vlm-performance` to verify the degradation is not due to downstream services.
2. **Freeze Ingestion**
   - Pause new PDF ingest jobs via the orchestration control plane (`/v1/orchestration/pause` API) to prevent backlog growth.
3. **Execute Automated Rollback**
   - Run `scripts/rollback_to_mineru.sh` (use `--dry-run` first). The script toggles the `PDF_PROCESSING_BACKEND` feature flag, reapplies archived MinerU manifests, and restarts the gateway plus orchestration workers.
   - Confirm script output shows successful rollout status for `gateway`, `pdf-orchestrator`, and `mineru-parser` deployments.
4. **Post-Rollback Validation**
   - Hit `/health/docling` and expect `503` while Docling is disabled; `/health/mineru` should return `200`.
   - Process two canary PDFs via the MinerU path and compare results with `tests/regression/test_mineru_vs_docling_comparison.py` to ensure quality parity.
5. **Communicate & Track**
   - Announce rollback in #doc-processing with PagerDuty incident link.
   - Create a follow-up ticket referencing `docs/security/docling_security_assessment.md` to document the incident and remediation timeline.

### Application Rollback

#### Kubernetes Rollback

##### Manual Rollback

```bash
#!/bin/bash
# scripts/rollback_application.sh

set -e

NAMESPACE="production"
DEPLOYMENT_NAME="medical-kg-rev-gateway"
PREVIOUS_VERSION="$1"

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "Usage: $0 <previous-version>"
    echo "Available versions:"
    kubectl get replicasets -n $NAMESPACE -l app.kubernetes.io/name=$DEPLOYMENT_NAME --sort-by=.metadata.creationTimestamp
    exit 1
fi

echo "Starting rollback for $DEPLOYMENT_NAME to version $PREVIOUS_VERSION"

# Get current version
CURRENT_VERSION=$(kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.status.currentPodHash}')

# Rollback to previous version
kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE --to-revision=$PREVIOUS_VERSION

# Wait for rollback to complete
echo "Waiting for rollback to complete..."
kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=600s

# Verify rollback
echo "Verifying rollback..."
kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=$DEPLOYMENT_NAME

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

echo "Rollback completed successfully!"
```

##### Automated Rollback

```yaml
# k8s/production/rollback-policy.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: medical-kg-rev-gateway
  namespace: production
spec:
  replicas: 3
  strategy:
    canary:
      steps:
      - setWeight: 20
      - pause: {duration: 10m}
      - analysis:
          templates:
          - templateName: success-rate
          args:
          - name: service-name
            value: medical-kg-rev-gateway
      - setWeight: 40
      - pause: {duration: 10m}
      - analysis:
          templates:
          - templateName: success-rate
          args:
          - name: service-name
            value: medical-kg-rev-gateway
      rollbackPolicy:
        failureThreshold: 3
        failureType: "ErrorRate"
        errorRateThreshold: 0.1
  selector:
    matchLabels:
      app: medical-kg-rev
      component: gateway
  template:
    metadata:
      labels:
        app: medical-kg-rev
        component: gateway
    spec:
      containers:
      - name: gateway
        image: medical-kg-rev/gateway:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: medical-kg-rev-config
        - secretRef:
            name: medical-kg-rev-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Blue-Green Rollback

##### Switch Back to Blue

```bash
#!/bin/bash
# scripts/blue_green_rollback.sh

set -e

NAMESPACE="production"
SERVICE_NAME="medical-kg-rev-gateway"

echo "Starting blue-green rollback for $SERVICE_NAME"

# Get current active service
CURRENT_ACTIVE=$(kubectl get service $SERVICE_NAME-active -n $NAMESPACE -o jsonpath='{.spec.selector.version}')

if [ "$CURRENT_ACTIVE" = "green" ]; then
    echo "Currently on green, switching back to blue..."
    kubectl patch service $SERVICE_NAME-active -n $NAMESPACE -p '{"spec":{"selector":{"version":"blue"}}}'
    NEW_ACTIVE="blue"
elif [ "$CURRENT_ACTIVE" = "blue" ]; then
    echo "Currently on blue, switching to green..."
    kubectl patch service $SERVICE_NAME-active -n $NAMESPACE -p '{"spec":{"selector":{"version":"green"}}}'
    NEW_ACTIVE="green"
else
    echo "Unknown active version: $CURRENT_ACTIVE"
    exit 1
fi

# Wait for traffic to switch
echo "Waiting for traffic to switch to $NEW_ACTIVE..."
sleep 30

# Verify traffic is flowing to new active
kubectl get pods -l version=$NEW_ACTIVE -n $NAMESPACE

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

# Clean up old environment
echo "Cleaning up old environment..."
kubectl delete -f k8s/production/$CURRENT_ACTIVE/

echo "Blue-green rollback completed successfully!"
```

#### Canary Rollback

##### Rollback Canary Deployment

```bash
#!/bin/bash
# scripts/canary_rollback.sh

set -e

NAMESPACE="production"
SERVICE_NAME="medical-kg-rev-gateway"

echo "Starting canary rollback for $SERVICE_NAME"

# Route all traffic back to stable
kubectl patch service $SERVICE_NAME -n $NAMESPACE -p '{"spec":{"traffic":{"canary":{"weight":0}}}}'

# Wait for traffic to switch
echo "Waiting for traffic to switch back to stable..."
sleep 30

# Verify traffic is flowing to stable
kubectl get pods -l version=stable -n $NAMESPACE

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

# Remove canary deployment
echo "Removing canary deployment..."
kubectl delete deployment $SERVICE_NAME-canary -n $NAMESPACE

# Remove canary service
kubectl delete service $SERVICE_NAME-canary -n $NAMESPACE

echo "Canary rollback completed successfully!"
```

### Database Rollback

#### PostgreSQL Rollback

##### Schema Rollback

```bash
#!/bin/bash
# scripts/postgresql_rollback.sh

set -e

DATABASE_URL="$1"
TARGET_REVISION="$2"

if [ -z "$DATABASE_URL" ] || [ -z "$TARGET_REVISION" ]; then
    echo "Usage: $0 <database_url> <target_revision>"
    echo "Example: $0 postgresql://user:pass@host:5432/db 001"
    exit 1
fi

echo "Starting PostgreSQL rollback to revision $TARGET_REVISION"

# Backup current database
echo "Creating backup before rollback..."
pg_dump "$DATABASE_URL" > backup_before_rollback_$(date +%Y%m%d_%H%M%S).sql

# Run Alembic rollback
echo "Running Alembic rollback..."
alembic downgrade "$TARGET_REVISION"

# Verify rollback
echo "Verifying rollback..."
alembic current

echo "PostgreSQL rollback completed successfully!"
```

##### Data Rollback

```bash
#!/bin/bash
# scripts/postgresql_data_rollback.sh

set -e

DATABASE_URL="$1"
BACKUP_FILE="$2"

if [ -z "$DATABASE_URL" ] || [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <database_url> <backup_file>"
    echo "Example: $0 postgresql://user:pass@host:5432/db backup_20240101_120000.sql"
    exit 1
fi

echo "Starting PostgreSQL data rollback from $BACKUP_FILE"

# Stop application services
echo "Stopping application services..."
kubectl scale deployment medical-kg-rev-gateway --replicas=0 -n production
kubectl scale deployment medical-kg-rev-services --replicas=0 -n production

# Wait for services to stop
kubectl wait --for=delete pod -l app=medical-kg-rev-gateway -n production --timeout=300s

# Restore database from backup
echo "Restoring database from backup..."
psql "$DATABASE_URL" < "$BACKUP_FILE"

# Start application services
echo "Starting application services..."
kubectl scale deployment medical-kg-rev-gateway --replicas=3 -n production
kubectl scale deployment medical-kg-rev-services --replicas=2 -n production

# Wait for services to start
kubectl rollout status deployment/medical-kg-rev-gateway -n production --timeout=600s

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

echo "PostgreSQL data rollback completed successfully!"
```

#### Neo4j Rollback

##### Graph Database Rollback

```bash
#!/bin/bash
# scripts/neo4j_rollback.sh

set -e

NEO4J_URI="$1"
NEO4J_PASSWORD="$2"
BACKUP_FILE="$3"

if [ -z "$NEO4J_URI" ] || [ -z "$NEO4J_PASSWORD" ] || [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <neo4j_uri> <neo4j_password> <backup_file>"
    echo "Example: $0 bolt://localhost:7687 password backup_20240101_120000.dump"
    exit 1
fi

echo "Starting Neo4j rollback from $BACKUP_FILE"

# Stop application services
echo "Stopping application services..."
kubectl scale deployment medical-kg-rev-gateway --replicas=0 -n production
kubectl scale deployment medical-kg-rev-services --replicas=0 -n production

# Wait for services to stop
kubectl wait --for=delete pod -l app=medical-kg-rev-gateway -n production --timeout=300s

# Stop Neo4j
echo "Stopping Neo4j..."
kubectl scale deployment neo4j-production --replicas=0 -n production

# Wait for Neo4j to stop
kubectl wait --for=delete pod -l app=neo4j-production -n production --timeout=300s

# Restore Neo4j from backup
echo "Restoring Neo4j from backup..."
kubectl run neo4j-restore --image=neo4j:5 --rm -i --restart=Never \
  --env="NEO4J_AUTH=neo4j/$NEO4J_PASSWORD" \
  --volume="type=bind,source=$BACKUP_FILE,target=/backup.dump" \
  --command -- neo4j-admin load --from=/backup.dump --database=neo4j --force

# Start Neo4j
echo "Starting Neo4j..."
kubectl scale deployment neo4j-production --replicas=1 -n production

# Wait for Neo4j to start
kubectl rollout status deployment/neo4j-production -n production --timeout=600s

# Start application services
echo "Starting application services..."
kubectl scale deployment medical-kg-rev-gateway --replicas=3 -n production
kubectl scale deployment medical-kg-rev-services --replicas=2 -n production

# Wait for services to start
kubectl rollout status deployment/medical-kg-rev-gateway -n production --timeout=600s

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

echo "Neo4j rollback completed successfully!"
```

#### Redis Rollback

##### Cache Rollback

```bash
#!/bin/bash
# scripts/redis_rollback.sh

set -e

REDIS_URL="$1"
BACKUP_FILE="$2"

if [ -z "$REDIS_URL" ] || [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <redis_url> <backup_file>"
    echo "Example: $0 redis://localhost:6379/0 backup_20240101_120000.rdb"
    exit 1
fi

echo "Starting Redis rollback from $BACKUP_FILE"

# Stop application services
echo "Stopping application services..."
kubectl scale deployment medical-kg-rev-gateway --replicas=0 -n production
kubectl scale deployment medical-kg-rev-services --replicas=0 -n production

# Wait for services to stop
kubectl wait --for=delete pod -l app=medical-kg-rev-gateway -n production --timeout=300s

# Stop Redis
echo "Stopping Redis..."
kubectl scale deployment redis-production --replicas=0 -n production

# Wait for Redis to stop
kubectl wait --for=delete pod -l app=redis-production -n production --timeout=300s

# Restore Redis from backup
echo "Restoring Redis from backup..."
kubectl run redis-restore --image=redis:7 --rm -i --restart=Never \
  --volume="type=bind,source=$BACKUP_FILE,target=/backup.rdb" \
  --command -- cp /backup.rdb /data/dump.rdb

# Start Redis
echo "Starting Redis..."
kubectl scale deployment redis-production --replicas=1 -n production

# Wait for Redis to start
kubectl rollout status deployment/redis-production -n production --timeout=600s

# Start application services
echo "Starting application services..."
kubectl scale deployment medical-kg-rev-gateway --replicas=3 -n production
kubectl scale deployment medical-kg-rev-services --replicas=2 -n production

# Wait for services to start
kubectl rollout status deployment/medical-kg-rev-gateway -n production --timeout=600s

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

echo "Redis rollback completed successfully!"
```

### Infrastructure Rollback

#### Kubernetes Cluster Rollback

##### Cluster Configuration Rollback

```bash
#!/bin/bash
# scripts/cluster_rollback.sh

set -e

NAMESPACE="production"
BACKUP_DIR="$1"

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup_directory>"
    echo "Example: $0 /backups/k8s/20240101_120000"
    exit 1
fi

echo "Starting Kubernetes cluster rollback from $BACKUP_DIR"

# Verify backup directory exists
if [ ! -d "$BACKUP_DIR" ]; then
    echo "Backup directory $BACKUP_DIR does not exist"
    exit 1
fi

# Stop all application services
echo "Stopping all application services..."
kubectl scale deployment --all --replicas=0 -n $NAMESPACE

# Wait for all pods to terminate
echo "Waiting for all pods to terminate..."
kubectl wait --for=delete pod --all -n $NAMESPACE --timeout=300s

# Restore configurations
echo "Restoring configurations..."
kubectl apply -f "$BACKUP_DIR/configmaps.yaml"
kubectl apply -f "$BACKUP_DIR/secrets.yaml"
kubectl apply -f "$BACKUP_DIR/services.yaml"
kubectl apply -f "$BACKUP_DIR/deployments.yaml"
kubectl apply -f "$BACKUP_DIR/ingress.yaml"

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment/medical-kg-rev-gateway -n $NAMESPACE --timeout=600s
kubectl rollout status deployment/medical-kg-rev-services -n $NAMESPACE --timeout=600s

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

echo "Kubernetes cluster rollback completed successfully!"
```

#### Network Rollback

##### Load Balancer Rollback

```bash
#!/bin/bash
# scripts/load_balancer_rollback.sh

set -e

NAMESPACE="production"
SERVICE_NAME="medical-kg-rev-gateway"

echo "Starting load balancer rollback for $SERVICE_NAME"

# Get current ingress configuration
echo "Current ingress configuration:"
kubectl get ingress $SERVICE_NAME-ingress -n $NAMESPACE -o yaml

# Rollback to previous ingress configuration
echo "Rolling back ingress configuration..."
kubectl apply -f k8s/production/ingress-previous.yaml

# Wait for ingress to update
echo "Waiting for ingress to update..."
sleep 30

# Verify ingress configuration
echo "Updated ingress configuration:"
kubectl get ingress $SERVICE_NAME-ingress -n $NAMESPACE -o yaml

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

echo "Load balancer rollback completed successfully!"
```

## Emergency Rollback Procedures

### Critical System Failure

#### Emergency Application Rollback

```bash
#!/bin/bash
# scripts/emergency_rollback.sh

set -e

NAMESPACE="production"
SERVICE_NAME="medical-kg-rev-gateway"

echo "🚨 Starting emergency rollback for $SERVICE_NAME"

# Get the last successful deployment
LAST_SUCCESSFUL=$(kubectl get deployment $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Progressing")].lastUpdateTime}' | head -1)

if [ -z "$LAST_SUCCESSFUL" ]; then
    echo "No successful deployment found. Rolling back to previous version."
    kubectl rollout undo deployment/$SERVICE_NAME -n $NAMESPACE
else
    echo "Rolling back to last successful deployment: $LAST_SUCCESSFUL"
    kubectl rollout undo deployment/$SERVICE_NAME -n $NAMESPACE --to-revision=$LAST_SUCCESSFUL
fi

# Wait for rollback to complete
echo "Waiting for rollback to complete..."
kubectl rollout status deployment/$SERVICE_NAME -n $NAMESPACE --timeout=300s

# Verify rollback
echo "Verifying rollback..."
kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=$SERVICE_NAME

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

# Notify team
echo "Sending notification..."
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"🚨 Emergency rollback completed for medical-kg-rev-gateway"}' \
  $SLACK_WEBHOOK_URL

echo "Emergency rollback completed!"
```

#### Emergency Database Rollback

```bash
#!/bin/bash
# scripts/emergency_database_rollback.sh

set -e

DATABASE_URL="$1"
BACKUP_FILE="$2"

if [ -z "$DATABASE_URL" ] || [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <database_url> <backup_file>"
    echo "Example: $0 postgresql://user:pass@host:5432/db backup_20240101_120000.sql"
    exit 1
fi

echo "🚨 Starting emergency database rollback from $BACKUP_FILE"

# Stop all application services immediately
echo "Stopping all application services..."
kubectl scale deployment --all --replicas=0 -n production

# Wait for services to stop
kubectl wait --for=delete pod --all -n production --timeout=300s

# Restore database from backup
echo "Restoring database from backup..."
psql "$DATABASE_URL" < "$BACKUP_FILE"

# Start application services
echo "Starting application services..."
kubectl scale deployment medical-kg-rev-gateway --replicas=3 -n production
kubectl scale deployment medical-kg-rev-services --replicas=2 -n production

# Wait for services to start
kubectl rollout status deployment/medical-kg-rev-gateway -n production --timeout=600s

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

# Notify team
echo "Sending notification..."
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"🚨 Emergency database rollback completed"}' \
  $SLACK_WEBHOOK_URL

echo "Emergency database rollback completed!"
```

### Data Corruption Recovery

#### Data Integrity Rollback

```bash
#!/bin/bash
# scripts/data_integrity_rollback.sh

set -e

DATABASE_URL="$1"
BACKUP_FILE="$2"

if [ -z "$DATABASE_URL" ] || [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <database_url> <backup_file>"
    echo "Example: $0 postgresql://user:pass@host:5432/db backup_20240101_120000.sql"
    exit 1
fi

echo "🚨 Starting data integrity rollback from $BACKUP_FILE"

# Verify backup file integrity
echo "Verifying backup file integrity..."
pg_restore --list "$BACKUP_FILE" > /dev/null
if [ $? -ne 0 ]; then
    echo "Backup file is corrupted or invalid"
    exit 1
fi

# Stop all application services
echo "Stopping all application services..."
kubectl scale deployment --all --replicas=0 -n production

# Wait for services to stop
kubectl wait --for=delete pod --all -n production --timeout=300s

# Create current state backup
echo "Creating current state backup..."
pg_dump "$DATABASE_URL" > current_state_backup_$(date +%Y%m%d_%H%M%S).sql

# Restore database from backup
echo "Restoring database from backup..."
psql "$DATABASE_URL" < "$BACKUP_FILE"

# Verify data integrity
echo "Verifying data integrity..."
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM users;"
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM documents;"
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM chunks;"

# Start application services
echo "Starting application services..."
kubectl scale deployment medical-kg-rev-gateway --replicas=3 -n production
kubectl scale deployment medical-kg-rev-services --replicas=2 -n production

# Wait for services to start
kubectl rollout status deployment/medical-kg-rev-gateway -n production --timeout=600s

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

# Notify team
echo "Sending notification..."
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"🚨 Data integrity rollback completed"}' \
  $SLACK_WEBHOOK_URL

echo "Data integrity rollback completed!"
```

## Rollback Validation

### Health Check Validation

#### Post-Rollback Health Checks

```python
# scripts/post_rollback_validation.py
import requests
import time
import sys
from typing import Dict, List, Tuple

class RollbackValidator:
    """Validate rollback success."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = 30

    def check_health_endpoint(self) -> Tuple[bool, str]:
        """Check health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            if response.status_code == 200:
                return True, "Health endpoint OK"
            else:
                return False, f"Health endpoint returned {response.status_code}"
        except Exception as e:
            return False, f"Health endpoint failed: {e}"

    def check_ready_endpoint(self) -> Tuple[bool, str]:
        """Check ready endpoint."""
        try:
            response = requests.get(f"{self.base_url}/ready", timeout=self.timeout)
            if response.status_code == 200:
                return True, "Ready endpoint OK"
            else:
                return False, f"Ready endpoint returned {response.status_code}"
        except Exception as e:
            return False, f"Ready endpoint failed: {e}"

    def check_api_endpoints(self) -> List[Tuple[bool, str]]:
        """Check API endpoints."""
        endpoints = [
            "/api/v1/documents",
            "/api/v1/users",
            "/api/v1/health"
        ]

        results = []
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
                if response.status_code in [200, 401, 403]:  # 401/403 are OK for protected endpoints
                    results.append((True, f"{endpoint} OK"))
                else:
                    results.append((False, f"{endpoint} returned {response.status_code}"))
            except Exception as e:
                results.append((False, f"{endpoint} failed: {e}"))

        return results

    def check_database_connectivity(self) -> Tuple[bool, str]:
        """Check database connectivity."""
        try:
            response = requests.get(f"{self.base_url}/health/database", timeout=self.timeout)
            if response.status_code == 200:
                return True, "Database connectivity OK"
            else:
                return False, f"Database connectivity failed: {response.status_code}"
        except Exception as e:
            return False, f"Database connectivity failed: {e}"

    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("Running post-rollback validation...")

        # Health endpoint
        health_ok, health_msg = self.check_health_endpoint()
        print(f"Health endpoint: {'✅' if health_ok else '❌'} {health_msg}")

        # Ready endpoint
        ready_ok, ready_msg = self.check_ready_endpoint()
        print(f"Ready endpoint: {'✅' if ready_ok else '❌'} {ready_msg}")

        # Database connectivity
        db_ok, db_msg = self.check_database_connectivity()
        print(f"Database connectivity: {'✅' if db_ok else '❌'} {db_msg}")

        # API endpoints
        api_results = self.check_api_endpoints()
        for ok, msg in api_results:
            print(f"API endpoint: {'✅' if ok else '❌'} {msg}")

        # Overall result
        all_checks = [health_ok, ready_ok, db_ok] + [ok for ok, _ in api_results]
        overall_success = all(all_checks)

        print(f"\nOverall result: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
        return overall_success

def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Post-rollback validation")
    parser.add_argument("--environment", required=True, choices=["staging", "production"])
    parser.add_argument("--base-url", help="Base URL for validation")

    args = parser.parse_args()

    if args.base_url:
        base_url = args.base_url
    else:
        base_urls = {
            "staging": "https://staging.medical-kg-rev.com",
            "production": "https://api.medical-kg-rev.com"
        }
        base_url = base_urls[args.environment]

    validator = RollbackValidator(base_url)
    success = validator.run_validation()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

### Performance Validation

#### Post-Rollback Performance Checks

```python
# scripts/post_rollback_performance.py
import requests
import time
import statistics
from typing import List, Dict, Any

class RollbackPerformanceValidator:
    """Validate rollback performance."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.timeout = 30

    def measure_response_time(self, endpoint: str, count: int = 10) -> List[float]:
        """Measure response times for an endpoint."""
        response_times = []

        for _ in range(count):
            start_time = time.time()
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
                end_time = time.time()
                response_times.append(end_time - start_time)
            except Exception:
                response_times.append(float('inf'))

        return response_times

    def check_performance_thresholds(self, response_times: List[float]) -> Dict[str, Any]:
        """Check performance against thresholds."""
        if not response_times:
            return {"status": "failed", "reason": "No successful requests"}

        # Calculate statistics
        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        max_time = max(response_times)

        # Thresholds
        avg_threshold = 0.5  # 500ms
        p95_threshold = 1.0  # 1000ms
        max_threshold = 2.0  # 2000ms

        # Check thresholds
        avg_ok = avg_time <= avg_threshold
        p95_ok = p95_time <= p95_threshold
        max_ok = max_time <= max_threshold

        return {
            "status": "passed" if all([avg_ok, p95_ok, max_ok]) else "failed",
            "avg_time": avg_time,
            "p95_time": p95_time,
            "max_time": max_time,
            "avg_threshold": avg_threshold,
            "p95_threshold": p95_threshold,
            "max_threshold": max_threshold,
            "avg_ok": avg_ok,
            "p95_ok": p95_ok,
            "max_ok": max_ok
        }

    def run_performance_validation(self) -> bool:
        """Run performance validation."""
        print("Running post-rollback performance validation...")

        endpoints = [
            "/health",
            "/api/v1/documents",
            "/api/v1/users"
        ]

        all_passed = True

        for endpoint in endpoints:
            print(f"\nChecking {endpoint}...")
            response_times = self.measure_response_time(endpoint)
            results = self.check_performance_thresholds(response_times)

            status_emoji = "✅" if results["status"] == "passed" else "❌"
            print(f"{status_emoji} Status: {results['status']}")
            print(f"   Average: {results['avg_time']:.3f}s (threshold: {results['avg_threshold']}s)")
            print(f"   P95: {results['p95_time']:.3f}s (threshold: {results['p95_threshold']}s)")
            print(f"   Max: {results['max_time']:.3f}s (threshold: {results['max_threshold']}s)")

            if results["status"] != "passed":
                all_passed = False

        print(f"\nOverall result: {'✅ SUCCESS' if all_passed else '❌ FAILURE'}")
        return all_passed

def main():
    """Main performance validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Post-rollback performance validation")
    parser.add_argument("--environment", required=True, choices=["staging", "production"])
    parser.add_argument("--base-url", help="Base URL for performance validation")

    args = parser.parse_args()

    if args.base_url:
        base_url = args.base_url
    else:
        base_urls = {
            "staging": "https://staging.medical-kg-rev.com",
            "production": "https://api.medical-kg-rev.com"
        }
        base_url = base_urls[args.environment]

    validator = RollbackPerformanceValidator(base_url)
    success = validator.run_performance_validation()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

## Rollback Automation

### Automated Rollback Triggers

#### GitHub Actions Rollback Workflow

```yaml
# .github/workflows/rollback.yml
name: Rollback

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to rollback'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
      rollback_type:
        description: 'Type of rollback'
        required: true
        default: 'application'
        type: choice
        options:
          - application
          - database
          - infrastructure
      target_revision:
        description: 'Target revision for rollback'
        required: false
        type: string

jobs:
  rollback:
    runs-on: ubuntu-latest
    environment: ${{ github.event.inputs.environment }}

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'v1.28.0'

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Execute rollback
        run: |
          export KUBECONFIG=kubeconfig
          case "${{ github.event.inputs.rollback_type }}" in
            application)
              if [ -n "${{ github.event.inputs.target_revision }}" ]; then
                ./scripts/rollback_application.sh ${{ github.event.inputs.target_revision }}
              else
                ./scripts/emergency_rollback.sh
              fi
              ;;
            database)
              ./scripts/emergency_database_rollback.sh $DATABASE_URL $BACKUP_FILE
              ;;
            infrastructure)
              ./scripts/cluster_rollback.sh $BACKUP_DIR
              ;;
          esac

      - name: Validate rollback
        run: |
          export KUBECONFIG=kubeconfig
          python scripts/post_rollback_validation.py --environment=${{ github.event.inputs.environment }}

      - name: Notify rollback
        uses: 8398a7/action-slack@v3
        if: always()
        with:
          status: ${{ job.status }}
          channel: '#deployments'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Monitoring-Based Rollback

#### Prometheus Alert-Based Rollback

```yaml
# prometheus/rollback-alerts.yml
groups:
- name: rollback-triggers
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
      rollback_trigger: "true"
    annotations:
      summary: "High error rate detected - triggering rollback"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: critical
      rollback_trigger: "true"
    annotations:
      summary: "High response time detected - triggering rollback"
      description: "95th percentile response time is {{ $value }} seconds"

  - alert: ServiceDown
    expr: up{job="medical-kg-rev"} == 0
    for: 1m
    labels:
      severity: critical
      rollback_trigger: "true"
    annotations:
      summary: "Service is down - triggering rollback"
      description: "Service {{ $labels.instance }} is down"
```

## Rollback Best Practices

### Preparation

1. **Backup Strategy**: Maintain comprehensive backups before deployments
2. **Rollback Testing**: Test rollback procedures regularly
3. **Documentation**: Keep rollback procedures documented and updated
4. **Team Training**: Train team members on rollback procedures
5. **Automation**: Automate rollback procedures where possible

### Execution

1. **Quick Response**: Respond quickly to deployment issues
2. **Communication**: Communicate rollback status to stakeholders
3. **Validation**: Validate rollback success thoroughly
4. **Monitoring**: Monitor system health during rollback
5. **Documentation**: Document rollback events and lessons learned

### Post-Rollback

1. **Root Cause Analysis**: Analyze the cause of deployment failure
2. **Process Improvement**: Improve deployment processes based on learnings
3. **Team Review**: Conduct team review of rollback procedures
4. **Documentation Update**: Update procedures based on experience
5. **Prevention**: Implement measures to prevent similar issues

## Troubleshooting

### Common Rollback Issues

#### 1. Rollback Failures

```bash
# Check deployment status
kubectl get deployments -n production

# Check rollout history
kubectl rollout history deployment/medical-kg-rev-gateway -n production

# Check pod status
kubectl get pods -n production

# Check events
kubectl get events -n production --sort-by='.lastTimestamp'
```

#### 2. Database Rollback Issues

```bash
# Check database status
kubectl get pods -l app=postgres-production -n production

# Check database logs
kubectl logs -f deployment/postgres-production -n production

# Check database connectivity
kubectl run test-pod --image=postgres:14 --rm -it --restart=Never -- psql -h postgres-production -U prod -d medical_kg_production
```

#### 3. Service Rollback Issues

```bash
# Check service status
kubectl get services -n production

# Check service endpoints
kubectl get endpoints -n production

# Check ingress status
kubectl get ingress -n production

# Test service connectivity
kubectl run test-pod --image=busybox --rm -it --restart=Never -- nslookup medical-kg-rev-gateway
```

### Debug Commands

```bash
# Check rollback status
kubectl rollout status deployment/medical-kg-rev-gateway -n production

# Check rollback history
kubectl rollout history deployment/medical-kg-rev-gateway -n production

# Check pod status
kubectl get pods -n production
kubectl describe pod <pod-name> -n production

# Check service status
kubectl get services -n production
kubectl describe service medical-kg-rev-gateway -n production

# Check ingress status
kubectl get ingress -n production
kubectl describe ingress medical-kg-rev-ingress -n production

# Check logs
kubectl logs -f deployment/medical-kg-rev-gateway -n production
kubectl logs -f deployment/medical-kg-rev-services -n production

# Check events
kubectl get events -n production --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n production
kubectl top nodes

# Check persistent volumes
kubectl get pv
kubectl get pvc -n production

# Check secrets
kubectl get secrets -n production
kubectl describe secret medical-kg-rev-secrets -n production

# Check configmaps
kubectl get configmaps -n production
kubectl describe configmap medical-kg-rev-config -n production
```

## Related Documentation

- [Deployment Overview](deployment_overview.md)
- [Infrastructure Requirements](infrastructure_requirements.md)
- [Deployment Procedures](deployment_procedures.md)
- [Monitoring and Logging](monitoring_logging.md)
- [Security Considerations](security_considerations.md)
- [Disaster Recovery Plan](disaster_recovery_plan.md)
