# Deployment Procedures

This document provides step-by-step deployment instructions for the Medical_KG_rev system, including configuration settings, commands, and validation procedures for different deployment environments.

## Overview

The Medical_KG_rev deployment procedures cover manual and automated deployment processes across development, staging, and production environments. This guide ensures consistent, reliable, and secure deployments.

## Prerequisites

### System Requirements

#### Development Environment

```bash
# Check system requirements
python --version  # Python 3.11+
node --version    # Node.js 18+
docker --version  # Docker 24.0+
kubectl version   # kubectl 1.28+
helm version      # Helm 3.12+

# Check available resources
free -h           # RAM: 8GB+
df -h             # Storage: 100GB+
lscpu             # CPU: 4 cores+
```

#### Staging/Production Environment

```bash
# Check Kubernetes cluster
kubectl cluster-info
kubectl get nodes
kubectl get namespaces

# Check available resources
kubectl top nodes
kubectl describe nodes

# Check storage classes
kubectl get storageclass
kubectl get pv
kubectl get pvc
```

### Required Tools

#### Installation Script

```bash
#!/bin/bash
# scripts/install_prerequisites.sh

set -e

echo "Installing prerequisites for Medical_KG_rev deployment..."

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install additional tools
sudo apt install -y git curl wget vim htop jq

echo "✅ Prerequisites installed successfully!"
echo "Please log out and log back in to apply Docker group changes."
```

## Environment Configuration

### Environment Variables

#### Development Environment

```bash
# .env.development
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8000
GATEWAY_LOG_LEVEL=DEBUG

DATABASE_URL=postgresql://dev:dev@localhost:5432/medical_kg_dev
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=dev
REDIS_URL=redis://localhost:6379/0
VECTOR_STORE_URL=http://localhost:6333

OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PUBMED_API_KEY=your_pubmed_api_key

JWT_SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key
```

#### Staging Environment

```bash
# .env.staging
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8000
GATEWAY_LOG_LEVEL=INFO

DATABASE_URL=postgresql://staging:staging@postgres-staging:5432/medical_kg_staging
NEO4J_URI=bolt://neo4j-staging:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=staging_password
REDIS_URL=redis://redis-staging:6379/0
VECTOR_STORE_URL=http://qdrant-staging:6333

OPENAI_API_KEY=your_staging_openai_api_key
ANTHROPIC_API_KEY=your_staging_anthropic_api_key
PUBMED_API_KEY=your_staging_pubmed_api_key

JWT_SECRET_KEY=your_staging_jwt_secret_key
ENCRYPTION_KEY=your_staging_encryption_key
```

#### Production Environment

```bash
# .env.production
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8000
GATEWAY_LOG_LEVEL=WARNING

DATABASE_URL=postgresql://prod:prod@postgres-production:5432/medical_kg_production
NEO4J_URI=bolt://neo4j-production:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=production_password
REDIS_URL=redis://redis-production:6379/0
VECTOR_STORE_URL=http://qdrant-production:6333

OPENAI_API_KEY=your_production_openai_api_key
ANTHROPIC_API_KEY=your_production_anthropic_api_key
PUBMED_API_KEY=your_production_pubmed_api_key

JWT_SECRET_KEY=your_production_jwt_secret_key
ENCRYPTION_KEY=your_production_encryption_key
```

### Kubernetes Configuration

#### Namespace Configuration

```yaml
# k8s/development/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: development
  labels:
    environment: development
    app: medical-kg-rev
---
# k8s/staging/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: staging
  labels:
    environment: staging
    app: medical-kg-rev
---
# k8s/production/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    environment: production
    app: medical-kg-rev
```

#### ConfigMap Configuration

```yaml
# k8s/development/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-kg-rev-config
  namespace: development
data:
  GATEWAY_HOST: "0.0.0.0"
  GATEWAY_PORT: "8000"
  GATEWAY_LOG_LEVEL: "DEBUG"
  DATABASE_URL: "postgresql://dev:dev@postgres-development:5432/medical_kg_dev"
  NEO4J_URI: "bolt://neo4j-development:7687"
  NEO4J_PASSWORD: "dev"
  REDIS_URL: "redis://redis-development:6379/0"
  VECTOR_STORE_URL: "http://qdrant-development:6333"
---
# k8s/staging/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-kg-rev-config
  namespace: staging
data:
  GATEWAY_HOST: "0.0.0.0"
  GATEWAY_PORT: "8000"
  GATEWAY_LOG_LEVEL: "INFO"
  DATABASE_URL: "postgresql://staging:staging@postgres-staging:5432/medical_kg_staging"
  NEO4J_URI: "bolt://neo4j-staging:7687"
  NEO4J_PASSWORD: "staging_password"
  REDIS_URL: "redis://redis-staging:6379/0"
  VECTOR_STORE_URL: "http://qdrant-staging:6333"
---
# k8s/production/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-kg-rev-config
  namespace: production
data:
  GATEWAY_HOST: "0.0.0.0"
  GATEWAY_PORT: "8000"
  GATEWAY_LOG_LEVEL: "WARNING"
  DATABASE_URL: "postgresql://prod:prod@postgres-production:5432/medical_kg_production"
  NEO4J_URI: "bolt://neo4j-production:7687"
  NEO4J_PASSWORD: "production_password"
  REDIS_URL: "redis://redis-production:6379/0"
  VECTOR_STORE_URL: "http://qdrant-production:6333"
```

#### Secret Configuration

```yaml
# k8s/development/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: medical-kg-rev-secrets
  namespace: development
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
  ANTHROPIC_API_KEY: <base64-encoded-key>
  PUBMED_API_KEY: <base64-encoded-key>
  JWT_SECRET_KEY: <base64-encoded-key>
  ENCRYPTION_KEY: <base64-encoded-key>
---
# k8s/staging/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: medical-kg-rev-secrets
  namespace: staging
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-staging-key>
  ANTHROPIC_API_KEY: <base64-encoded-staging-key>
  PUBMED_API_KEY: <base64-encoded-staging-key>
  JWT_SECRET_KEY: <base64-encoded-staging-key>
  ENCRYPTION_KEY: <base64-encoded-staging-key>
---
# k8s/production/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: medical-kg-rev-secrets
  namespace: production
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-production-key>
  ANTHROPIC_API_KEY: <base64-encoded-production-key>
  PUBMED_API_KEY: <base64-encoded-production-key>
  JWT_SECRET_KEY: <base64-encoded-production-key>
  ENCRYPTION_KEY: <base64-encoded-production-key>
```

## Development Environment Deployment

### Local Development Setup

#### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/Medical_KG_rev.git
cd Medical_KG_rev

# Checkout the development branch
git checkout develop
git pull origin develop
```

#### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env.development

# Edit environment variables
vim .env.development

# Set environment variables
export $(cat .env.development | xargs)
```

#### Step 4: Start Services

```bash
# Start database services
docker-compose up -d postgres neo4j redis qdrant

# Wait for services to be ready
python scripts/wait_for_services.py

# Run database migrations
python -m Medical_KG_rev.storage.migrations migrate

# Start the application
python -m Medical_KG_rev.gateway.main
```

#### Step 5: Verify Deployment

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check ready endpoint
curl http://localhost:8000/ready

# Run tests
pytest tests/unit/ -v
```

### Docker Development Setup

#### Step 1: Build Development Images

```bash
# Build all development images
docker-compose -f docker-compose.dev.yml build

# Or build specific service
docker-compose -f docker-compose.dev.yml build gateway
```

#### Step 2: Start Development Environment

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Check service status
docker-compose -f docker-compose.dev.yml ps
```

#### Step 3: Verify Deployment

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check service logs
docker-compose -f docker-compose.dev.yml logs gateway

# Run tests
docker-compose -f docker-compose.dev.yml exec gateway pytest tests/unit/ -v
```

## Staging Environment Deployment

### Manual Deployment

#### Step 1: Prepare Staging Environment

```bash
# Set kubectl context
kubectl config use-context staging

# Create namespace
kubectl apply -f k8s/staging/namespace.yaml

# Create secrets
kubectl apply -f k8s/staging/secrets.yaml

# Create configmap
kubectl apply -f k8s/staging/configmap.yaml
```

#### Step 2: Deploy Database Services

```bash
# Deploy PostgreSQL
kubectl apply -f k8s/staging/postgresql.yaml

# Deploy Neo4j
kubectl apply -f k8s/staging/neo4j.yaml

# Deploy Redis
kubectl apply -f k8s/staging/redis.yaml

# Deploy Qdrant
kubectl apply -f k8s/staging/qdrant.yaml

# Wait for services to be ready
kubectl wait --for=condition=available deployment/postgres-staging -n staging --timeout=300s
kubectl wait --for=condition=available deployment/neo4j-staging -n staging --timeout=300s
kubectl wait --for=condition=available deployment/redis-staging -n staging --timeout=300s
kubectl wait --for=condition=available deployment/qdrant-staging -n staging --timeout=300s
```

#### Step 3: Run Database Migrations

```bash
# Run PostgreSQL migrations
kubectl run migration-job --image=medical-kg-rev/migration:latest \
  --env="DATABASE_URL=postgresql://staging:staging@postgres-staging:5432/medical_kg_staging" \
  --restart=Never \
  --rm -i --tty

# Run Neo4j migrations
kubectl run neo4j-migration-job --image=medical-kg-rev/neo4j-migration:latest \
  --env="NEO4J_URI=bolt://neo4j-staging:7687" \
  --env="NEO4J_PASSWORD=staging_password" \
  --restart=Never \
  --rm -i --tty
```

#### Step 4: Deploy Application Services

```bash
# Deploy gateway
kubectl apply -f k8s/staging/gateway-deployment.yaml

# Deploy services
kubectl apply -f k8s/staging/services-deployment.yaml

# Deploy adapters
kubectl apply -f k8s/staging/adapters-deployment.yaml

# Deploy orchestration
kubectl apply -f k8s/staging/orchestration-deployment.yaml

# Deploy knowledge graph
kubectl apply -f k8s/staging/kg-deployment.yaml

# Deploy storage
kubectl apply -f k8s/staging/storage-deployment.yaml
```

#### Step 5: Deploy Ingress and Services

```bash
# Deploy services
kubectl apply -f k8s/staging/gateway-service.yaml
kubectl apply -f k8s/staging/services-service.yaml
kubectl apply -f k8s/staging/adapters-service.yaml
kubectl apply -f k8s/staging/orchestration-service.yaml
kubectl apply -f k8s/staging/kg-service.yaml
kubectl apply -f k8s/staging/storage-service.yaml

# Deploy ingress
kubectl apply -f k8s/staging/ingress.yaml

# Wait for deployment to be ready
kubectl rollout status deployment/medical-kg-rev-gateway -n staging --timeout=600s
```

#### Step 6: Verify Deployment

```bash
# Check pod status
kubectl get pods -n staging

# Check service status
kubectl get services -n staging

# Check ingress status
kubectl get ingress -n staging

# Run health checks
python scripts/post_deployment_health_checks.py --environment=staging

# Run smoke tests
python scripts/smoke_tests.py --environment=staging
```

### Automated Deployment

#### GitHub Actions Workflow

```yaml
# .github/workflows/deploy-staging.yml
name: Deploy to Staging

on:
  push:
    branches: [develop]

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    environment: staging

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

      - name: Deploy to staging
        run: |
          export KUBECONFIG=kubeconfig
          kubectl apply -f k8s/staging/
          kubectl rollout status deployment/medical-kg-rev-gateway -n staging

      - name: Run health checks
        run: |
          export KUBECONFIG=kubeconfig
          python scripts/post_deployment_health_checks.py --environment=staging

      - name: Run smoke tests
        run: |
          export KUBECONFIG=kubeconfig
          python scripts/smoke_tests.py --environment=staging

      - name: Notify deployment
        uses: 8398a7/action-slack@v3
        if: always()
        with:
          status: ${{ job.status }}
          channel: '#deployments'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## Production Environment Deployment

### Blue-Green Deployment

#### Step 1: Prepare Blue Environment

```bash
# Set kubectl context
kubectl config use-context production

# Deploy blue environment
kubectl apply -f k8s/production/blue/

# Wait for blue deployment to be ready
kubectl rollout status deployment/medical-kg-rev-gateway-blue -n production --timeout=600s
```

#### Step 2: Deploy Green Environment

```bash
# Deploy green environment
kubectl apply -f k8s/production/green/

# Wait for green deployment to be ready
kubectl rollout status deployment/medical-kg-rev-gateway-green -n production --timeout=600s
```

#### Step 3: Switch Traffic to Green

```bash
# Update service to point to green
kubectl patch service medical-kg-rev-gateway -n production -p '{"spec":{"selector":{"version":"green"}}}'

# Verify traffic is flowing to green
kubectl get pods -l version=green -n production
```

#### Step 4: Verify Green Deployment

```bash
# Run health checks
python scripts/post_deployment_health_checks.py --environment=production

# Run performance tests
python scripts/post_deployment_performance_checks.py --environment=production

# Monitor metrics
kubectl top pods -n production
```

#### Step 5: Cleanup Blue Environment

```bash
# Remove blue environment
kubectl delete -f k8s/production/blue/

# Verify cleanup
kubectl get pods -l version=blue -n production
```

### Canary Deployment

#### Step 1: Deploy Canary Version

```bash
# Deploy canary version
kubectl apply -f k8s/production/canary/

# Wait for canary deployment to be ready
kubectl rollout status deployment/medical-kg-rev-gateway-canary -n production --timeout=600s
```

#### Step 2: Route Traffic to Canary

```bash
# Route 10% traffic to canary
kubectl apply -f k8s/production/canary-traffic-split.yaml

# Monitor canary metrics
kubectl get pods -l version=canary -n production
```

#### Step 3: Gradually Increase Traffic

```bash
# Increase to 25%
kubectl patch service medical-kg-rev-gateway -n production -p '{"spec":{"traffic":{"canary":{"weight":25}}}}'

# Wait and monitor
sleep 300

# Increase to 50%
kubectl patch service medical-kg-rev-gateway -n production -p '{"spec":{"traffic":{"canary":{"weight":50}}}}'

# Wait and monitor
sleep 300

# Increase to 75%
kubectl patch service medical-kg-rev-gateway -n production -p '{"spec":{"traffic":{"canary":{"weight":75}}}}'

# Wait and monitor
sleep 300
```

#### Step 4: Complete Canary Deployment

```bash
# Route 100% traffic to canary
kubectl patch service medical-kg-rev-gateway -n production -p '{"spec":{"traffic":{"canary":{"weight":100}}}}'

# Update stable service
kubectl patch service medical-kg-rev-gateway-stable -n production -p '{"spec":{"selector":{"version":"canary"}}}'

# Remove canary service
kubectl delete service medical-kg-rev-gateway-canary -n production
```

### Automated Production Deployment

#### GitHub Actions Workflow

```yaml
# .github/workflows/deploy-production.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy-production:
    runs-on: ubuntu-latest
    environment: production
    needs: deploy-staging

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'v1.28.0'

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG_PRODUCTION }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to production
        run: |
          export KUBECONFIG=kubeconfig
          kubectl apply -f k8s/production/
          kubectl rollout status deployment/medical-kg-rev-gateway -n production

      - name: Run post-deployment tests
        run: |
          export KUBECONFIG=kubeconfig
          python scripts/post_deployment_tests.py --environment=production

      - name: Notify deployment
        uses: 8398a7/action-slack@v3
        if: always()
        with:
          status: ${{ job.status }}
          channel: '#deployments'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## Database Deployment

### PostgreSQL Deployment

#### Step 1: Deploy PostgreSQL

```yaml
# k8s/production/postgresql.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-production
  namespace: production
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres-production
  template:
    metadata:
      labels:
        app: postgres-production
    spec:
      containers:
      - name: postgres
        image: postgres:14
        env:
        - name: POSTGRES_DB
          value: "medical_kg_production"
        - name: POSTGRES_USER
          value: "prod"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secrets
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-production
  namespace: production
spec:
  selector:
    app: postgres-production
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

#### Step 2: Run Migrations

```bash
# Run PostgreSQL migrations
kubectl run migration-job --image=medical-kg-rev/migration:latest \
  --env="DATABASE_URL=postgresql://prod:prod@postgres-production:5432/medical_kg_production" \
  --restart=Never \
  --rm -i --tty
```

### Neo4j Deployment

#### Step 1: Deploy Neo4j

```yaml
# k8s/production/neo4j.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neo4j-production
  namespace: production
spec:
  replicas: 1
  selector:
    matchLabels:
      app: neo4j-production
  template:
    metadata:
      labels:
        app: neo4j-production
    spec:
      containers:
      - name: neo4j
        image: neo4j:5
        env:
        - name: NEO4J_AUTH
          value: "neo4j/production_password"
        - name: NEO4J_PLUGINS
          value: '["apoc"]'
        ports:
        - containerPort: 7687
        - containerPort: 7474
        volumeMounts:
        - name: neo4j-storage
          mountPath: /data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: neo4j-storage
        persistentVolumeClaim:
          claimName: neo4j-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: neo4j-production
  namespace: production
spec:
  selector:
    app: neo4j-production
  ports:
  - port: 7687
    targetPort: 7687
    name: bolt
  - port: 7474
    targetPort: 7474
    name: http
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: neo4j-pvc
  namespace: production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
```

#### Step 2: Run Neo4j Migrations

```bash
# Run Neo4j migrations
kubectl run neo4j-migration-job --image=medical-kg-rev/neo4j-migration:latest \
  --env="NEO4J_URI=bolt://neo4j-production:7687" \
  --env="NEO4J_PASSWORD=production_password" \
  --restart=Never \
  --rm -i --tty
```

## Monitoring Deployment

### Health Checks

#### Application Health Checks

```python
# scripts/health_checks.py
import requests
import time
import sys
from typing import Dict, List, Tuple

class HealthChecker:
    """Application health checker."""

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

    def run_all_checks(self) -> bool:
        """Run all health checks."""
        print("Running health checks...")

        checks = [
            self.check_health_endpoint,
            self.check_ready_endpoint,
            self.check_database_connectivity
        ]

        all_passed = True
        for check in checks:
            passed, message = check()
            status_emoji = "✅" if passed else "❌"
            print(f"{status_emoji} {message}")
            if not passed:
                all_passed = False

        print(f"\nOverall result: {'✅ SUCCESS' if all_passed else '❌ FAILURE'}")
        return all_passed

def main():
    """Main health check function."""
    import argparse

    parser = argparse.ArgumentParser(description="Health checks")
    parser.add_argument("--environment", required=True, choices=["staging", "production"])
    parser.add_argument("--base-url", help="Base URL for health checks")

    args = parser.parse_args()

    if args.base_url:
        base_url = args.base_url
    else:
        base_urls = {
            "staging": "https://staging.medical-kg-rev.com",
            "production": "https://api.medical-kg-rev.com"
        }
        base_url = base_urls[args.environment]

    checker = HealthChecker(base_url)
    success = checker.run_all_checks()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

### Performance Monitoring

#### Performance Checks

```python
# scripts/performance_checks.py
import requests
import time
import statistics
from typing import List, Dict, Any

class PerformanceChecker:
    """Performance checker."""

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

    def run_performance_checks(self) -> bool:
        """Run performance checks."""
        print("Running performance checks...")

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
    """Main performance check function."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance checks")
    parser.add_argument("--environment", required=True, choices=["staging", "production"])
    parser.add_argument("--base-url", help="Base URL for performance checks")

    args = parser.parse_args()

    if args.base_url:
        base_url = args.base_url
    else:
        base_urls = {
            "staging": "https://staging.medical-kg-rev.com",
            "production": "https://api.medical-kg-rev.com"
        }
        base_url = base_urls[args.environment]

    checker = PerformanceChecker(base_url)
    success = checker.run_performance_checks()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Deployment Issues

#### 1. Pod Startup Failures

```bash
# Check pod status
kubectl get pods -n production

# Check pod logs
kubectl logs -f deployment/medical-kg-rev-gateway -n production

# Check pod events
kubectl describe pod <pod-name> -n production

# Check resource usage
kubectl top pods -n production
```

#### 2. Service Connectivity Issues

```bash
# Check service status
kubectl get services -n production

# Check service endpoints
kubectl get endpoints -n production

# Test service connectivity
kubectl run test-pod --image=busybox --rm -it --restart=Never -- nslookup medical-kg-rev-gateway
```

#### 3. Database Connection Issues

```bash
# Check database pods
kubectl get pods -l app=postgres-production -n production

# Check database logs
kubectl logs -f deployment/postgres-production -n production

# Test database connectivity
kubectl run test-pod --image=postgres:14 --rm -it --restart=Never -- psql -h postgres-production -U prod -d medical_kg_production
```

### Debug Commands

```bash
# Check deployment status
kubectl get deployments -n production
kubectl rollout status deployment/medical-kg-rev-gateway -n production

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

## Best Practices

### Deployment Strategy

1. **Automated Deployments**: Use CI/CD for consistent deployments
2. **Blue-Green Deployment**: Minimize downtime with blue-green strategy
3. **Canary Releases**: Gradually roll out changes to production
4. **Health Checks**: Implement comprehensive health checks
5. **Rollback Plan**: Always have a rollback strategy ready

### Quality Assurance

1. **Testing**: Comprehensive testing at each stage
2. **Validation**: Post-deployment validation and monitoring
3. **Performance**: Performance testing and validation
4. **Security**: Security scanning and validation
5. **Documentation**: Keep deployment documentation updated

### Monitoring and Alerting

1. **Real-time Monitoring**: Monitor deployments in real-time
2. **Alerting**: Set up alerts for deployment failures
3. **Logging**: Comprehensive logging for troubleshooting
4. **Metrics**: Collect and analyze deployment metrics
5. **Dashboards**: Use dashboards for deployment visibility

## Related Documentation

- [Deployment Overview](deployment_overview.md)
- [Infrastructure Requirements](infrastructure_requirements.md)
- [Rollback Procedures](rollback_procedures.md)
- [Monitoring and Logging](monitoring_logging.md)
- [Security Considerations](security_considerations.md)
- [Disaster Recovery Plan](disaster_recovery_plan.md)
