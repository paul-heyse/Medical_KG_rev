# Deployment Process Overview

This document provides a comprehensive overview of the end-to-end deployment process for the Medical_KG_rev system, covering the journey from development to production with detailed procedures, best practices, and automation strategies.

## Overview

The Medical_KG_rev deployment process follows a multi-stage approach with automated testing, validation, and deployment across different environments. This ensures code quality, system reliability, and minimal downtime during deployments.

## Deployment Architecture

### Deployment Pipeline

```mermaid
graph TD
    A[Code Push] --> B[Development Environment]
    B --> C[Code Quality Checks]
    C --> D[Unit Tests]
    D --> E[Integration Tests]
    E --> F[Contract Tests]
    F --> G[Performance Tests]
    G --> H[Security Scans]
    H --> I[Build Artifacts]
    I --> J[Staging Environment]
    J --> K[Staging Validation]
    K --> L[Production Environment]
    L --> M[Post-Deployment Validation]
    M --> N[Monitoring & Alerting]

    C --> O[Quality Gate]
    D --> O
    E --> O
    F --> O
    G --> O
    H --> O
    O --> P[Fail Pipeline]

    K --> Q[Staging Gate]
    Q --> R[Fail Deployment]
```

### Deployment Environments

#### Development Environment

- **Purpose**: Local development and testing
- **URL**: `http://localhost:8000`
- **Database**: Local PostgreSQL/Neo4j/Redis
- **Deployment**: Manual via `docker-compose up`
- **Features**: Hot reload, debug mode, detailed logging

#### Staging Environment

- **Purpose**: Pre-production testing and validation
- **URL**: `https://staging.medical-kg-rev.com`
- **Database**: Staging PostgreSQL/Neo4j/Redis
- **Deployment**: Automated via CI/CD on `develop` branch
- **Features**: Production-like configuration, performance testing

#### Production Environment

- **Purpose**: Live system serving end users
- **URL**: `https://api.medical-kg-rev.com`
- **Database**: Production PostgreSQL/Neo4j/Redis
- **Deployment**: Automated via CI/CD on `main` branch
- **Features**: High availability, monitoring, alerting

## Deployment Stages

### Stage 1: Code Preparation

#### Development Workflow

1. **Feature Development**

   ```bash
   # Create feature branch
   git checkout -b feature/add-new-adapter

   # Make changes
   git add .
   git commit -m "feat: add OpenAlex adapter implementation"

   # Push to remote
   git push origin feature/add-new-adapter
   ```

2. **Code Review Process**
   - Create pull request
   - Automated code quality checks
   - Peer code review
   - Address feedback and make changes
   - Merge to develop branch

3. **Integration Testing**

   ```bash
   # Run integration tests
   pytest tests/integration/

   # Run contract tests
   pytest tests/contract/

   # Run performance tests
   k6 run tests/performance/load_test.js
   ```

#### Quality Gates

- **Code Quality**: Linting, type checking, security scans
- **Test Coverage**: Minimum 80% code coverage
- **Performance**: API response time < 500ms
- **Security**: No high-severity vulnerabilities

### Stage 2: Build and Package

#### Docker Image Build

```dockerfile
# Multi-stage build for production
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy built application
COPY --from=builder /app /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "Medical_KG_rev.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Build Pipeline

```yaml
# .github/workflows/build.yml
name: Build and Package

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [gateway, services, adapters, orchestration, kg, storage]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: medical-kg-rev/${{ matrix.service }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/${{ matrix.service }}/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
```

### Stage 3: Staging Deployment

#### Staging Environment Setup

```yaml
# k8s/staging/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: staging
  labels:
    environment: staging
    app: medical-kg-rev
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
```

#### Staging Deployment Process

1. **Deploy to Staging**

   ```bash
   # Deploy to staging environment
   kubectl apply -f k8s/staging/

   # Wait for deployment to be ready
   kubectl rollout status deployment/medical-kg-rev-gateway -n staging
   ```

2. **Staging Validation**

   ```bash
   # Run smoke tests
   python scripts/smoke_tests.py --environment=staging

   # Run integration tests
   pytest tests/integration/ --environment=staging

   # Run performance tests
   k6 run tests/performance/staging_load_test.js
   ```

3. **Staging Approval**
   - Manual review of staging environment
   - Performance validation
   - Security checks
   - User acceptance testing

### Stage 4: Production Deployment

#### Production Environment Setup

```yaml
# k8s/production/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    environment: production
    app: medical-kg-rev
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

#### Production Deployment Strategies

##### Blue-Green Deployment

```yaml
# k8s/production/blue-green-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: medical-kg-rev-gateway
  namespace: production
spec:
  replicas: 3
  strategy:
    blueGreen:
      activeService: medical-kg-rev-gateway-active
      previewService: medical-kg-rev-gateway-preview
      autoPromotionEnabled: false
      scaleDownDelaySeconds: 30
      prePromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: medical-kg-rev-gateway-preview
      postPromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: medical-kg-rev-gateway-active
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

##### Canary Deployment

```yaml
# k8s/production/canary-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: medical-kg-rev-gateway
  namespace: production
spec:
  replicas: 5
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
      - setWeight: 60
      - pause: {duration: 10m}
      - analysis:
          templates:
          - templateName: success-rate
          args:
          - name: service-name
            value: medical-kg-rev-gateway
      - setWeight: 80
      - pause: {duration: 10m}
      - analysis:
          templates:
          - templateName: success-rate
          args:
          - name: service-name
            value: medical-kg-rev-gateway
      canaryService: medical-kg-rev-gateway-canary
      stableService: medical-kg-rev-gateway-stable
      trafficRouting:
        nginx:
          stableIngress: medical-kg-rev-gateway-stable
          annotationPrefix: nginx.ingress.kubernetes.io
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

### Stage 5: Post-Deployment Validation

#### Health Checks

```python
# scripts/post_deployment_health_checks.py
import requests
import time
import sys
from typing import Dict, List, Tuple

class HealthChecker:
    """Post-deployment health checker."""

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

    def run_all_checks(self) -> bool:
        """Run all health checks."""
        print("Running post-deployment health checks...")

        # Health endpoint
        health_ok, health_msg = self.check_health_endpoint()
        print(f"Health endpoint: {'✅' if health_ok else '❌'} {health_msg}")

        # Ready endpoint
        ready_ok, ready_msg = self.check_ready_endpoint()
        print(f"Ready endpoint: {'✅' if ready_ok else '❌'} {ready_msg}")

        # API endpoints
        api_results = self.check_api_endpoints()
        for ok, msg in api_results:
            print(f"API endpoint: {'✅' if ok else '❌'} {msg}")

        # Overall result
        all_checks = [health_ok, ready_ok] + [ok for ok, _ in api_results]
        overall_success = all(all_checks)

        print(f"\nOverall result: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
        return overall_success

def main():
    """Main health check function."""
    import argparse

    parser = argparse.ArgumentParser(description="Post-deployment health checks")
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

#### Performance Validation

```python
# scripts/post_deployment_performance_checks.py
import requests
import time
import statistics
from typing import List, Dict, Any

class PerformanceChecker:
    """Post-deployment performance checker."""

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
        print("Running post-deployment performance checks...")

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

    parser = argparse.ArgumentParser(description="Post-deployment performance checks")
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

## Deployment Automation

### CI/CD Pipeline Integration

#### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main, develop]

jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/develop'
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

  deploy-production:
    if: github.ref == 'refs/heads/main'
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

### Deployment Scripts

#### Automated Deployment Script

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

ENVIRONMENT=$1
VERSION=$2

if [ -z "$ENVIRONMENT" ] || [ -z "$VERSION" ]; then
    echo "Usage: $0 <environment> <version>"
    echo "Environments: staging, production"
    exit 1
fi

echo "Deploying Medical_KG_rev version $VERSION to $ENVIRONMENT..."

# Set environment-specific variables
case $ENVIRONMENT in
    staging)
        NAMESPACE="staging"
        KUBECONFIG_FILE="kubeconfig-staging"
        BASE_URL="https://staging.medical-kg-rev.com"
        ;;
    production)
        NAMESPACE="production"
        KUBECONFIG_FILE="kubeconfig-production"
        BASE_URL="https://api.medical-kg-rev.com"
        ;;
    *)
        echo "Invalid environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Configure kubectl
export KUBECONFIG=$KUBECONFIG_FILE

# Deploy application
echo "Deploying application..."
kubectl apply -f k8s/$ENVIRONMENT/

# Wait for deployment
echo "Waiting for deployment to complete..."
kubectl rollout status deployment/medical-kg-rev-gateway -n $NAMESPACE

# Run health checks
echo "Running health checks..."
python scripts/post_deployment_health_checks.py --environment=$ENVIRONMENT --base-url=$BASE_URL

# Run performance checks
echo "Running performance checks..."
python scripts/post_deployment_performance_checks.py --environment=$ENVIRONMENT --base-url=$BASE_URL

echo "✅ Deployment to $ENVIRONMENT completed successfully!"
```

## Deployment Monitoring

### Real-time Monitoring

#### Deployment Status Dashboard

```python
# scripts/deployment_monitor.py
import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Any

class DeploymentMonitor:
    """Monitor deployment status in real-time."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.metrics = []

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect deployment metrics."""
        try:
            # Health check
            health_response = requests.get(f"{self.base_url}/health", timeout=10)
            health_status = health_response.status_code == 200

            # Metrics endpoint
            metrics_response = requests.get(f"{self.base_url}/metrics", timeout=10)
            metrics_data = metrics_response.text if metrics_response.status_code == 200 else ""

            # Performance metrics
            start_time = time.time()
            api_response = requests.get(f"{self.base_url}/api/v1/documents", timeout=10)
            response_time = time.time() - start_time

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "health_status": health_status,
                "response_time": response_time,
                "api_status": api_response.status_code,
                "metrics": metrics_data
            }
        except Exception as e:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "health_status": False,
                "response_time": float('inf'),
                "api_status": 0
            }

    def monitor_deployment(self, duration_minutes: int = 30):
        """Monitor deployment for specified duration."""
        print(f"Monitoring deployment for {duration_minutes} minutes...")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        while time.time() < end_time:
            metrics = self.collect_metrics()
            self.metrics.append(metrics)

            # Print status
            status_emoji = "✅" if metrics.get("health_status", False) else "❌"
            print(f"{status_emoji} {metrics['timestamp']} - Health: {metrics.get('health_status', False)} - Response: {metrics.get('response_time', 0):.3f}s")

            time.sleep(60)  # Check every minute

        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate deployment summary."""
        if not self.metrics:
            print("No metrics collected")
            return

        # Calculate statistics
        health_checks = [m.get("health_status", False) for m in self.metrics]
        response_times = [m.get("response_time", 0) for m in self.metrics if m.get("response_time", 0) != float('inf')]

        health_success_rate = sum(health_checks) / len(health_checks) * 100
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0

        print(f"\n📊 Deployment Summary:")
        print(f"   Health Success Rate: {health_success_rate:.1f}%")
        print(f"   Average Response Time: {avg_response_time:.3f}s")
        print(f"   Maximum Response Time: {max_response_time:.3f}s")
        print(f"   Total Checks: {len(self.metrics)}")

def main():
    """Main monitoring function."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor deployment")
    parser.add_argument("--environment", required=True, choices=["staging", "production"])
    parser.add_argument("--duration", type=int, default=30, help="Monitoring duration in minutes")
    parser.add_argument("--base-url", help="Base URL for monitoring")

    args = parser.parse_args()

    if args.base_url:
        base_url = args.base_url
    else:
        base_urls = {
            "staging": "https://staging.medical-kg-rev.com",
            "production": "https://api.medical-kg-rev.com"
        }
        base_url = base_urls[args.environment]

    monitor = DeploymentMonitor(base_url)
    monitor.monitor_deployment(args.duration)

if __name__ == "__main__":
    main()
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

## Troubleshooting

### Common Deployment Issues

#### 1. Deployment Failures

```bash
# Check deployment status
kubectl get deployments -n production

# Check pod logs
kubectl logs -f deployment/medical-kg-rev-gateway -n production

# Check events
kubectl get events -n production --sort-by='.lastTimestamp'
```

#### 2. Health Check Failures

```bash
# Check health endpoint
curl -f http://medical-kg-rev-gateway/health

# Check ready endpoint
curl -f http://medical-kg-rev-gateway/ready

# Check service status
kubectl get services -n production
```

#### 3. Performance Issues

```bash
# Check resource usage
kubectl top pods -n production

# Check resource limits
kubectl describe deployment medical-kg-rev-gateway -n production

# Check horizontal pod autoscaler
kubectl get hpa -n production
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
```

## Related Documentation

- [Infrastructure Requirements](infrastructure_requirements.md)
- [Deployment Procedures](deployment_procedures.md)
- [Rollback Procedures](rollback_procedures.md)
- [Monitoring and Logging](monitoring_logging.md)
- [Security Considerations](security_considerations.md)
- [Disaster Recovery Plan](disaster_recovery_plan.md)
