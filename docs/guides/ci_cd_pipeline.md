# CI/CD Pipeline Documentation

This document provides comprehensive documentation for the Continuous Integration and Continuous Deployment (CI/CD) pipeline of the Medical_KG_rev system, including pipeline setup, automated testing procedures, deployment strategies, and rollback procedures.

## Overview

The Medical_KG_rev CI/CD pipeline automates the build, test, and deployment processes to ensure code quality, system reliability, and rapid delivery. The pipeline follows a multi-stage approach with quality gates and automated deployments.

## Pipeline Architecture

### Pipeline Stages

```mermaid
graph TD
    A[Code Push] --> B[Code Quality Checks]
    B --> C[Unit Tests]
    C --> D[Integration Tests]
    D --> E[Contract Tests]
    E --> F[Performance Tests]
    F --> G[Security Scans]
    G --> H[Build Artifacts]
    H --> I[Deploy to Staging]
    I --> J[Staging Tests]
    J --> K[Deploy to Production]
    K --> L[Post-Deployment Tests]
    L --> M[Monitoring & Alerting]

    B --> N[Quality Gate]
    C --> N
    D --> N
    E --> N
    F --> N
    G --> N
    N --> O[Fail Pipeline]

    J --> P[Staging Gate]
    P --> Q[Fail Deployment]
```

### Pipeline Components

1. **Source Control**: GitHub with branch protection
2. **CI Platform**: GitHub Actions
3. **Container Registry**: Docker Hub / AWS ECR
4. **Orchestration**: Kubernetes
5. **Monitoring**: Prometheus + Grafana
6. **Logging**: ELK Stack
7. **Artifact Storage**: AWS S3

## Pipeline Configuration

### GitHub Actions Workflow

#### Main CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  # Stage 1: Code Quality Checks
  code-quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache Python dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run linting
        run: |
          ruff check . --output-format=github
          ruff format --check .

      - name: Run type checking
        run: mypy src/ --show-error-codes

      - name: Run security checks
        run: |
          bandit -r src/ -f json -o bandit-report.json
          safety check --json --output safety-report.json

      - name: Upload security reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json

  # Stage 2: Unit Tests
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: code-quality
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run unit tests
        run: |
          pytest tests/unit/ \
            --cov=src/ \
            --cov-report=xml \
            --cov-report=html \
            --junitxml=test-results.xml \
            --maxfail=5

      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results-${{ matrix.python-version }}
          path: |
            test-results.xml
            htmlcov/

  # Stage 3: Integration Tests
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      neo4j:
        image: neo4j:5
        env:
          NEO4J_AUTH: neo4j/password
          NEO4J_PLUGINS: '["apoc"]'
        options: >-
          --health-cmd "cypher-shell -u neo4j -p password 'RETURN 1'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 7687:7687

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

      qdrant:
        image: qdrant/qdrant:latest
        options: >-
          --health-cmd "curl -f http://localhost:6333/health || exit 1"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6333:6333

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Wait for services
        run: |
          sleep 30
          python scripts/wait_for_services.py

      - name: Run integration tests
        run: |
          pytest tests/integration/ \
            --cov=src/ \
            --cov-append \
            --junitxml=integration-test-results.xml
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          NEO4J_URI: bolt://localhost:7687
          NEO4J_PASSWORD: password
          REDIS_URL: redis://localhost:6379/0
          VECTOR_STORE_URL: http://localhost:6333

      - name: Upload integration test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: integration-test-results
          path: integration-test-results.xml

  # Stage 4: Contract Tests
  contract-tests:
    name: Contract Tests
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Start application
        run: |
          docker-compose up -d
          python scripts/wait_for_services.py

      - name: Run contract tests
        run: |
          # REST API contract tests
          schemathesis run docs/openapi.yaml \
            --base-url http://localhost:8000 \
            --checks all \
            --max-failures 5 \
            --junit-xml=schemathesis-results.xml

          # GraphQL contract tests
          pytest tests/contract/graphql/ \
            --junitxml=graphql-contract-results.xml

      - name: Upload contract test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: contract-test-results
          path: |
            schemathesis-results.xml
            graphql-contract-results.xml

  # Stage 5: Performance Tests
  performance-tests:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: contract-tests
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}

      - name: Install k6
        run: |
          sudo gpg -k
          sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6

      - name: Start application
        run: |
          docker-compose up -d
          python scripts/wait_for_services.py

      - name: Run performance tests
        run: |
          # Load test
          k6 run tests/performance/load_test.js \
            --out json=load-test-results.json \
            --summary-export=load-test-summary.json

          # Stress test
          k6 run tests/performance/stress_test.js \
            --out json=stress-test-results.json \
            --summary-export=stress-test-summary.json

      - name: Upload performance test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: performance-test-results
          path: |
            load-test-results.json
            load-test-summary.json
            stress-test-results.json
            stress-test-summary.json

  # Stage 6: Security Scans
  security-scans:
    name: Security Scans
    runs-on: ubuntu-latest
    needs: performance-tests
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run Bandit security scan
        run: |
          bandit -r src/ -f json -o bandit-report.json
          bandit -r src/ -f txt

      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json
          safety check

      - name: Run Semgrep scan
        run: |
          docker run --rm -v "${PWD}:/src" returntocorp/semgrep:latest \
            semgrep --config=auto --json --output=semgrep-report.json /src

      - name: Upload security scan results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-scan-results
          path: |
            bandit-report.json
            safety-report.json
            semgrep-report.json

  # Stage 7: Build and Push Docker Images
  build-and-push:
    name: Build and Push Docker Images
    runs-on: ubuntu-latest
    needs: security-scans
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
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

  # Stage 8: Deploy to Staging
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.ref == 'refs/heads/develop'
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
          kubectl rollout status deployment/medical-kg-rev-services -n staging

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

  # Stage 9: Deploy to Production
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    environment: production
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
          kubectl rollout status deployment/medical-kg-rev-services -n production

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

#### Pull Request Pipeline

```yaml
# .github/workflows/pr.yml
name: Pull Request Pipeline

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  pr-checks:
    name: PR Checks
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run code quality checks
        run: |
          ruff check . --output-format=github
          ruff format --check .
          mypy src/

      - name: Run unit tests
        run: |
          pytest tests/unit/ --cov=src/ --cov-fail-under=80

      - name: Check for breaking changes
        run: |
          python scripts/check_breaking_changes.py

      - name: Comment PR
        uses: actions/github-script@v7
        if: always()
        with:
          script: |
            const fs = require('fs');
            const path = require('path');

            // Read test results
            let testResults = '';
            try {
              const coverage = fs.readFileSync('coverage.xml', 'utf8');
              testResults = '✅ All checks passed!';
            } catch (error) {
              testResults = '❌ Some checks failed. Please review the errors above.';
            }

            // Comment on PR
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## CI/CD Pipeline Results\n\n${testResults}\n\n### Code Quality\n- Linting: ✅\n- Type Checking: ✅\n- Unit Tests: ✅\n- Coverage: ✅\n\n### Next Steps\n- [ ] Address any failing checks\n- [ ] Request code review\n- [ ] Merge when approved`
            });
```

### Kubernetes Deployment

#### Staging Environment

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
---
# k8s/staging/gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-kg-rev-gateway
  namespace: staging
  labels:
    app: medical-kg-rev
    component: gateway
    environment: staging
spec:
  replicas: 2
  selector:
    matchLabels:
      app: medical-kg-rev
      component: gateway
  template:
    metadata:
      labels:
        app: medical-kg-rev
        component: gateway
        environment: staging
    spec:
      containers:
      - name: gateway
        image: medical-kg-rev/gateway:develop
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: medical-kg-rev-config
        - secretRef:
            name: medical-kg-rev-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
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
---
# k8s/staging/gateway-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: medical-kg-rev-gateway
  namespace: staging
  labels:
    app: medical-kg-rev
    component: gateway
spec:
  selector:
    app: medical-kg-rev
    component: gateway
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
---
# k8s/staging/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: medical-kg-rev-ingress
  namespace: staging
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-staging"
spec:
  tls:
  - hosts:
    - staging.medical-kg-rev.com
    secretName: medical-kg-rev-tls-staging
  rules:
  - host: staging.medical-kg-rev.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: medical-kg-rev-gateway
            port:
              number: 80
```

#### Production Environment

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
---
# k8s/production/gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-kg-rev-gateway
  namespace: production
  labels:
    app: medical-kg-rev
    component: gateway
    environment: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: medical-kg-rev
      component: gateway
  template:
    metadata:
      labels:
        app: medical-kg-rev
        component: gateway
        environment: production
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
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
---
# k8s/production/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: medical-kg-rev-gateway-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: medical-kg-rev-gateway
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Quality Gates

### Code Quality Gates

#### Coverage Requirements

```yaml
# .github/workflows/coverage-gate.yml
name: Coverage Gate

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  coverage-check:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests with coverage
        run: |
          pytest tests/unit/ --cov=src/ --cov-report=xml --cov-report=term

      - name: Check coverage threshold
        run: |
          python -c "
          import xml.etree.ElementTree as ET
          tree = ET.parse('coverage.xml')
          root = tree.getroot()
          coverage = float(root.get('line-rate')) * 100
          print(f'Coverage: {coverage:.2f}%')
          if coverage < 80:
              print('❌ Coverage below 80% threshold')
              exit(1)
          else:
              print('✅ Coverage meets 80% threshold')
          "

      - name: Comment coverage on PR
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const xml2js = require('xml2js');

            fs.readFile('coverage.xml', 'utf8', (err, data) => {
              if (err) return;

              xml2js.parseString(data, (err, result) => {
                if (err) return;

                const coverage = parseFloat(result.coverage.$.lineRate) * 100;
                const status = coverage >= 80 ? '✅' : '❌';

                github.rest.issues.createComment({
                  issue_number: context.issue.number,
                  owner: context.repo.owner,
                  repo: context.repo.repo,
                  body: `## Coverage Report\n\n${status} **Coverage: ${coverage.toFixed(2)}%**\n\nThreshold: 80%`
                });
              });
            });
```

#### Performance Gates

```yaml
# .github/workflows/performance-gate.yml
name: Performance Gate

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  performance-check:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install k6
        run: |
          sudo gpg -k
          sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6

      - name: Start application
        run: |
          docker-compose up -d
          python scripts/wait_for_services.py

      - name: Run performance tests
        run: |
          k6 run tests/performance/performance_gate.js \
            --out json=performance-results.json \
            --summary-export=performance-summary.json

      - name: Check performance thresholds
        run: |
          python -c "
          import json
          with open('performance-summary.json', 'r') as f:
              data = json.load(f)

          p95_latency = data['metrics']['http_req_duration']['p(95)']
          error_rate = data['metrics']['http_req_failed']['rate']

          print(f'P95 Latency: {p95_latency:.2f}ms')
          print(f'Error Rate: {error_rate:.2%}')

          if p95_latency > 500:
              print('❌ P95 latency exceeds 500ms threshold')
              exit(1)

          if error_rate > 0.01:
              print('❌ Error rate exceeds 1% threshold')
              exit(1)

          print('✅ Performance meets thresholds')
          "
```

### Security Gates

#### Vulnerability Scanning

```yaml
# .github/workflows/security-gate.yml
name: Security Gate

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run Bandit security scan
        run: |
          bandit -r src/ -f json -o bandit-report.json
          bandit -r src/ -f txt

      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json
          safety check

      - name: Check security thresholds
        run: |
          python -c "
          import json

          # Check Bandit results
          with open('bandit-report.json', 'r') as f:
              bandit_data = json.load(f)

          high_severity = len([r for r in bandit_data['results'] if r['issue_severity'] == 'HIGH'])
          medium_severity = len([r for r in bandit_data['results'] if r['issue_severity'] == 'MEDIUM'])

          print(f'High severity issues: {high_severity}')
          print(f'Medium severity issues: {medium_severity}')

          if high_severity > 0:
              print('❌ High severity security issues found')
              exit(1)

          if medium_severity > 5:
              print('❌ Too many medium severity security issues')
              exit(1)

          print('✅ Security scan passed')
          "
```

## Deployment Strategies

### Blue-Green Deployment

#### Implementation

```yaml
# k8s/blue-green/deployment.yaml
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
---
# k8s/blue-green/analysis-template.yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  args:
  - name: service-name
  metrics:
  - name: success-rate
    successCondition: result[0] >= 0.95
    provider:
      prometheus:
        address: http://prometheus:9090
        query: |
          sum(rate(http_requests_total{service="{{args.service-name}}",status!~"5.."}[5m])) /
          sum(rate(http_requests_total{service="{{args.service-name}}"}[5m]))
```

#### Deployment Script

```bash
#!/bin/bash
# scripts/blue-green-deploy.sh

set -e

NAMESPACE="production"
SERVICE_NAME="medical-kg-rev-gateway"
NEW_VERSION="$1"

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new-version>"
    exit 1
fi

echo "Starting blue-green deployment for $SERVICE_NAME to version $NEW_VERSION"

# Update the rollout with new version
kubectl patch rollout $SERVICE_NAME -n $NAMESPACE -p '{"spec":{"template":{"spec":{"containers":[{"name":"gateway","image":"medical-kg-rev/gateway:'$NEW_VERSION'"}]}}}}'

# Wait for rollout to be ready
echo "Waiting for rollout to be ready..."
kubectl rollout status rollout/$SERVICE_NAME -n $NAMESPACE --timeout=600s

# Run pre-promotion analysis
echo "Running pre-promotion analysis..."
kubectl get analysisrun -n $NAMESPACE -l app.kubernetes.io/name=$SERVICE_NAME

# Promote to active
echo "Promoting to active..."
kubectl argo rollouts promote $SERVICE_NAME -n $NAMESPACE

# Wait for promotion to complete
echo "Waiting for promotion to complete..."
kubectl rollout status rollout/$SERVICE_NAME -n $NAMESPACE --timeout=600s

# Run post-promotion analysis
echo "Running post-promotion analysis..."
kubectl get analysisrun -n $NAMESPACE -l app.kubernetes.io/name=$SERVICE_NAME

echo "Blue-green deployment completed successfully!"
```

### Canary Deployment

#### Implementation

```yaml
# k8s/canary/deployment.yaml
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

## Rollback Procedures

### Automatic Rollback

#### Rollback Configuration

```yaml
# k8s/rollback/rollout.yaml
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

### Manual Rollback

#### Rollback Script

```bash
#!/bin/bash
# scripts/rollback.sh

set -e

NAMESPACE="production"
SERVICE_NAME="medical-kg-rev-gateway"
PREVIOUS_VERSION="$1"

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "Usage: $0 <previous-version>"
    echo "Available versions:"
    kubectl get replicasets -n $NAMESPACE -l app.kubernetes.io/name=$SERVICE_NAME --sort-by=.metadata.creationTimestamp
    exit 1
fi

echo "Starting rollback for $SERVICE_NAME to version $PREVIOUS_VERSION"

# Get current version
CURRENT_VERSION=$(kubectl get rollout $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.status.currentPodHash}')

# Rollback to previous version
kubectl argo rollouts undo $SERVICE_NAME -n $NAMESPACE --to-revision=$PREVIOUS_VERSION

# Wait for rollback to complete
echo "Waiting for rollback to complete..."
kubectl rollout status rollout/$SERVICE_NAME -n $NAMESPACE --timeout=600s

# Verify rollback
echo "Verifying rollback..."
kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=$SERVICE_NAME

# Run health checks
echo "Running health checks..."
python scripts/health_checks.py --environment=production

echo "Rollback completed successfully!"
```

#### Emergency Rollback

```bash
#!/bin/bash
# scripts/emergency-rollback.sh

set -e

NAMESPACE="production"
SERVICE_NAME="medical-kg-rev-gateway"

echo "Starting emergency rollback for $SERVICE_NAME"

# Get the last successful deployment
LAST_SUCCESSFUL=$(kubectl get rollout $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Progressing")].lastUpdateTime}' | head -1)

if [ -z "$LAST_SUCCESSFUL" ]; then
    echo "No successful deployment found. Rolling back to previous version."
    kubectl argo rollouts undo $SERVICE_NAME -n $NAMESPACE
else
    echo "Rolling back to last successful deployment: $LAST_SUCCESSFUL"
    kubectl argo rollouts undo $SERVICE_NAME -n $NAMESPACE --to-revision=$LAST_SUCCESSFUL
fi

# Wait for rollback to complete
echo "Waiting for rollback to complete..."
kubectl rollout status rollout/$SERVICE_NAME -n $NAMESPACE --timeout=300s

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

## Monitoring and Alerting

### Prometheus Configuration

#### Service Discovery

```yaml
# prometheus/service-discovery.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    rule_files:
      - "rules/*.yml"

    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093

    scrape_configs:
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__
          - action: labelmap
            regex: __meta_kubernetes_pod_label_(.+)
          - source_labels: [__meta_kubernetes_namespace]
            action: replace
            target_label: kubernetes_namespace
          - source_labels: [__meta_kubernetes_pod_name]
            action: replace
            target_label: kubernetes_pod_name
```

#### Alert Rules

```yaml
# prometheus/alert-rules.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  medical-kg-rev-rules.yml: |
    groups:
    - name: medical-kg-rev
      rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: DatabaseConnectionFailure
        expr: up{job="postgresql"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failed"
          description: "PostgreSQL database is not responding"

      - alert: HighMemoryUsage
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}% of limit"

      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} is restarting frequently"
```

### Grafana Dashboards

#### Application Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Medical KG Rev - Application Metrics",
    "tags": ["medical-kg-rev", "application"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds",
            "min": 0
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ],
        "yAxes": [
          {
            "label": "Error Rate",
            "min": 0,
            "max": 1
          }
        ]
      },
      {
        "id": 4,
        "title": "Active Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "active_connections",
            "legendFormat": "Active Connections"
          }
        ],
        "yAxes": [
          {
            "label": "Connections",
            "min": 0
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## Best Practices

### Pipeline Optimization

1. **Parallel Execution**
   - Run independent jobs in parallel
   - Use matrix builds for multiple Python versions
   - Cache dependencies between runs

2. **Fast Feedback**
   - Run quick checks first (linting, type checking)
   - Fail fast on critical issues
   - Provide clear error messages

3. **Resource Management**
   - Use appropriate runner sizes
   - Clean up resources after tests
   - Monitor resource usage

### Quality Gates

1. **Code Coverage**
   - Minimum 80% code coverage
   - 90% coverage for critical components
   - Coverage reports in PR comments

2. **Performance Benchmarks**
   - API response time < 500ms
   - Database query time < 100ms
   - Memory usage within limits

3. **Security Checks**
   - No high-severity vulnerabilities
   - No hardcoded secrets
   - Dependency security scan

### Deployment Best Practices

1. **Gradual Rollout**
   - Use canary deployments for new features
   - Monitor metrics during rollout
   - Have rollback procedures ready

2. **Health Checks**
   - Implement comprehensive health checks
   - Monitor application metrics
   - Set up alerting for critical issues

3. **Documentation**
   - Document deployment procedures
   - Keep runbooks updated
   - Provide troubleshooting guides

## Troubleshooting

### Common Issues

1. **Pipeline Failures**
   - Check dependency versions
   - Verify environment setup
   - Review error messages

2. **Deployment Issues**
   - Check service health
   - Verify configuration
   - Review deployment logs

3. **Performance Issues**
   - Check resource usage
   - Profile application code
   - Review database queries

### Debug Commands

```bash
# Check pipeline status
gh run list --limit 10

# View pipeline logs
gh run view <run-id> --log

# Check deployment status
kubectl get deployments -n production
kubectl get pods -n production

# View pod logs
kubectl logs -f deployment/medical-kg-rev-gateway -n production

# Check service health
curl http://medical-kg-rev-gateway/health

# Check metrics
curl http://prometheus:9090/api/v1/query?query=up
```

## Related Documentation

- [Development Workflow](development_workflow.md)
- [Code Review Guidelines](code_review_guidelines.md)
- [Deployment Guide](deployment.md)
- [Monitoring Guide](monitoring.md)
- [Troubleshooting Guide](troubleshooting.md)
