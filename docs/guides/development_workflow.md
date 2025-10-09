# Development Workflow Guide

This document provides comprehensive guidance on the development workflow for the Medical_KG_rev system, including branching strategies, code review procedures, continuous integration practices, and deployment protocols.

## Overview

The Medical_KG_rev development workflow follows GitFlow principles with feature branches, pull requests, automated testing, and continuous deployment. This ensures code quality, collaboration efficiency, and system reliability.

## Development Environment Setup

### Prerequisites

Before starting development, ensure you have the following installed:

- **Python 3.11+**: Core runtime environment
- **Docker & Docker Compose**: Containerized services
- **Git**: Version control
- **Node.js 18+**: For frontend development (if applicable)
- **PostgreSQL 14+**: Database (via Docker)
- **Neo4j 5+**: Graph database (via Docker)
- **Redis 7+**: Cache (via Docker)

### Initial Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/Medical_KG_rev.git
   cd Medical_KG_rev
   ```

2. **Set up Python environment**

   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Configure environment variables**

   ```bash
   # Copy environment template
   cp .env.example .env

   # Edit environment variables
   nano .env
   ```

4. **Start development services**

   ```bash
   # Start all services
   docker-compose up -d

   # Verify services are running
   docker-compose ps
   ```

5. **Run database migrations**

   ```bash
   # Apply migrations
   python -m Medical_KG_rev.storage.migrations migrate

   # Verify database setup
   python -m Medical_KG_rev.storage.migrations status
   ```

6. **Run initial tests**

   ```bash
   # Run all tests
   pytest

   # Run specific test categories
   pytest tests/unit/
   pytest tests/integration/
   pytest tests/contract/
   ```

## Branching Strategy

### Branch Types

#### Main Branches

- **`main`**: Production-ready code
- **`develop`**: Integration branch for features
- **`release/*`**: Release preparation branches
- **`hotfix/*`**: Critical production fixes

#### Feature Branches

- **`feature/*`**: New features and enhancements
- **`bugfix/*`**: Bug fixes
- **`refactor/*`**: Code refactoring
- **`docs/*`**: Documentation updates

### Branch Naming Convention

```bash
# Feature branches
feature/add-user-authentication
feature/implement-graphql-api
feature/add-biomedical-adapters

# Bug fix branches
bugfix/fix-database-connection-pool
bugfix/resolve-memory-leak-in-embeddings

# Refactoring branches
refactor/simplify-gateway-routing
refactor/optimize-vector-search

# Documentation branches
docs/update-api-documentation
docs/add-deployment-guide

# Release branches
release/v1.2.0
release/v2.0.0

# Hotfix branches
hotfix/critical-security-patch
hotfix/fix-production-crash
```

### Branch Workflow

#### Creating Feature Branches

1. **Start from develop**

   ```bash
   git checkout develop
   git pull origin develop
   ```

2. **Create feature branch**

   ```bash
   git checkout -b feature/add-new-adapter
   ```

3. **Push branch to remote**

   ```bash
   git push -u origin feature/add-new-adapter
   ```

#### Working on Features

1. **Make changes and commit**

   ```bash
   # Make changes
   git add .
   git commit -m "feat: add OpenAlex adapter implementation"

   # Push changes
   git push origin feature/add-new-adapter
   ```

2. **Keep branch updated**

   ```bash
   # Fetch latest changes
   git fetch origin

   # Rebase on develop
   git rebase origin/develop

   # Resolve conflicts if any
   git add .
   git rebase --continue
   ```

#### Completing Features

1. **Create pull request**
   - Open PR from feature branch to develop
   - Add description, tests, and documentation
   - Request code review

2. **Address feedback**

   ```bash
   # Make changes based on feedback
   git add .
   git commit -m "fix: address code review feedback"
   git push origin feature/add-new-adapter
   ```

3. **Merge and cleanup**

   ```bash
   # After PR is approved and merged
   git checkout develop
   git pull origin develop
   git branch -d feature/add-new-adapter
   git push origin --delete feature/add-new-adapter
   ```

## Code Review Process

### Pull Request Guidelines

#### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Contract tests added/updated
- [ ] Performance tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Tests pass locally
- [ ] CI/CD pipeline passes

## Related Issues
Closes #123
Fixes #456
```

#### PR Size Guidelines

- **Small PRs**: < 200 lines changed
- **Medium PRs**: 200-500 lines changed
- **Large PRs**: 500-1000 lines changed
- **Extra Large PRs**: > 1000 lines changed (require special approval)

### Code Review Checklist

#### For Authors

- [ ] **Code Quality**
  - [ ] Code follows PEP 8 style guidelines
  - [ ] Functions and classes have proper docstrings
  - [ ] Type hints are used consistently
  - [ ] No unused imports or variables
  - [ ] Error handling is implemented

- [ ] **Testing**
  - [ ] Unit tests cover new functionality
  - [ ] Integration tests verify component interactions
  - [ ] Contract tests ensure API compatibility
  - [ ] Performance tests validate non-functional requirements
  - [ ] All tests pass locally

- [ ] **Documentation**
  - [ ] Code is self-documenting
  - [ ] Docstrings follow Google style
  - [ ] README updated if needed
  - [ ] API documentation updated
  - [ ] Configuration documentation updated

- [ ] **Security**
  - [ ] No hardcoded secrets
  - [ ] Input validation implemented
  - [ ] SQL injection prevention
  - [ ] XSS prevention
  - [ ] CSRF protection

#### For Reviewers

- [ ] **Functionality**
  - [ ] Code solves the intended problem
  - [ ] Edge cases are handled
  - [ ] Error scenarios are covered
  - [ ] Performance implications considered

- [ ] **Architecture**
  - [ ] Code follows established patterns
  - [ ] Dependencies are appropriate
  - [ ] Separation of concerns maintained
  - [ ] No circular dependencies

- [ ] **Maintainability**
  - [ ] Code is readable and understandable
  - [ ] Complex logic is commented
  - [ ] Configuration is externalized
  - [ ] Logging is appropriate

### Review Process

#### Review Assignment

1. **Automatic Assignment**
   - Code owners are automatically assigned
   - At least 2 reviewers required for changes > 100 lines
   - At least 1 reviewer required for changes < 100 lines

2. **Manual Assignment**
   - Assign domain experts for complex changes
   - Include security team for security-related changes
   - Include performance team for performance-critical changes

#### Review Timeline

- **Small PRs**: Review within 24 hours
- **Medium PRs**: Review within 48 hours
- **Large PRs**: Review within 72 hours
- **Critical PRs**: Review within 4 hours

#### Review Actions

- **Approve**: Code is ready to merge
- **Request Changes**: Code needs modifications
- **Comment**: General feedback or questions
- **Block**: Code cannot be merged

## Continuous Integration

### CI Pipeline Stages

#### 1. Code Quality Checks

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run linting
        run: |
          ruff check .
          ruff format --check .

      - name: Run type checking
        run: mypy src/

      - name: Run security checks
        run: |
          bandit -r src/
          safety check
```

#### 2. Testing Stages

```yaml
  unit-tests:
    runs-on: ubuntu-latest
    needs: code-quality
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run unit tests
        run: pytest tests/unit/ --cov=src/ --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

#### 3. Integration Testing

```yaml
  integration-tests:
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

      neo4j:
        image: neo4j:5
        env:
          NEO4J_AUTH: neo4j/password
        options: >-
          --health-cmd "cypher-shell -u neo4j -p password 'RETURN 1'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run integration tests
        run: pytest tests/integration/
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          NEO4J_URI: bolt://localhost:7687
          NEO4J_PASSWORD: password
          REDIS_URL: redis://localhost:6379/0
```

#### 4. Contract Testing

```yaml
  contract-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run contract tests
        run: |
          pytest tests/contract/
          schemathesis run docs/openapi.yaml --base-url http://localhost:8000
```

#### 5. Performance Testing

```yaml
  performance-tests:
    runs-on: ubuntu-latest
    needs: contract-tests
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Start services
        run: docker-compose up -d

      - name: Wait for services
        run: sleep 30

      - name: Run performance tests
        run: |
          k6 run tests/performance/load_test.js
          k6 run tests/performance/stress_test.js
```

### CI Best Practices

#### Pipeline Optimization

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

#### Quality Gates

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

## Deployment Process

### Deployment Environments

#### Development Environment

- **Purpose**: Local development and testing
- **URL**: `http://localhost:8000`
- **Database**: Local PostgreSQL/Neo4j/Redis
- **Deployment**: Manual via `docker-compose up`

#### Staging Environment

- **Purpose**: Pre-production testing
- **URL**: `https://staging.medical-kg-rev.com`
- **Database**: Staging PostgreSQL/Neo4j/Redis
- **Deployment**: Automated via CI/CD on `develop` branch

#### Production Environment

- **Purpose**: Live system
- **URL**: `https://api.medical-kg-rev.com`
- **Database**: Production PostgreSQL/Neo4j/Redis
- **Deployment**: Automated via CI/CD on `main` branch

### Deployment Strategies

#### Blue-Green Deployment

1. **Preparation**

   ```bash
   # Deploy to green environment
   kubectl apply -f k8s/green/

   # Wait for deployment to be ready
   kubectl wait --for=condition=available deployment/medical-kg-rev-green
   ```

2. **Switch Traffic**

   ```bash
   # Update service to point to green
   kubectl patch service medical-kg-rev -p '{"spec":{"selector":{"version":"green"}}}'

   # Verify traffic is flowing
   kubectl get pods -l version=green
   ```

3. **Cleanup**

   ```bash
   # Remove blue environment
   kubectl delete -f k8s/blue/
   ```

#### Canary Deployment

1. **Deploy Canary**

   ```bash
   # Deploy canary version
   kubectl apply -f k8s/canary/

   # Route 10% traffic to canary
   kubectl apply -f k8s/canary-traffic-split.yaml
   ```

2. **Monitor Metrics**

   ```bash
   # Monitor canary metrics
   kubectl get pods -l version=canary

   # Check Prometheus metrics
   curl http://prometheus:9090/api/v1/query?query=error_rate
   ```

3. **Gradual Rollout**

   ```bash
   # Increase traffic to 50%
   kubectl patch service medical-kg-rev -p '{"spec":{"traffic":{"canary":{"weight":50}}}}'

   # Full rollout
   kubectl patch service medical-kg-rev -p '{"spec":{"traffic":{"canary":{"weight":100}}}}'
   ```

### Rollback Procedures

#### Automatic Rollback

```yaml
# k8s/rollback-policy.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: medical-kg-rev
spec:
  strategy:
    canary:
      steps:
      - setWeight: 20
      - pause: {duration: 10m}
      - setWeight: 40
      - pause: {duration: 10m}
      - setWeight: 60
      - pause: {duration: 10m}
      - setWeight: 80
      - pause: {duration: 10m}
      rollbackPolicy:
        failureThreshold: 3
        failureType: "ErrorRate"
        errorRateThreshold: 0.1
```

#### Manual Rollback

1. **Identify Issue**

   ```bash
   # Check deployment status
   kubectl get deployments

   # Check pod logs
   kubectl logs -l app=medical-kg-rev --tail=100
   ```

2. **Rollback to Previous Version**

   ```bash
   # Rollback deployment
   kubectl rollout undo deployment/medical-kg-rev

   # Verify rollback
   kubectl rollout status deployment/medical-kg-rev
   ```

3. **Verify System Health**

   ```bash
   # Check health endpoints
   curl http://medical-kg-rev/health

   # Check metrics
   curl http://prometheus:9090/api/v1/query?query=up
   ```

## Monitoring and Observability

### Application Monitoring

#### Health Checks

```python
# Example: Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    checks = {
        "database": await check_database_health(),
        "neo4j": await check_neo4j_health(),
        "redis": await check_redis_health(),
        "vector_store": await check_vector_store_health(),
    }

    overall_status = "healthy" if all(checks.values()) else "unhealthy"

    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

#### Metrics Collection

```python
# Example: Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])

# Business metrics
DOCUMENTS_PROCESSED = Counter('documents_processed_total', 'Total documents processed', ['source', 'status'])
EMBEDDINGS_GENERATED = Counter('embeddings_generated_total', 'Total embeddings generated', ['model'])

# System metrics
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active database connections')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
```

### Logging Strategy

#### Structured Logging

```python
# Example: Structured logging
import structlog

logger = structlog.get_logger()

# Request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        "request_processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time,
        user_id=request.state.user_id if hasattr(request.state, 'user_id') else None
    )

    return response
```

#### Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about program execution
- **WARNING**: Something unexpected happened
- **ERROR**: A serious problem occurred
- **CRITICAL**: A very serious error occurred

### Alerting

#### Alert Rules

```yaml
# prometheus/alerts.yml
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
```

## Best Practices

### Code Quality

1. **Follow PEP 8**: Use consistent code formatting
2. **Type Hints**: Use type hints for all functions
3. **Docstrings**: Document all public functions and classes
4. **Error Handling**: Implement proper error handling
5. **Testing**: Write comprehensive tests

### Collaboration

1. **Communication**: Use clear, concise commit messages
2. **Documentation**: Keep documentation up to date
3. **Code Reviews**: Provide constructive feedback
4. **Knowledge Sharing**: Share knowledge through documentation
5. **Mentoring**: Help junior developers grow

### Security

1. **Secrets Management**: Never commit secrets to version control
2. **Input Validation**: Validate all inputs
3. **Dependency Management**: Keep dependencies updated
4. **Access Control**: Implement proper access controls
5. **Audit Logging**: Log security-relevant events

### Performance

1. **Profiling**: Profile code for performance bottlenecks
2. **Caching**: Implement appropriate caching strategies
3. **Database Optimization**: Optimize database queries
4. **Resource Management**: Manage resources efficiently
5. **Monitoring**: Monitor performance metrics

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check dependency versions
   - Verify environment setup
   - Review error messages

2. **Test Failures**
   - Check test data setup
   - Verify test environment
   - Review test assertions

3. **Deployment Issues**
   - Check service health
   - Verify configuration
   - Review deployment logs

4. **Performance Issues**
   - Check resource usage
   - Profile application code
   - Review database queries

### Debug Commands

```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs -f service_name

# Check database connectivity
python -c "from Medical_KG_rev.storage.database import DatabaseManager; db = DatabaseManager(); print(db.test_connection())"

# Run specific tests
pytest tests/unit/test_specific.py -v

# Check code coverage
pytest --cov=src/ --cov-report=html

# Run linting
ruff check src/
ruff format src/

# Check type hints
mypy src/
```

## Related Documentation

- [Code Review Guidelines](code_review_guidelines.md)
- [CI/CD Pipeline](ci_cd_pipeline.md)
- [Environment Setup](environment_setup.md)
- [Deployment Guide](deployment.md)
- [Troubleshooting Guide](troubleshooting.md)
