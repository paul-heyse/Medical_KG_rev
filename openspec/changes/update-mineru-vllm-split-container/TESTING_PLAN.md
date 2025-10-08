# Comprehensive Testing Plan: MinerU vLLM Split-Container Architecture

**Change ID**: `update-mineru-vllm-split-container`
**Version**: 1.0
**Date**: 2025-10-08
**Status**: Ready for Execution

---

## Table of Contents

1. [Test Strategy Overview](#test-strategy-overview)
2. [Test Data Requirements](#test-data-requirements)
3. [Test Environment Specifications](#test-environment-specifications)
4. [Performance Baseline Establishment](#performance-baseline-establishment)
5. [Unit Testing](#unit-testing)
6. [Integration Testing](#integration-testing)
7. [End-to-End Testing](#end-to-end-testing)
8. [Performance Testing](#performance-testing)
9. [Chaos Engineering](#chaos-engineering)
10. [Security Testing](#security-testing)
11. [Quality Regression Testing](#quality-regression-testing)
12. [User Acceptance Testing](#user-acceptance-testing)
13. [Test Execution Schedule](#test-execution-schedule)
14. [Test Metrics and Reporting](#test-metrics-and-reporting)

---

## Test Strategy Overview

###

 Test Pyramid

```
                 ┌─────────────┐
                 │   Manual    │  5% (UAT, Exploratory)
                 │   Testing   │
                 └─────────────┘
            ┌──────────────────────┐
            │   End-to-End Tests   │  15% (100 PDFs, Critical Paths)
            └──────────────────────┘
       ┌────────────────────────────────┐
       │    Integration Tests            │  30% (Worker-vLLM, System)
       └────────────────────────────────┘
  ┌──────────────────────────────────────────┐
  │         Unit Tests                        │  50% (HTTP Client, Circuit Breaker)
  └──────────────────────────────────────────┘
```

### Test Objectives

1. **Functional Correctness**: Verify split-container architecture works as designed
2. **Performance**: Confirm 20-30% throughput improvement, <5s worker startup
3. **Resilience**: Validate circuit breaker, retry logic, graceful degradation
4. **Quality Parity**: Ensure PDF output quality ≥95% match with baseline
5. **Security**: Confirm network isolation, authentication, audit logging
6. **Operational Readiness**: Validate monitoring, alerting, runbooks

### Test Success Criteria

| Category | Success Criteria |
|----------|------------------|
| **Unit Tests** | 100% pass rate, ≥90% code coverage for new code |
| **Integration Tests** | 100% pass rate, all worker-vLLM scenarios validated |
| **E2E Tests** | ≥98% pass rate (100 PDFs), ≥95% quality match |
| **Performance Tests** | Throughput +20% min, P95 latency ≤baseline +10% |
| **Chaos Tests** | All recovery scenarios validated, no data loss |
| **Security Tests** | No high/critical vulnerabilities, all policies enforced |
| **UAT** | All acceptance criteria met, stakeholder sign-off |

---

## Test Data Requirements

### PDF Test Corpus

#### Clinical Trials (NCT Documents)

- **Count**: 50 PDFs
- **Source**: ClinicalTrials.gov sample dataset
- **Characteristics**:
  - Length: 10-50 pages
  - Tables: 2-10 per document (trial design, demographics, outcomes)
  - Figures: 1-5 per document (CONSORT diagrams, survival curves)
  - Text complexity: High (medical terminology, structured sections)
- **Purpose**: Primary test data (most common use case)

#### Drug Labels (SPL Format)

- **Count**: 20 PDFs
- **Source**: OpenFDA sample drug labels
- **Characteristics**:
  - Length: 5-30 pages
  - Tables: 5-15 per document (dosing, adverse events, interactions)
  - Figures: 0-2 per document (chemical structures, rarely)
  - Text structure: Highly structured (Indications, Dosage, Warnings)
- **Purpose**: Test table extraction accuracy

#### Biomedical Research Papers (PMC)

- **Count**: 20 PDFs
- **Source**: PubMed Central open-access sample
- **Characteristics**:
  - Length: 8-15 pages
  - Tables: 2-5 per document (results, demographics)
  - Figures: 3-8 per document (plots, diagrams, microscopy)
  - Text complexity: High (academic writing, equations, references)
- **Purpose**: Test figure extraction and layout handling

#### Edge Cases

- **Count**: 10 PDFs
- **Types**:
  - Scanned PDFs (OCR required): 3
  - Multi-column layouts: 2
  - Non-English (Spanish, German): 2
  - Large files (>50 pages): 2
  - Poor quality scans (low DPI): 1
- **Purpose**: Stress test edge case handling

#### Performance Test Set

- **Count**: 200 PDFs
- **Composition**: Mix of above categories
- **Purpose**: Throughput and load testing

### Test Data Preparation

#### Data Acquisition

```bash
# Download clinical trial PDFs
python scripts/test_data/download_clinical_trials.py --count 50 --output tests/data/clinical_trials/

# Download drug labels
python scripts/test_data/download_drug_labels.py --count 20 --output tests/data/drug_labels/

# Download PMC papers
python scripts/test_data/download_pmc_papers.py --count 20 --output tests/data/pmc_papers/

# Prepare edge cases
python scripts/test_data/prepare_edge_cases.py --output tests/data/edge_cases/
```

#### Data Anonymization (HIPAA Compliance)

- **Requirement**: Remove PII from test PDFs if using real documents
- **Tool**: Custom anonymization script
- **Process**:

  ```bash
  # Redact patient names, DOB, MRN from test PDFs
  python scripts/test_data/anonymize_pdfs.py \
    --input tests/data/clinical_trials/ \
    --output tests/data/clinical_trials_anon/ \
    --redact-patterns patterns/hipaa_pii.yaml
  ```

- **Validation**: Manual review of 10% sample by compliance officer

#### Baseline Output Generation

```bash
# Generate baseline outputs using current monolithic implementation
python scripts/test_data/generate_baseline_outputs.py \
  --input tests/data/ \
  --output tests/data/baseline_outputs/ \
  --backend monolithic

# Store baseline metadata
python scripts/test_data/extract_baseline_metadata.py \
  --input tests/data/baseline_outputs/ \
  --output tests/data/baseline_metadata.json
```

### Test Data Storage

- **Location**: MinIO bucket `test-data` or local NFS
- **Structure**:

  ```
  tests/data/
  ├── clinical_trials/     # 50 PDFs
  ├── drug_labels/         # 20 PDFs
  ├── pmc_papers/          # 20 PDFs
  ├── edge_cases/          # 10 PDFs
  ├── performance/         # 200 PDFs
  ├── baseline_outputs/    # Baseline processed results
  └── baseline_metadata.json  # Baseline metrics
  ```

---

## Test Environment Specifications

### Environment 1: Local Development

**Purpose**: Unit tests, rapid iteration

**Specifications**:

- **OS**: Ubuntu 24.04 or macOS 14+
- **Python**: 3.12+
- **GPU**: Not required (tests use mocked vLLM responses)
- **Docker**: Required for integration tests with Docker Compose
- **Resources**: 16GB RAM, 4 CPU cores

**Setup**:

```bash
# Clone repo
git clone https://github.com/your-org/Medical_KG_rev.git
cd Medical_KG_rev

# Install dependencies
poetry install

# Start test dependencies (Redis, Kafka)
docker-compose -f docker-compose.test.yml up -d

# Run unit tests
pytest tests/services/mineru/test_vllm_client.py -v
```

### Environment 2: Staging (Full Stack)

**Purpose**: Integration, E2E, performance, chaos testing

**Specifications**:

- **Infrastructure**: Kubernetes cluster (GKE/EKS)
- **Nodes**:
  - **GPU Node**: 1x n1-standard-8 with NVIDIA T4 or RTX 4090 (32GB VRAM)
  - **CPU Nodes**: 3x n1-standard-4
- **Services**:
  - vLLM server (1 pod, GPU node)
  - MinerU workers (8 pods, CPU nodes)
  - Kafka (3 brokers)
  - Neo4j (1 instance)
  - OpenSearch (3 nodes)
  - Redis (1 instance)
  - MinIO (1 instance)
  - Prometheus + Grafana
- **Namespace**: `medical-kg-staging`

**Setup**:

```bash
# Deploy staging environment
kubectl apply -f ops/k8s/overlays/staging/

# Wait for all pods ready
kubectl wait --for=condition=ready pod --all -n medical-kg-staging --timeout=10m

# Run smoke test
python scripts/test_staging_deployment.py
```

### Environment 3: Pre-Production (Production-like)

**Purpose**: Final validation before production, UAT

**Specifications**:

- **Infrastructure**: Same as production
- **GPU**: RTX 5090 (32GB VRAM)
- **Data**: Anonymized production data subset (10%)
- **Configuration**: Identical to production
- **Namespace**: `medical-kg-preprod`

**Access Control**: Restricted (requires approval for access)

---

## Performance Baseline Establishment

### Baseline Metrics (Current Monolithic Implementation)

#### Throughput Baseline

**Procedure**:

1. Deploy current monolithic implementation to staging
2. Process 200 PDFs over 4 hours
3. Calculate PDFs/hour

**Baseline Measurements** (to be collected):

```bash
# Run baseline throughput test
python tests/performance/baseline_throughput.py \
  --pdfs tests/data/performance/ \
  --duration 4h \
  --output baseline_throughput.json

# Expected results (example):
{
  "total_pdfs": 200,
  "duration_hours": 4.0,
  "throughput_per_hour": 50,
  "worker_count": 4,
  "gpu_utilization_avg": 0.65
}
```

#### Latency Baseline

**Procedure**:

1. Process 100 PDFs individually
2. Measure processing time for each

**Baseline Measurements**:

```bash
# Run baseline latency test
python tests/performance/baseline_latency.py \
  --pdfs tests/data/clinical_trials/ \
  --output baseline_latency.json

# Expected results (example):
{
  "count": 100,
  "mean_seconds": 45.2,
  "p50_seconds": 42.0,
  "p95_seconds": 78.5,
  "p99_seconds": 95.3
}
```

#### Quality Baseline

**Procedure**:

1. Process 50 PDFs
2. Extract metadata: page count, table count, figure count, block count

**Baseline Measurements**:

```bash
# Run baseline quality test
python tests/performance/baseline_quality.py \
  --pdfs tests/data/clinical_trials/ \
  --output baseline_quality.json

# Expected results (example):
{
  "pdf_count": 50,
  "avg_tables_per_pdf": 5.2,
  "avg_figures_per_pdf": 3.8,
  "avg_blocks_per_pdf": 450,
  "extraction_accuracy": 0.92  # Manual validation on sample
}
```

### Baseline Validation

- [ ] Throughput baseline collected and documented
- [ ] Latency baseline collected (P50, P95, P99)
- [ ] Quality baseline collected (tables, figures, blocks)
- [ ] GPU utilization baseline collected
- [ ] Worker startup time baseline collected
- [ ] Error rate baseline collected (should be <1%)

**Baseline Report**: Store in `tests/data/baseline_report.json` for automated comparison

---

## Unit Testing

### Test Scope

**Components Under Test**:

1. `VLLMClient` HTTP client (connection pooling, retries)
2. `CircuitBreaker` state machine
3. Configuration parsing
4. Error handling and exceptions

**Test Count**: 25+ unit tests

**Coverage Target**: ≥90% for new code

### Key Test Cases

#### VLLMClient Tests

```python
# tests/services/mineru/test_vllm_client.py

def test_vllm_client_initialization():
    """Test client creates connection pool with correct settings."""
    client = VLLMClient(base_url="http://vllm:8000")
    assert client.client.limits.max_connections == 10
    assert client.client.limits.max_keepalive_connections == 5

@pytest.mark.asyncio
async def test_chat_completion_success(mock_vllm_server):
    """Test successful chat completion request."""
    client = VLLMClient(base_url=mock_vllm_server)
    response = await client.chat_completion(
        messages=[{"role": "user", "content": "Test"}]
    )
    assert "choices" in response
    assert response["choices"][0]["message"]["content"]

@pytest.mark.asyncio
async def test_chat_completion_timeout():
    """Test timeout handling with retry."""
    client = VLLMClient(base_url="http://slow-server:8000")
    with pytest.raises(httpx.TimeoutException):
        await client.chat_completion(messages=[...])
    # Verify 3 retry attempts were made
    assert client.retry_count == 3

@pytest.mark.asyncio
async def test_chat_completion_http_error():
    """Test HTTP error handling (no retry on 4xx)."""
    client = VLLMClient(base_url=mock_vllm_server_400)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat_completion(messages=[...])
    # Verify no retries on 4xx
    assert client.retry_count == 0
```

#### CircuitBreaker Tests

```python
# tests/services/mineru/test_circuit_breaker.py

def test_circuit_breaker_opens_after_failures():
    """Test circuit opens after threshold failures."""
    cb = CircuitBreaker(failure_threshold=5)

    # Record 5 failures
    for _ in range(5):
        cb.record_failure()

    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() == False

def test_circuit_breaker_half_open_recovery():
    """Test circuit transitions to half-open after timeout."""
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=1.0)

    # Open circuit
    for _ in range(5):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Wait for recovery timeout
    time.sleep(1.1)
    assert cb.can_execute() == True
    assert cb.state == CircuitState.HALF_OPEN

def test_circuit_breaker_closes_after_successes():
    """Test circuit closes after threshold successes in half-open."""
    cb = CircuitBreaker(success_threshold=2)
    cb.state = CircuitState.HALF_OPEN

    # Record 2 successes
    cb.record_success()
    cb.record_success()

    assert cb.state == CircuitState.CLOSED
```

### Running Unit Tests

```bash
# Run all unit tests
pytest tests/services/mineru/ -v --cov=src/Medical_KG_rev/services/mineru --cov-report=html

# Run specific test file
pytest tests/services/mineru/test_vllm_client.py -v

# Run with coverage threshold
pytest tests/ --cov --cov-fail-under=90

# Generate HTML coverage report
pytest tests/ --cov --cov-report=html
open htmlcov/index.html
```

---

## Integration Testing

### Test Scope

**Integration Points**:

1. Worker → vLLM server HTTP communication
2. Circuit breaker integration with VLLMClient
3. Worker startup connectivity check
4. Distributed tracing propagation
5. Metrics collection

**Test Count**: 15+ integration tests

### Test Environment

**Docker Compose Setup**:

```yaml
# docker-compose.test.yml
services:
  vllm-server:
    image: vllm/vllm-openai:latest
    command: >
      python -m vllm.entrypoints.openai.api_server
        --model Qwen/Qwen2.5-VL-7B-Instruct
        --host 0.0.0.0 --port 8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  mineru-worker:
    build: .
    environment:
      - VLLM_SERVER_URL=http://vllm-server:8000
    depends_on:
      - vllm-server
```

### Key Test Cases

```python
# tests/integration/test_vllm_server_integration.py

@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_connects_to_vllm_server():
    """Test worker successfully connects to vLLM server."""
    client = VLLMClient(base_url="http://vllm-server:8000")

    # Check health endpoint
    response = await client.client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_requests_batching():
    """Test vLLM server batches concurrent requests."""
    client = VLLMClient(base_url="http://vllm-server:8000")

    # Send 4 concurrent requests
    tasks = [
        client.chat_completion(messages=[{"role": "user", "content": f"Test {i}"}])
        for i in range(4)
    ]
    results = await asyncio.gather(*tasks)

    # All requests should succeed
    assert all(r["choices"][0]["message"]["content"] for r in results)

@pytest.mark.integration
async def test_vllm_server_restart_recovery():
    """Test workers recover when vLLM server restarts."""
    client = VLLMClient(base_url="http://vllm-server:8000")

    # Stop vLLM server
    subprocess.run(["docker", "stop", "vllm-server"])

    # Requests should fail, circuit breaker opens
    for _ in range(5):
        try:
            await client.chat_completion(messages=[...])
        except Exception:
            pass

    assert client.circuit_breaker.state == CircuitState.OPEN

    # Restart vLLM server
    subprocess.run(["docker", "start", "vllm-server"])
    time.sleep(60)  # Wait for model loading

    # Circuit breaker should recover
    await asyncio.sleep(60)  # Recovery timeout
    result = await client.chat_completion(messages=[...])
    assert result["choices"][0]["message"]["content"]
    assert client.circuit_breaker.state == CircuitState.CLOSED
```

### Running Integration Tests

```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Wait for services ready
sleep 120  # vLLM model loading

# Run integration tests
pytest tests/integration/ -v -m integration

# Cleanup
docker-compose -f docker-compose.test.yml down
```

---

## End-to-End Testing

### Test Scope

**Full Pipeline**: Kafka ingestion → Worker processing → vLLM inference → Output storage

**Test Count**: 100 PDFs (automated), 20 PDFs (manual validation)

### Test Procedure

```bash
# 1. Deploy full stack to staging
kubectl apply -f ops/k8s/overlays/staging/

# 2. Run E2E test suite
python tests/e2e/test_pdf_pipeline.py \
  --pdfs tests/data/clinical_trials/ \
  --count 100 \
  --output tests/e2e/results/

# 3. Compare outputs with baseline
python tests/e2e/compare_with_baseline.py \
  --test-output tests/e2e/results/ \
  --baseline tests/data/baseline_outputs/ \
  --threshold 0.95

# 4. Generate E2E report
python tests/e2e/generate_report.py \
  --output tests/e2e/report.html
```

### Success Criteria

- **Pass Rate**: ≥98/100 PDFs (2% failure allowed for edge cases)
- **Quality Match**: ≥95% structural similarity with baseline
- **Processing Time**: <90s P95 latency
- **No Data Loss**: All 100 PDFs retrievable from storage

---

## Performance Testing

See `tests/performance/` directory and tasks.md Phase 6 for detailed load testing procedures.

**Key Tests**:

1. **Throughput Test**: 200 PDFs over 4 hours
2. **Latency Test**: P95 latency under various loads
3. **Concurrent Workers Test**: 8, 12, 16 workers
4. **vLLM Server Saturation**: Find max throughput

---

## Chaos Engineering

### Chaos Scenarios

#### Scenario 1: vLLM Server Crash

```bash
# Kill vLLM server pod
kubectl delete pod -l app=vllm-server -n medical-kg-staging

# Monitor workers
kubectl logs -f -l app=mineru-worker -n medical-kg-staging | grep circuit_breaker

# Expected: Circuit breakers open, workers queue requests, recovery when pod restarts
```

#### Scenario 2: Network Partition

```bash
# Introduce network delay
kubectl exec -it mineru-worker-xxx -n medical-kg-staging -- tc qdisc add dev eth0 root netem delay 5000ms

# Expected: Timeouts, retries, eventual circuit breaker open
```

#### Scenario 3: Resource Exhaustion

```bash
# Fill vLLM GPU memory
# Submit 100 concurrent large requests

# Expected: OOM handling, pod restart, graceful recovery
```

#### Scenario 4: Cascading Failures

```bash
# Kill Kafka broker during PDF processing
kubectl delete pod kafka-0 -n medical-kg-staging

# Expected: Workers stop consuming, Kafka recovers, workers resume
```

### Chaos Testing Tools

- **Chaos Mesh**: Kubernetes-native chaos engineering
- **Litmus**: CNCF chaos engineering framework
- **Manual**: kubectl commands for targeted scenarios

---

## Security Testing

### Security Test Checklist

- [ ] **Container Vulnerability Scan**: Trivy/Snyk on vLLM and worker images
- [ ] **Network Policy Test**: Verify only workers can access vLLM server
- [ ] **Secrets Exposure**: Verify secrets not in logs/metrics
- [ ] **API Authentication**: Test vLLM server API key (if enabled)
- [ ] **Input Validation**: Test malicious PDF inputs
- [ ] **Audit Logging**: Verify all vLLM calls logged

**Tools**: Trivy, OWASP ZAP, Kube-bench

---

## Quality Regression Testing

### Automated Quality Comparison

```python
# tests/quality/compare_outputs.py

def compare_pdf_outputs(baseline_dir, test_dir, threshold=0.95):
    """
    Compare test outputs with baseline outputs.

    Compares:
    - Number of blocks extracted
    - Number of tables extracted
    - Number of figures extracted
    - Text content similarity (fuzzy match)

    Returns similarity score 0.0-1.0
    """
    similarity_scores = []

    for pdf_name in os.listdir(baseline_dir):
        baseline = json.load(open(f"{baseline_dir}/{pdf_name}"))
        test = json.load(open(f"{test_dir}/{pdf_name}"))

        score = calculate_similarity(baseline, test)
        similarity_scores.append(score)

    avg_similarity = sum(similarity_scores) / len(similarity_scores)

    assert avg_similarity >= threshold, f"Quality regression detected: {avg_similarity} < {threshold}"

    return avg_similarity
```

---

## User Acceptance Testing

### UAT Criteria

1. **Functional Parity**: All PDF types process successfully
2. **Performance**: No noticeable slowdown from user perspective
3. **Error Handling**: Graceful error messages (no cryptic errors)
4. **Monitoring**: Dashboards show clear system health

### UAT Participants

- Product Manager
- 2-3 Data Scientists (power users)
- 1 Operations Engineer

### UAT Duration

- **Phase**: Week 4 of implementation
- **Duration**: 3 days
- **Deliverable**: UAT sign-off document

---

## Test Execution Schedule

| Week | Tests | Environment | Owner |
|------|-------|-------------|-------|
| **Week 1** | Unit tests | Local dev | Engineering |
| **Week 2** | Integration tests | Docker Compose | Engineering |
| **Week 3** | E2E + Performance | Staging | Engineering + QA |
| **Week 4** | Chaos + Security | Staging | SRE + Security |
| **Week 5** | UAT | Pre-prod | Product + Users |

---

## Test Metrics and Reporting

### Automated Test Report

```bash
# Generate comprehensive test report
python scripts/generate_test_report.py \
  --output tests/reports/test_report_$(date +%Y%m%d).html

# Report includes:
# - Test pass/fail counts
# - Code coverage
# - Performance comparison (baseline vs split-container)
# - Quality regression analysis
# - Security scan results
```

### Test Dashboard (Grafana)

Create Grafana dashboard for CI/CD test metrics:

- Test pass rate over time
- Code coverage trend
- Performance trend (throughput, latency)
- Flaky test detection

---

**Document Control**:

- **Version**: 1.0
- **Last Updated**: 2025-10-08
- **Owner**: QA Lead
- **Approvers**: Engineering Lead
