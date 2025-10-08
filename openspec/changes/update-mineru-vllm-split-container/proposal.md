# Change Proposal: Update MinerU to vLLM Split-Container Architecture

## Why

The current MinerU GPU service (archived in `openspec/changes/archive/add-mineru-gpu-cli-integration`) uses a **monolithic single-container architecture** where MinerU workers invoke the vision-language model (VLM) in-process via the `vlm-vllm-engine` backend. While functional, this design has critical operational and performance limitations:

### Current Architecture Limitations

1. **Model Loading Redundancy**: Each MinerU worker process loads the VLM weights independently into GPU memory, causing:
   - **Duplicated VRAM consumption**: 4 workers × full model size = wasted memory
   - **Slow worker startup**: 30-60s to load model weights per worker
   - **GPU memory fragmentation**: Multiple concurrent loads compete for VRAM

2. **No Cross-Worker Batching**: Workers process PDFs independently with isolated VLM inference:
   - **Missed batching opportunities**: Multiple concurrent VLM requests not coalesced
   - **Suboptimal GPU utilization**: GPU alternates between workers instead of continuous processing
   - **Higher per-request latency**: No continuous batching benefits

3. **Tight Coupling and Deployment Friction**:
   - **Dependency conflicts**: MinerU Python environment tightly coupled with vLLM/PyTorch versions
   - **All-or-nothing upgrades**: Cannot upgrade vLLM inference engine without rebuilding MinerU container
   - **No independent scaling**: Cannot scale inference capacity separately from worker capacity
   - **Blast radius**: vLLM crashes or OOM errors take down entire MinerU worker

4. **Limited Production Readiness**:
   - **No standardized serving**: Cannot integrate with KServe, Ray Serve, or model serving platforms
   - **Poor observability**: Inference metrics mixed with PDF processing metrics
   - **Difficult multi-model support**: Running different VLM versions requires separate MinerU deployments

### Industry Best Practice: Split-Container Pattern

The **vLLM project explicitly documents and recommends** a split-container client/server architecture for production deployments:

- **vLLM OpenAI-compatible server** ([vLLM docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)): Dedicated inference service exposing HTTP API
- **MinerU HTTP client mode** ([MinerU docs](https://opendatalab.github.io/MinerU/usage/quick_usage/)): `mineru -b vlm-http-client -u http://vllm-server:8000`
- **Hot model sharing** ([vLLM docs](https://docs.vllm.ai/en/stable/index.html)): Single model instance serves multiple clients
- **Continuous batching** ([vLLM docs](https://docs.vllm.ai/en/stable/index.html)): Coalesce concurrent requests for higher throughput

### Quantified Benefits

Based on vLLM and MinerU documentation and production deployments:

| Metric | Current (In-Process) | Proposed (Split-Container) | Improvement |
|--------|---------------------|---------------------------|-------------|
| **VRAM per worker** | 7GB (full model) | 0.5GB (client only) | **14x reduction** |
| **Model load time** | 30-60s per worker | 0s (hot model) | **Instant startup** |
| **Throughput (VLM)** | 4 isolated engines | Continuous batching | **20-30% higher** |
| **Worker startup** | 60s (model load) | 5s (no model) | **12x faster** |
| **Deployment flexibility** | Monolithic | Independent services | **Decoupled** |

### Problem Statement

The current monolithic MinerU architecture wastes GPU resources, prevents cross-worker batching optimizations, and creates operational friction. **We need to adopt the vLLM-recommended split-container pattern** to achieve production-grade efficiency, observability, and operational hygiene.

---

## What Changes

### Core Architectural Changes

#### 1. **Split into Two Independent Containers**

**Container A: vLLM Server (Dedicated VLM Inference)**

- Base image: `vllm/vllm-openai:latest` (official vLLM Docker image)
- Model: `Qwen/Qwen2.5-VL-7B-Instruct` (vision-language model for document understanding)
- Endpoint: OpenAI-compatible HTTP API on port `:8000`
- Resource allocation: 1 GPU (RTX 5090), 16-24GB VRAM for model + KV cache
- Purpose: Centralized VLM inference service with continuous batching

**Container B: MinerU Workers (PDF Processing)**

- Base image: Custom MinerU image (from official Dockerfile or slim Python base)
- Backend: `vlm-http-client` pointing to vLLM server
- Resource allocation: CPU-bound (no GPU required), 4-8 workers per host
- Purpose: PDF parsing, layout analysis, calling vLLM via HTTP for VLM tasks

#### 2. **Update MinerU Configuration to HTTP Client Backend**

**Before** (current in-process):

```bash
mineru -p /workspace/in -o /workspace/out \
  -b vlm-vllm-engine \
  --gpu-memory-utilization 0.90
```

**After** (HTTP client):

```bash
mineru -p /workspace/in -o /workspace/out \
  -b vlm-http-client \
  -u http://vllm-server:8000
```

Key changes:

- Replace `-b vlm-vllm-engine` with `-b vlm-http-client`
- Add `-u http://vllm-server:8000` to specify vLLM server URL
- Remove GPU-specific flags (handled by server)
- Workers become stateless HTTP clients

#### 3. **Implement vLLM Server Deployment**

**Docker Compose Service**:

```yaml
services:
  vllm-server:
    image: vllm/vllm-openai:latest
    command: >
      python -m vllm.entrypoints.openai.api_server
        --model Qwen/Qwen2.5-VL-7B-Instruct
        --host 0.0.0.0
        --port 8000
        --gpu-memory-utilization 0.92
        --max-model-len 32768
        --tensor-parallel-size 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
```

**Kubernetes Deployment**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: medical-kg
spec:
  replicas: 1  # Single server per GPU
  selector:
    matchLabels:
      app: vllm-server
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-rtx-5090
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
        - python
        - -m
        - vllm.entrypoints.openai.api_server
        - --model=Qwen/Qwen2.5-VL-7B-Instruct
        - --host=0.0.0.0
        - --port=8000
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: 32Gi
          limits:
            nvidia.com/gpu: 1
            memory: 48Gi
        ports:
        - containerPort: 8000
          name: http
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
```

#### 4. **Update MinerU Worker Pool Architecture**

**Key Changes**:

- **Remove GPU requirements**: Workers no longer need `nvidia.com/gpu` resource requests
- **Increase worker count**: Scale from 4 to 8-12 workers (CPU-bound, not GPU-bound)
- **Stateless design**: Workers hold no GPU state, pure HTTP clients
- **Faster startup**: No model loading, < 5s to ready state
- **Independent scaling**: Scale workers and vLLM server separately

**Worker Deployment**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mineru-workers
  namespace: medical-kg
spec:
  replicas: 8  # Scale independently of GPU
  selector:
    matchLabels:
      app: mineru-worker
  template:
    spec:
      containers:
      - name: mineru
        image: ghcr.io/your-org/mineru-client:latest
        env:
        - name: VLLM_SERVER_URL
          value: "http://vllm-server:8000"
        - name: MINERU_BACKEND
          value: "vlm-http-client"
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka:9092"
        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
          limits:
            cpu: 4000m
            memory: 8Gi
```

#### 5. **Implement Service Communication Layer**

**Components**:

- **Load balancer**: Nginx or Kubernetes Service for vLLM server (round-robin, health checks)
- **Connection pooling**: HTTP client connection pool in workers (persistent connections)
- **Retry logic**: Exponential backoff for transient vLLM server failures
- **Circuit breaker**: Fail-fast when vLLM server unavailable (prevent cascade failures)

**Worker HTTP Client**:

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class VLLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(300.0),  # 5 minutes for large documents
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    async def chat_completion(self, messages: list, images: list) -> dict:
        """Call vLLM OpenAI-compatible chat completion endpoint."""
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "messages": messages,
                "images": images,  # Base64-encoded or URLs
                "max_tokens": 4096,
                "temperature": 0.0
            }
        )
        response.raise_for_status()
        return response.json()
```

#### 6. **Update Configuration Management**

**New Configuration Section** (`config/mineru.yaml`):

```yaml
mineru:
  deployment_mode: "split-container"  # NEW: was "monolithic"

  vllm_server:
    enabled: true
    base_url: "http://vllm-server:8000"  # Kubernetes Service DNS
    model: "Qwen/Qwen2.5-VL-7B-Instruct"
    gpu_memory_utilization: 0.92
    max_model_len: 32768
    tensor_parallel_size: 1
    health_check_interval_seconds: 30

  workers:
    count: 8  # Increased from 4 (no longer GPU-bound)
    backend: "vlm-http-client"  # Was "vlm-vllm-engine"
    cpu_per_worker: 2  # CPU cores
    memory_per_worker_gb: 4
    batch_size: 4  # PDFs per worker batch
    timeout_seconds: 300

  http_client:
    connection_pool_size: 10
    keepalive_connections: 5
    timeout_seconds: 300
    retry_attempts: 3
    retry_backoff_multiplier: 1
    circuit_breaker:
      enabled: true
      failure_threshold: 5
      recovery_timeout_seconds: 60
```

#### 7. **Observability Enhancements**

**vLLM Server Metrics** (Prometheus):

- `vllm:num_requests_running` - Active inference requests
- `vllm:num_requests_waiting` - Queued requests
- `vllm:time_to_first_token_seconds` - Latency to first token
- `vllm:time_per_output_token_seconds` - Token generation speed
- `vllm:gpu_cache_usage_perc` - KV cache utilization
- `vllm:prompt_tokens_total` - Total input tokens processed
- `vllm:generation_tokens_total` - Total output tokens generated

**Worker Metrics** (Prometheus):

- `mineru_vllm_client_request_duration_seconds` - HTTP request latency to vLLM
- `mineru_vllm_client_failures_total` - Failed vLLM API calls
- `mineru_vllm_client_retries_total` - Retry attempts
- `mineru_vllm_circuit_breaker_state` - Circuit breaker status (0=closed, 1=open)

**Distributed Tracing** (OpenTelemetry):

- Trace ID propagation from worker → vLLM server
- Spans: `mineru.vllm.chat_completion`, `vllm.inference.forward_pass`

### Breaking Changes

- **BREAKING**: MinerU workers no longer require GPU resources (`nvidia.com/gpu` removed)
- **BREAKING**: Configuration key `mineru.workers.backend` changed from `vlm-vllm-engine` to `vlm-http-client`
- **BREAKING**: New required configuration: `mineru.vllm_server.base_url`
- **BREAKING**: Worker Docker image no longer includes vLLM dependencies (smaller image)
- **BREAKING**: Worker startup behavior changed (no model loading, immediate readiness)

---

## Supporting Documentation

This proposal is supported by comprehensive operational planning documents:

1. **[DEPLOYMENT_STRATEGY.md](./DEPLOYMENT_STRATEGY.md)**: Detailed deployment procedures, rollback plans, go/no-go criteria, and communication plan
2. **[TESTING_PLAN.md](./TESTING_PLAN.md)**: Comprehensive testing strategy with 100+ test cases, test data requirements, and quality gates
3. **[SECURITY.md](./SECURITY.md)**: Threat model (STRIDE), security controls, network isolation, and incident response
4. **[OPERATIONS.md](./OPERATIONS.md)**: SLO definitions, incident response runbooks, capacity planning, and maintenance procedures
5. **[TRAINING_AND_COMMUNICATION.md](./TRAINING_AND_COMMUNICATION.md)**: Training materials, stakeholder communication matrix, and knowledge transfer plan
6. **[COST_ANALYSIS.md](./COST_ANALYSIS.md)**: 3-year TCO analysis, ROI calculation, and budget impact
7. **[RISK_AND_READINESS.md](./RISK_AND_READINESS.md)**: Risk register, operational readiness review checklist, and quality gates
8. **[GAP_ANALYSIS.md](./GAP_ANALYSIS.md)**: Comprehensive gap analysis and remediation status

These documents provide production-ready operational planning covering deployment, security, operations, training, costs, and risk management.

---

## Impact

### Affected Specs

This change affects the following capabilities (will create spec deltas):

1. **mineru-service** - Core MinerU service specification
2. **gpu-microservices** - GPU service architecture patterns
3. **orchestration** - Worker pool management and Kafka integration

### Affected Code

**New Components**:

- `src/Medical_KG_rev/services/mineru/vllm_client.py` - HTTP client for vLLM server
- `src/Medical_KG_rev/services/mineru/circuit_breaker.py` - Circuit breaker implementation
- `ops/docker/vllm-server/Dockerfile` - vLLM server container (if custom needed)
- `ops/k8s/base/deployment-vllm-server.yaml` - Kubernetes deployment for vLLM
- `ops/k8s/base/service-vllm-server.yaml` - Kubernetes service for vLLM
- `scripts/check_vllm_health.py` - vLLM server health check script

**Modified Components**:

- `src/Medical_KG_rev/services/mineru/service.py` - Replace in-process engine with HTTP client
- `src/Medical_KG_rev/services/mineru/worker.py` - Remove GPU initialization, add HTTP client
- `src/Medical_KG_rev/services/mineru/config.py` - Add vLLM server configuration
- `ops/docker/mineru-worker/Dockerfile` - Remove vLLM dependencies, slim down image
- `ops/k8s/base/deployment-mineru-workers.yaml` - Remove GPU requirements, scale up replicas
- `docker-compose.yml` - Add vLLM server service, update worker configuration
- `config/mineru.yaml` - Add vLLM server section, update worker backend

**Removed Components**:

- GPU allocation logic in worker initialization (no longer needed)
- In-process vLLM engine imports and initialization code

**Testing**:

- `tests/services/mineru/test_vllm_client.py` - HTTP client tests
- `tests/services/mineru/test_circuit_breaker.py` - Circuit breaker tests
- `tests/services/mineru/test_split_container_e2e.py` - End-to-end split-container tests
- `tests/integration/test_vllm_server_integration.py` - vLLM server integration tests

### Migration Path

#### Phase 1: Infrastructure Setup (Week 1)

**Tasks**:

1. Deploy vLLM server container in staging environment
2. Verify model loading and OpenAI-compatible API functionality
3. Benchmark inference latency and throughput
4. Configure monitoring and alerting for vLLM server

**Validation Criteria**:

- vLLM server health check passes
- Model inference < 2s latency for typical VLM requests
- Prometheus metrics exposed and scraped
- No GPU memory leaks after 1000+ requests

#### Phase 2: Worker Implementation (Week 2)

**Tasks**:

1. Implement HTTP client for vLLM OpenAI-compatible API
2. Add retry logic, circuit breaker, connection pooling
3. Update MinerU worker to use HTTP client backend
4. Remove GPU initialization and in-process engine code

**Validation Criteria**:

- Worker startup < 5s (vs 60s before)
- HTTP client successfully calls vLLM server
- Circuit breaker opens on repeated failures
- Workers auto-recover when vLLM server restarts

#### Phase 3: Configuration and Deployment (Week 3)

**Tasks**:

1. Update `config/mineru.yaml` with vLLM server settings
2. Rebuild MinerU worker Docker image (slim, no vLLM)
3. Update Kubernetes deployments and services
4. Deploy to staging with 8 workers + 1 vLLM server

**Validation Criteria**:

- Workers discover vLLM server via Kubernetes DNS
- Load balancer distributes requests evenly
- No worker errors during normal operation
- GPU utilization > 85% on vLLM server

#### Phase 4: Testing and Validation (Week 4)

**Tasks**:

1. Run end-to-end PDF processing tests (100+ PDFs)
2. Compare quality with previous in-process implementation
3. Benchmark throughput and latency improvements
4. Chaos testing: kill vLLM server, verify worker resilience

**Validation Criteria**:

- PDF processing quality equivalent or better
- Throughput increase of 20-30% (continuous batching)
- Worker failures < 1% during vLLM server outages
- All integration tests passing

#### Phase 5: Production Rollout (Week 5)

**Tasks**:

1. Deploy to production with feature flag (10% traffic)
2. Monitor error rates, latency, resource usage for 3 days
3. Gradually increase to 50% traffic, then 100%
4. Decommission old monolithic deployment

**Validation Criteria**:

- Error rate < 0.1% (same as baseline)
- P95 latency within 10% of baseline
- No GPU OOM errors
- Worker pod crashes < 1 per day

### Rollback Plan

**Feature Flag**:

```yaml
mineru:
  deployment_mode: "monolithic"  # Revert to in-process engine
```

**Container Rollback**:

- Tag images: `mineru-worker:monolithic` (old), `mineru-worker:split` (new)
- Kubernetes rollout undo: `kubectl rollout undo deployment/mineru-workers`
- Docker Compose: `docker-compose down && docker-compose up -d` with old image

**Data Safety**:

- Kafka DLQ preserves failed PDFs for reprocessing
- No data loss during rollback (stateless workers)

### Resource Impact

**Before (Monolithic)**:

- 4 workers × 7GB VRAM = 28GB GPU memory
- Worker startup: 60s (model loading)
- GPU utilization: 60-70% (context switching between workers)

**After (Split-Container)**:

- vLLM server: 16-24GB VRAM (single hot model)
- 8 workers: 0 GPU memory (HTTP clients)
- Worker startup: < 5s (no model)
- GPU utilization: 85-90% (continuous batching)

**Net Savings**:

- **VRAM**: 8-12GB freed (can run larger models or increase batch size)
- **Worker capacity**: 2x (8 workers vs 4)
- **Startup time**: 12x faster (5s vs 60s)

### Security Considerations

**New Attack Surface**:

- HTTP API exposed between workers and vLLM server (mitigate with network policies)
- Potential SSRF if workers can specify vLLM URL (validate URL whitelist)

**Mitigations**:

- **Network segmentation**: Workers and vLLM in same private network, no external access
- **API authentication**: Optional API key for vLLM server (`--api-key` flag)
- **Input validation**: Sanitize PDF content before sending to vLLM
- **Rate limiting**: Per-worker rate limits on vLLM requests
- **Audit logging**: Log all vLLM API calls with worker ID, tenant ID, correlation ID

### Documentation Updates

**New Documentation**:

- `docs/gpu-microservices.md` - Add vLLM split-container architecture section
- `docs/devops/vllm-deployment.md` - vLLM server deployment guide
- `docs/troubleshooting/vllm-connectivity.md` - Debugging worker-to-vLLM issues
- `docs/runbooks/vllm_server_restart.md` - Operational runbook for vLLM maintenance

**Updated Documentation**:

- `docs/gpu-microservices.md` - Update MinerU architecture diagrams
- `docs/operations/mineru_service_runbook.md` - Update runbook with split-container procedures
- `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md` - Update GPU services section
- `README.md` - Update deployment instructions

### Performance Impact

**Expected Improvements** (based on vLLM benchmarks and MinerU docs):

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **GPU Memory** | 28GB (4 workers) | 16-24GB (server) | **20-40% freed** |
| **Worker Count** | 4 (GPU-bound) | 8-12 (CPU-bound) | **2-3x capacity** |
| **Throughput** | 160-960 PDFs/hr | 200-1200 PDFs/hr | **20-30% higher** |
| **Worker Startup** | 60s (model load) | <5s (instant) | **12x faster** |
| **GPU Utilization** | 60-70% | 85-90% | **25% more efficient** |
| **Fault Tolerance** | Worker crash = lost GPU | Server crash = workers queue | **Better resilience** |

### Monitoring & Alerting

**New Alerts**:

```yaml
- alert: VLLMServerDown
  expr: up{job="vllm-server"} == 0
  for: 1m
  severity: critical
  annotations:
    summary: "vLLM server is down, MinerU workers cannot process PDFs"

- alert: VLLMHighLatency
  expr: histogram_quantile(0.95, vllm:time_to_first_token_seconds) > 5
  for: 5m
  severity: warning
  annotations:
    summary: "vLLM inference latency P95 > 5s"

- alert: MinerUVLLMClientFailures
  expr: rate(mineru_vllm_client_failures_total[5m]) > 0.05
  for: 5m
  severity: warning
  annotations:
    summary: "MinerU workers experiencing >5% vLLM client failures"

- alert: VLLMGPUMemoryHigh
  expr: vllm:gpu_cache_usage_perc > 95
  for: 10m
  severity: warning
  annotations:
    summary: "vLLM KV cache usage >95%, consider scaling"
```

**Dashboards** (Grafana):

- **vLLM Server Health**: GPU utilization, memory, request queue depth, throughput
- **Worker-to-vLLM Connectivity**: Request latency, failure rate, circuit breaker state
- **End-to-End PDF Processing**: PDF throughput, processing time, quality metrics
