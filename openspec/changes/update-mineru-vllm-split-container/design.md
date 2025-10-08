# Technical Design: MinerU vLLM Split-Container Architecture

## Context

The archived MinerU implementation (`openspec/changes/archive/add-mineru-gpu-cli-integration`) uses a **monolithic architecture** where each MinerU worker runs the vLLM inference engine in-process (`-b vlm-vllm-engine`). This design describes the transition to a **split-container architecture** with a dedicated vLLM server and lightweight MinerU worker clients, following industry best practices documented by both vLLM and MinerU projects.

### System Architecture Context

**Current System** (Medical_KG_rev):

- Multi-protocol API gateway with orchestrated ingestion pipelines
- GPU-accelerated services for embeddings, PDF parsing (MinerU), and entity extraction
- Fail-fast GPU architecture (no CPU fallbacks)
- Kafka-based event-driven orchestration with job ledger state tracking
- Multi-tenant isolation with comprehensive observability (Prometheus, OpenTelemetry)

**Hardware Platform**:

- **GPU**: NVIDIA RTX 5090 (32GB VRAM, Blackwell architecture)
- **CUDA**: Version 12.8 (driver R570+)
- **CPU**: Modern multi-core (16-32 cores)
- **Memory**: 64-128GB RAM
- **OS**: Ubuntu 24.04 LTS

**Docker Infrastructure**:

- NVIDIA Container Toolkit for GPU passthrough
- Docker Compose for development environments
- Kubernetes (GKE/EKS) for production deployment

### Key Constraints

1. **Must use official images**: `vllm/vllm-openai` for server, MinerU official Dockerfile as base
2. **OpenAI-compatible API**: vLLM server exposes `/v1/chat/completions` endpoint
3. **Hot model sharing**: Single vLLM instance serves multiple MinerU workers
4. **Continuous batching**: vLLM automatically batches concurrent requests
5. **Fail-fast policy**: Services fail explicitly if GPU unavailable (no CPU fallback)
6. **CUDA 12.8 requirement**: Blackwell GPUs require specific CUDA version
7. **Network isolation**: Workers and vLLM in same private network (no external routing)

### References and Documentation Sources

This design is based on official documentation from:

- **vLLM Docker Deployment**: [https://docs.vllm.ai/en/v0.8.4/deployment/docker.html](https://docs.vllm.ai/en/v0.8.4/deployment/docker.html)
- **vLLM OpenAI-Compatible Server**: [https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- **MinerU Docker Deployment**: [https://opendatalab.github.io/MinerU/quick_start/docker_deployment/](https://opendatalab.github.io/MinerU/quick_start/docker_deployment/)
- **MinerU HTTP Client Mode**: [https://opendatalab.github.io/MinerU/usage/quick_usage/](https://opendatalab.github.io/MinerU/usage/quick_usage/)
- **vLLM Continuous Batching**: [https://docs.vllm.ai/en/stable/index.html](https://docs.vllm.ai/en/stable/index.html)
- **NVIDIA Container Toolkit**: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Qwen2.5-VL with vLLM**: [https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)

## Goals / Non-Goals

### Goals

1. **Adopt industry-standard split-container pattern** for VLM inference with MinerU clients
2. **Maximize GPU utilization** through hot model sharing and continuous batching (target: >85%)
3. **Enable independent scaling** of inference capacity (vLLM) and processing capacity (workers)
4. **Reduce worker VRAM footprint** from 7GB to <0.5GB (14x reduction)
5. **Improve operational hygiene** with service separation, independent upgrades, clear boundaries
6. **Achieve production readiness** with standardized serving, observability, and fault tolerance

### Non-Goals

- **Multi-model serving**: Single vLLM instance serves one VLM model (Qwen2.5-VL-7B-Instruct)
- **Custom vLLM engine**: Use official `vllm/vllm-openai` image without modifications
- **CPU fallback for VLM**: Maintain fail-fast GPU-only policy
- **Multi-region deployment**: Single-region deployment (can extend later)
- **Real-time streaming**: Batch-oriented processing (not interactive)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                              │
│  Kafka Topics: pdf.parse.requests.v1 → pdf.parse.results.v1             │
│  Job Ledger: State tracking, retries, DLQ                               │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             │ Job Distribution
                             │
         ┌───────────────────┴───────────────────┬───────────────────┐
         │                   │                   │                   │
    ┌────▼─────┐      ┌──────▼────┐      ┌──────▼────┐      ┌──────▼────┐
    │ Worker 1 │      │ Worker 2  │      │ Worker 3  │ ...  │ Worker 8  │
    │ CPU: 2   │      │ CPU: 2    │      │ CPU: 2    │      │ CPU: 2    │
    │ RAM: 4GB │      │ RAM: 4GB  │      │ RAM: 4GB  │      │ RAM: 4GB  │
    │ GPU: NO  │      │ GPU: NO   │      │ GPU: NO   │      │ GPU: NO   │
    └────┬─────┘      └──────┬────┘      └──────┬────┘      └──────┬────┘
         │                   │                   │                   │
         │    HTTP POST /v1/chat/completions (OpenAI-compatible)    │
         │                   │                   │                   │
         └───────────────────┴───────────────────┴───────────────────┘
                             │
                             │ Load Balancer / K8s Service
                             │ (Round-robin, health checks)
                             │
                    ┌────────▼─────────┐
                    │  vLLM Server     │
                    │  Port: 8000      │
                    │                  │
                    │  Model:          │
                    │  Qwen2.5-VL-7B   │
                    │                  │
                    │  VRAM: 16-24GB   │
                    │  GPU: RTX 5090   │
                    │  CUDA: 12.8      │
                    │                  │
                    │  Features:       │
                    │  - Hot Model     │
                    │  - Continuous    │
                    │    Batching      │
                    │  - KV Cache      │
                    │  - Metrics       │
                    └──────────────────┘
```

### Data Flow

1. **Job Submission**: Orchestrator publishes PDF processing job to Kafka (`pdf.parse.requests.v1`)
2. **Worker Consumption**: MinerU worker consumes message, downloads PDF from MinIO/S3
3. **PDF Parsing**: Worker invokes `mineru` CLI with `-b vlm-http-client` backend
4. **VLM Inference**: MinerU CLI makes HTTP requests to vLLM server for vision tasks (layout understanding, table detection)
5. **Continuous Batching**: vLLM server coalesces concurrent requests from multiple workers into GPU batches
6. **Response**: vLLM returns inference results, worker continues PDF processing
7. **Result Publishing**: Worker publishes parsed document to Kafka (`pdf.parse.results.v1`)

## Key Design Decisions

### Decision 1: Use vLLM Official Docker Image

**Rationale**:

The vLLM project publishes an official `vllm/vllm-openai` Docker image that:

- Contains pre-built CUDA 12.8 + PyTorch 2.6 binaries optimized for modern GPUs
- Includes OpenAI-compatible HTTP server (`/v1/chat/completions` endpoint)
- Has battle-tested dependencies and build configurations
- Receives regular updates and security patches
- Works out-of-the-box with Blackwell GPUs (RTX 50-series)

**Implementation**:

```dockerfile
# We use the official image directly - no custom Dockerfile needed
# Base image: vllm/vllm-openai:latest (or pinned version like v0.10.2)
```

**Container startup command** (Docker Compose):

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
        --trust-remote-code
```

**Command breakdown** (per vLLM docs):

- `python -m vllm.entrypoints.openai.api_server`: Standard OpenAI-compatible server
- `--model`: Hugging Face model ID (downloads automatically on first run)
- `--host 0.0.0.0`: Bind to all interfaces (accessible to workers)
- `--port 8000`: Standard HTTP port (OpenAI API convention)
- `--gpu-memory-utilization 0.92`: Use 92% of VRAM for model + KV cache (safe on 32GB GPU)
- `--max-model-len 32768`: Maximum sequence length (supports long documents)
- `--tensor-parallel-size 1`: Single GPU (no multi-GPU sharding needed for 7B model)
- `--trust-remote-code`: Allow custom model code (required for Qwen2.5-VL)

**Alternatives Considered**:

- **Custom Dockerfile**: Rejected - introduces maintenance burden, version drift, build failures
- **NGC PyTorch container + pip install vllm**: Rejected - official image is more reliable
- **Build from source**: Rejected - unnecessary complexity, slower iteration

**References**:

- vLLM Docker guide: [https://docs.vllm.ai/en/v0.8.4/deployment/docker.html](https://docs.vllm.ai/en/v0.8.4/deployment/docker.html)
- OpenAI server docs: [https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

### Decision 2: Qwen2.5-VL-7B-Instruct as Vision-Language Model

**Rationale**:

MinerU's VLM backend requires a vision-language model for document understanding tasks (layout analysis, table detection, reading order). Qwen2.5-VL-7B-Instruct is chosen because:

- **vLLM native support**: Official recipe at [vLLM Qwen2.5-VL docs](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)
- **Document-optimized**: Trained on document understanding tasks (OCR, layout, tables)
- **7B parameters**: Fits comfortably on RTX 5090 with room for KV cache
- **Fast inference**: ~1-2s per VLM request with vLLM optimization
- **Open license**: Permissive license for commercial use

**Model Specifications**:

- **Size**: 7B parameters (~14GB fp16, ~7GB int8 quantized)
- **VRAM Usage**: 16-24GB including KV cache and activations
- **Context Length**: 32k tokens (sufficient for multi-page PDFs)
- **Vision Encoder**: Integrated vision tower for image understanding
- **Performance**: ~50 tokens/sec generation with vLLM on RTX 5090

**Alternatives Considered**:

- **LLaVA-1.5**: Older architecture, less optimized for documents
- **CogVLM**: Larger model (34B), doesn't fit on single 32GB GPU
- **GPT-4V API**: External API dependency, latency, cost concerns

**References**:

- Qwen2.5-VL vLLM recipe: [https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)

### Decision 3: MinerU HTTP Client Backend (`vlm-http-client`)

**Rationale**:

MinerU officially documents two vLLM backends:

1. **In-process engine** (`-b vlm-vllm-engine`): Loads model in same process
2. **HTTP client** (`-b vlm-http-client`): Calls external vLLM server via HTTP

The HTTP client backend is chosen because:

- **Recommended by MinerU docs** for production deployments ([MinerU HTTP client docs](https://opendatalab.github.io/MinerU/usage/quick_usage/))
- **Service isolation**: Separates PDF processing from GPU inference
- **Hot model sharing**: Multiple workers reuse same model instance
- **Stateless workers**: Workers don't hold GPU state, easy to scale
- **Fault tolerance**: Worker crash doesn't affect vLLM server

**Implementation**:

```bash
# Worker invokes MinerU CLI with HTTP client backend
mineru -p /workspace/in -o /workspace/out \
  -b vlm-http-client \
  -u http://vllm-server:8000
```

**Backend Comparison**:

| Feature | In-Process (`vlm-vllm-engine`) | HTTP Client (`vlm-http-client`) |
|---------|-------------------------------|--------------------------------|
| **Model Loading** | Per worker (60s startup) | Once at server start |
| **VRAM per Worker** | 7GB (full model) | <0.5GB (client only) |
| **Batching** | No (isolated workers) | Yes (continuous batching) |
| **Worker Scaling** | GPU-bound (4 workers max) | CPU-bound (8-12 workers) |
| **Fault Isolation** | Worker crash = GPU reset | Independent failures |
| **Upgrade Path** | Rebuild all workers | Upgrade server only |

**References**:

- MinerU HTTP client mode: [https://opendatalab.github.io/MinerU/usage/quick_usage/](https://opendatalab.github.io/MinerU/usage/quick_usage/)

### Decision 4: OpenAI-Compatible API Contract

**Rationale**:

vLLM's OpenAI-compatible server provides a **standard HTTP API** that:

- Matches OpenAI `/v1/chat/completions` endpoint spec
- Supports multimodal inputs (text + images) for VLMs
- Uses JSON request/response format
- Enables easy client library reuse (httpx, openai-python)
- Allows migration to other providers (OpenAI, Anthropic, etc.) without code changes

**API Contract**:

**Request** (POST `/v1/chat/completions`):

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Analyze this document page and identify tables."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KG..."
          }
        }
      ]
    }
  ],
  "max_tokens": 4096,
  "temperature": 0.0
}
```

**Response**:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1704123456,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The page contains 2 tables..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 512,
    "completion_tokens": 128,
    "total_tokens": 640
  }
}
```

**Client Implementation**:

```python
import httpx
import base64

async def call_vllm(image_bytes: bytes, prompt: str) -> str:
    """Call vLLM OpenAI-compatible endpoint for VLM inference."""

    # Encode image as base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    # Make HTTP request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://vllm-server:8000/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.0
            },
            timeout=300.0
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
```

**References**:

- vLLM OpenAI server docs: [https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

### Decision 5: Docker Compose for Development, Kubernetes for Production

**Rationale**:

Following the project's existing deployment strategy:

- **Development**: Docker Compose for local iteration (simple, fast setup)
- **Production**: Kubernetes for scaling, fault tolerance, observability

**Docker Compose Configuration**:

```yaml
version: '3.8'

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
        --trust-remote-code
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
    environment:
      - CUDA_VISIBLE_DEVICES=0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    networks:
      - medical-kg-net

  mineru-worker:
    image: ghcr.io/your-org/mineru-client:latest
    environment:
      - VLLM_SERVER_URL=http://vllm-server:8000
      - MINERU_BACKEND=vlm-http-client
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      vllm-server:
        condition: service_healthy
    volumes:
      - ./data/in:/workspace/in
      - ./data/out:/workspace/out
    networks:
      - medical-kg-net
    deploy:
      replicas: 8

networks:
  medical-kg-net:
    driver: bridge
```

**Docker flags explained**:

- `ipc: host`: Share host IPC namespace for PyTorch shared memory (prevents crashes)
- `ulimits.memlock: -1`: Unlimited locked memory for NCCL (GPU communication)
- `ulimits.stack: 67108864`: Large stack size for deep model inference
- `~/.cache/huggingface`: Cache downloaded models (avoid re-download on restart)

**References**:

- vLLM Docker guide: [https://docs.vllm.ai/en/v0.8.4/deployment/docker.html](https://docs.vllm.ai/en/v0.8.4/deployment/docker.html)
- MinerU Docker deployment: [https://opendatalab.github.io/MinerU/quick_start/docker_deployment/](https://opendatalab.github.io/MinerU/quick_start/docker_deployment/)

### Decision 6: Kubernetes Service for Load Balancing

**Rationale**:

Kubernetes `Service` provides built-in load balancing, health checks, and service discovery for vLLM server:

- **Round-robin distribution**: Evenly distribute requests across vLLM pods (if multiple)
- **Health checks**: Automatically remove unhealthy pods from rotation
- **DNS-based discovery**: Workers connect via `http://vllm-server:8000` (no hardcoded IPs)
- **Session affinity**: Optional (not needed for stateless inference)

**Kubernetes Service**:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
  namespace: medical-kg
  labels:
    app: vllm-server
spec:
  type: ClusterIP  # Internal only (not exposed externally)
  selector:
    app: vllm-server
  ports:
  - name: http
    protocol: TCP
    port: 8000
    targetPort: 8000
  sessionAffinity: None  # Stateless, no session pinning needed
```

**Worker Configuration**:

Workers reference the service by DNS name:

```python
VLLM_SERVER_URL = "http://vllm-server:8000"  # Resolves to Service ClusterIP
```

**Alternatives Considered**:

- **Nginx reverse proxy**: Rejected - Kubernetes Service is simpler, native integration
- **Istio service mesh**: Rejected - overkill for single-service communication
- **Direct pod IPs**: Rejected - breaks on pod restart (ephemeral IPs)

### Decision 7: Connection Pooling and Retry Logic

**Rationale**:

MinerU workers make frequent HTTP requests to vLLM server (multiple per PDF). Implementing **connection pooling** and **retry logic** improves:

- **Performance**: Reuse TCP connections (avoid handshake overhead)
- **Reliability**: Auto-retry transient failures (network blips, server restarts)
- **Resilience**: Circuit breaker prevents cascade failures

**Implementation**:

```python
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class VLLMClient:
    """HTTP client for vLLM server with connection pooling and retries."""

    def __init__(self, base_url: str):
        self.base_url = base_url

        # Connection pool configuration
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=10.0,   # 10s to establish connection
                read=300.0,     # 5 minutes for inference
                write=10.0,     # 10s to send request
                pool=5.0        # 5s to get connection from pool
            ),
            limits=httpx.Limits(
                max_connections=10,         # Total connections
                max_keepalive_connections=5 # Persistent connections
            ),
            transport=httpx.AsyncHTTPTransport(retries=0)  # No auto-retries (we handle manually)
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True
    )
    async def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0
    ) -> dict:
        """
        Call vLLM /v1/chat/completions endpoint with retries.

        Retry logic:
        - 3 attempts max
        - Exponential backoff: 4s, 8s, 16s (with jitter)
        - Retry on: timeouts, network errors
        - No retry on: HTTP 400 (bad request), HTTP 500 (server error)
        """
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()
```

**Retry Strategy Rationale**:

- **3 attempts**: Balances reliability vs latency (60s max delay)
- **Exponential backoff**: Gives server time to recover from load spikes
- **Timeout exceptions**: Retry (server may be busy)
- **Network errors**: Retry (transient connectivity issues)
- **HTTP 4xx/5xx**: No retry (permanent errors, log and fail fast)

**Alternatives Considered**:

- **Synchronous client**: Rejected - blocks worker thread during inference
- **No retry logic**: Rejected - increases PDF failure rate due to transients
- **Aggressive retries (10+ attempts)**: Rejected - increases latency too much

### Decision 8: Circuit Breaker Pattern

**Rationale**:

If vLLM server is down or overloaded, we want workers to **fail fast** rather than pile up requests:

- **Prevent cascade failures**: Stop sending requests to unhealthy server
- **Reduce latency**: Fail immediately instead of waiting for timeout
- **Auto-recovery**: Test health and reopen circuit when server recovers

**Implementation**:

```python
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

class CircuitState(Enum):
    CLOSED = "closed"   # Normal operation
    OPEN = "open"       # Server unhealthy, reject requests
    HALF_OPEN = "half_open"  # Testing if server recovered

@dataclass
class CircuitBreaker:
    """Circuit breaker for vLLM server calls."""

    failure_threshold: int = 5  # Open after 5 consecutive failures
    recovery_timeout: float = 60.0  # Try recovery after 60s
    success_threshold: int = 2  # Close after 2 successful tests

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    success_count: int = field(default=0, init=False)
    last_failure_time: datetime | None = field(default=None, init=False)

    def record_success(self):
        """Record successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._close_circuit()
        else:
            self.failure_count = 0

    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self._open_circuit()

    def can_execute(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._half_open_circuit()
                    return True
            return False

        # HALF_OPEN: Allow test request
        return True

    def _open_circuit(self):
        """Open circuit (reject requests)."""
        self.state = CircuitState.OPEN
        logger.error("Circuit breaker OPENED - vLLM server unhealthy")

    def _half_open_circuit(self):
        """Half-open circuit (test recovery)."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info("Circuit breaker HALF-OPEN - testing vLLM server recovery")

    def _close_circuit(self):
        """Close circuit (normal operation)."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker CLOSED - vLLM server recovered")
```

**Usage in Worker**:

```python
circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

async def safe_vllm_call(messages: list[dict]) -> dict:
    """Make vLLM call with circuit breaker protection."""

    if not circuit_breaker.can_execute():
        raise CircuitBreakerOpenError("vLLM server circuit breaker is OPEN")

    try:
        result = await vllm_client.chat_completion(messages)
        circuit_breaker.record_success()
        return result
    except Exception as e:
        circuit_breaker.record_failure()
        raise
```

**Prometheus Metrics**:

```python
circuit_breaker_state = Gauge(
    'mineru_vllm_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half-open, 2=open)',
    labelnames=['worker_id']
)

circuit_breaker_failures_total = Counter(
    'mineru_vllm_circuit_breaker_failures_total',
    'Circuit breaker failure count',
    labelnames=['worker_id']
)
```

## Performance Characteristics

### Expected Throughput

**vLLM Server** (single RTX 5090):

- **Batch size**: 4-8 concurrent requests (continuous batching)
- **Latency**: 1-2s per VLM request (varies by prompt/image size)
- **Throughput**: 20-30 VLM requests/minute under load

**MinerU Workers** (8 workers, CPU-bound):

- **VLM calls per PDF**: 3-5 (layout analysis, table detection, figure extraction)
- **Processing time per PDF**: 30-90s (includes VLM + post-processing)
- **Throughput**: 6-10 PDFs/worker/hour = **48-80 PDFs/hour total**

**Comparison to Baseline**:

| Configuration | Workers | GPU Usage | Throughput | Startup Time |
|---------------|---------|-----------|------------|--------------|
| **Old (In-Process)** | 4 | 28GB (4×7GB) | 40-72 PDFs/hr | 60s |
| **New (Split)** | 8 | 16-24GB (server) | 48-80 PDFs/hr | <5s |
| **Improvement** | +100% | -20% VRAM | +20-30% | 12x faster |

### Resource Utilization

**vLLM Server**:

- **VRAM**: 16-24GB (model 14GB + KV cache 2-10GB)
- **GPU Utilization**: 85-90% (continuous batching)
- **CPU**: 4-8 cores for data preprocessing
- **Network**: 10-50 MB/s (image uploads from workers)

**MinerU Workers** (per worker):

- **CPU**: 2 cores (PDF parsing, post-processing)
- **RAM**: 4GB (temporary PDF storage, output buffers)
- **Network**: 5-10 MB/s (Kafka, object storage, vLLM calls)
- **Disk I/O**: 20-50 MB/s (PDF read, image write)

## Fault Tolerance and Error Handling

### Failure Scenarios

| Failure | Detection | Recovery | Impact |
|---------|-----------|----------|--------|
| **vLLM server down** | Health check fails | Workers circuit breaker opens | PDFs fail, retry after 60s |
| **vLLM server OOM** | HTTP 503 response | Restart pod, workers retry | 30-60s downtime |
| **Worker crash** | Kafka message not ACKed | Auto-retry by Kafka | No data loss |
| **Network partition** | HTTP timeout | Retry with backoff | Temporary latency increase |
| **Model download failure** | vLLM startup fails | Retry with cached models | Startup delay |

### Health Checks

**vLLM Server Health Check**:

```python
# Kubernetes liveness probe
GET http://vllm-server:8000/health

# Expected response (HTTP 200):
{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "gpu_memory_used_gb": 18.5,
  "gpu_memory_total_gb": 32.0
}
```

**Worker Health Check**:

```python
# Check vLLM connectivity on worker startup
async def check_vllm_connectivity():
    try:
        response = await vllm_client.client.get("/health")
        if response.status_code == 200:
            logger.info("vLLM server connectivity OK")
            return True
    except Exception as e:
        logger.error(f"vLLM server unreachable: {e}")
        return False

# Fail-fast: Exit if vLLM unreachable on startup
if not await check_vllm_connectivity():
    logger.error("Cannot reach vLLM server, exiting")
    sys.exit(1)
```

## Observability and Monitoring

### Prometheus Metrics

**vLLM Server** (built-in metrics):

```python
# Exposed at http://vllm-server:8000/metrics

# Request metrics
vllm:num_requests_running  # Active inference requests
vllm:num_requests_waiting  # Queued requests
vllm:request_success_total  # Successful requests
vllm:request_failure_total  # Failed requests

# Performance metrics
vllm:time_to_first_token_seconds  # Latency to first token
vllm:time_per_output_token_seconds  # Token generation speed
vllm:e2e_request_latency_seconds  # End-to-end latency

# Resource metrics
vllm:gpu_cache_usage_perc  # KV cache utilization
vllm:gpu_memory_usage_bytes  # GPU memory usage
vllm:cpu_memory_usage_bytes  # Host memory usage

# Token metrics
vllm:prompt_tokens_total  # Input tokens processed
vllm:generation_tokens_total  # Output tokens generated
```

**Worker Metrics** (custom):

```python
from prometheus_client import Counter, Histogram, Gauge

mineru_vllm_request_duration_seconds = Histogram(
    'mineru_vllm_request_duration_seconds',
    'vLLM HTTP request latency',
    labelnames=['worker_id', 'status'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

mineru_vllm_request_failures_total = Counter(
    'mineru_vllm_request_failures_total',
    'Failed vLLM requests',
    labelnames=['worker_id', 'error_type']
)

mineru_vllm_circuit_breaker_state = Gauge(
    'mineru_vllm_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half-open, 2=open)',
    labelnames=['worker_id']
)

mineru_pdf_vllm_calls_per_pdf = Histogram(
    'mineru_pdf_vllm_calls_per_pdf',
    'Number of vLLM calls per PDF',
    buckets=[1, 2, 3, 5, 10, 20]
)
```

### OpenTelemetry Distributed Tracing

**Trace Propagation**:

```python
from opentelemetry import trace
from opentelemetry.propagate import inject

tracer = trace.get_tracer(__name__)

async def call_vllm_with_tracing(messages: list[dict]) -> dict:
    """Make vLLM call with distributed tracing."""

    with tracer.start_as_current_span("mineru.vllm.chat_completion") as span:
        # Propagate trace context in HTTP headers
        headers = {}
        inject(headers)

        span.set_attribute("vllm.model", "Qwen/Qwen2.5-VL-7B-Instruct")
        span.set_attribute("vllm.max_tokens", 4096)
        span.set_attribute("vllm.num_messages", len(messages))

        try:
            response = await vllm_client.client.post(
                "/v1/chat/completions",
                json={"model": "Qwen/Qwen2.5-VL-7B-Instruct", "messages": messages},
                headers=headers
            )
            response.raise_for_status()

            result = response.json()
            span.set_attribute("vllm.prompt_tokens", result["usage"]["prompt_tokens"])
            span.set_attribute("vllm.completion_tokens", result["usage"]["completion_tokens"])

            return result
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
```

**Trace Visualization** (Jaeger):

```
PDF Processing [10.5s]
├─ mineru.cli.invoke [10.2s]
│  ├─ mineru.vllm.chat_completion (layout) [2.1s]
│  ├─ mineru.vllm.chat_completion (table) [1.8s]
│  └─ mineru.vllm.chat_completion (figure) [1.5s]
└─ mineru.output.parse [0.3s]
```

### Alerting Rules

```yaml
groups:
- name: vllm_server_alerts
  rules:
  - alert: VLLMServerDown
    expr: up{job="vllm-server"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "vLLM server is down"
      description: "vLLM server has been down for 1 minute. MinerU workers cannot process PDFs."

  - alert: VLLMHighLatency
    expr: histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[5m])) > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "vLLM inference latency high"
      description: "95th percentile latency is {{ $value }}s (threshold: 10s)"

  - alert: VLLMGPUMemoryHigh
    expr: vllm:gpu_memory_usage_bytes / (32 * 1024^3) > 0.95
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "vLLM GPU memory usage high"
      description: "GPU memory usage is {{ $value | humanizePercentage }} (threshold: 95%)"

  - alert: MinerUVLLMClientFailures
    expr: rate(mineru_vllm_request_failures_total[5m]) / rate(mineru_vllm_request_duration_seconds_count[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "MinerU vLLM client failure rate high"
      description: "{{ $value | humanizePercentage }} of vLLM requests failing (threshold: 5%)"

  - alert: MinerUCircuitBreakerOpen
    expr: mineru_vllm_circuit_breaker_state == 2
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "MinerU circuit breaker OPEN"
      description: "Worker {{ $labels.worker_id }} circuit breaker is OPEN. vLLM server unhealthy."
```

## Security Considerations

### Network Isolation

**Threat**: Unauthorized access to vLLM server from external networks

**Mitigation**:

- **Kubernetes NetworkPolicy**: Restrict vLLM server to same namespace

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vllm-server-netpol
  namespace: medical-kg
spec:
  podSelector:
    matchLabels:
      app: vllm-server
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: mineru-worker  # Only MinerU workers can access
    ports:
    - protocol: TCP
      port: 8000
```

### Input Validation

**Threat**: Malicious PDF input causes vLLM server crash or prompt injection

**Mitigation**:

- **PDF size limits**: Max 100MB per PDF
- **Image size limits**: Max 10MB per image sent to vLLM
- **Content sanitization**: Validate PDF structure before processing
- **Prompt templates**: Use fixed templates, no user-controlled prompts

### API Authentication (Optional)

**Threat**: Rogue workers impersonate legitimate workers

**Mitigation** (if needed):

- **API keys**: vLLM server supports `--api-key` flag

```bash
# Server startup with API key
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --api-key $VLLM_API_KEY
```

- **Worker configuration**:

```python
# Workers send API key in Authorization header
headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
```

## Migration Strategy

### Phase 1: vLLM Server Deployment (Week 1)

**Objective**: Deploy and validate vLLM server independently

**Tasks**:

1. Deploy vLLM server to staging Kubernetes cluster
2. Verify model loading and GPU allocation
3. Smoke test OpenAI API endpoint with curl/Python
4. Run load test: 100+ concurrent requests
5. Monitor GPU memory, latency, throughput for 24 hours

**Success Criteria**:

- Server starts successfully, loads model in <60s
- Health check returns HTTP 200
- P95 latency <5s for typical VLM requests
- GPU memory stable (no leaks)
- No crashes after 1000+ requests

### Phase 2: Worker Implementation (Week 2)

**Objective**: Update MinerU workers to use HTTP client backend

**Tasks**:

1. Implement `VLLMClient` class with connection pooling, retries, circuit breaker
2. Update `MinerUWorker` to use HTTP client instead of in-process engine
3. Remove GPU initialization code and vLLM dependencies
4. Build new Docker image (`mineru-client:split`)
5. Unit test HTTP client with mocked vLLM responses

**Success Criteria**:

- Worker Docker image <2GB (vs 8GB for old monolithic image)
- Worker startup <5s (vs 60s)
- Unit tests pass (100% coverage for HTTP client)
- Integration test: worker + real vLLM server process 10 PDFs successfully

### Phase 3: Integration Testing (Week 3)

**Objective**: End-to-end testing with split-container architecture

**Tasks**:

1. Deploy 1 vLLM server + 4 workers to staging
2. Process 100 PDFs from test corpus (clinical trials, drug labels, papers)
3. Compare output quality with baseline (old monolithic implementation)
4. Benchmark throughput and latency
5. Chaos testing: kill vLLM server, verify worker resilience

**Success Criteria**:

- PDF processing quality equivalent (95%+ match with baseline)
- Throughput increase of 20-30%
- Workers survive vLLM server restarts (circuit breaker works)
- No data loss or corruption

### Phase 4: Production Rollout (Week 4-5)

**Objective**: Gradual rollout to production

**Tasks**:

1. **Week 4, Day 1-2**: Deploy to production with feature flag (0% traffic)
2. **Week 4, Day 3-4**: Enable 10% traffic to new split-container workers
3. **Week 4, Day 5-7**: Monitor error rates, latency, quality for 3 days
4. **Week 5, Day 1-2**: Increase to 50% traffic if metrics good
5. **Week 5, Day 3-5**: Monitor for 3 days, fix any issues
6. **Week 5, Day 6-7**: Increase to 100% traffic, decommission old monolithic workers

**Rollback Triggers**:

- Error rate increase >50% vs baseline
- P95 latency increase >100% vs baseline
- PDF quality regression detected
- vLLM server instability (crashes >1/hour)

## Open Questions and Resolutions

### Q1: Should we pre-download model in Docker image or lazy-load at runtime?

**Answer**: Pre-download in Docker image

**Rationale**:

- Ensures model availability (no network dependency at runtime)
- Faster pod startup (no download wait)
- Easier to version-control (image tag = model version)

**Implementation**:

```dockerfile
# If we need a custom vllm-server image (usually not needed)
FROM vllm/vllm-openai:latest
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')"
```

### Q2: How to handle vLLM server upgrades without downtime?

**Answer**: Blue/green deployment with Kubernetes

**Implementation**:

1. Deploy new vLLM server version with different label (`version: v2`)
2. Update worker configuration to point to new service
3. Wait for workers to drain old requests
4. Scale down old vLLM server
5. Rollback if issues detected (switch back to `version: v1` service)

### Q3: What if multiple tenants need different VLM models?

**Answer**: Out of scope for initial implementation

**Future Extension**:

- Deploy multiple vLLM servers (one per model)
- Route workers to appropriate server based on tenant_id
- Use Istio/Nginx for model-aware routing

### Q4: Should workers batch PDFs before calling vLLM?

**Answer**: No - MinerU CLI handles batching internally

**Rationale**:

- MinerU already batches VLM requests within a single PDF processing job
- vLLM server handles cross-worker batching via continuous batching
- Worker-level batching would add complexity without clear benefit

## References

### Official Documentation

- **vLLM Docker**: [https://docs.vllm.ai/en/v0.8.4/deployment/docker.html](https://docs.vllm.ai/en/v0.8.4/deployment/docker.html)
- **vLLM OpenAI Server**: [https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- **vLLM Continuous Batching**: [https://docs.vllm.ai/en/stable/index.html](https://docs.vllm.ai/en/stable/index.html)
- **Qwen2.5-VL**: [https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)
- **MinerU Docker**: [https://opendatalab.github.io/MinerU/quick_start/docker_deployment/](https://opendatalab.github.io/MinerU/quick_start/docker_deployment/)
- **MinerU HTTP Client**: [https://opendatalab.github.io/MinerU/usage/quick_usage/](https://opendatalab.github.io/MinerU/usage/quick_usage/)
- **NVIDIA Container Toolkit**: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Internal Documentation

- `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md` - System architecture overview
- `openspec/changes/archive/add-mineru-gpu-cli-integration/` - Previous MinerU implementation
- `docs/gpu-microservices.md` - GPU service design patterns
