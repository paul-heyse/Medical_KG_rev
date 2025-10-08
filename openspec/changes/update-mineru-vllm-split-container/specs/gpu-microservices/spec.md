# Spec Delta: GPU Microservices (Split-Container Pattern)

## ADDED Requirements

### Requirement: Split-Container Service Pattern

The system SHALL support a **split-container pattern** for GPU services where model inference is separated from application logic in independent containers.

**Pattern Components**:

1. **Inference Server Container**: GPU-bound service running ML models with standardized HTTP API
2. **Client Containers**: Stateless CPU-bound services that call inference server via HTTP

**Benefits**:

- Hot model sharing across multiple clients (reduced VRAM per client)
- Continuous batching for higher GPU utilization (>85%)
- Independent scaling of inference capacity and client capacity
- Operational hygiene: separate upgrades, monitoring, fault isolation

#### Scenario: Inference server serves multiple clients

- **GIVEN** an inference server (e.g., vLLM) is running with model loaded
- **AND** 8 client containers are running
- **WHEN** clients make concurrent inference requests
- **THEN** the server SHALL batch requests for GPU processing (continuous batching)
- **AND** all clients SHALL receive responses successfully
- **AND** GPU utilization SHALL exceed 85% (efficient batching)
- **AND** clients SHALL use <1GB memory each (no model weights)

#### Scenario: Inference server scales independently

- **GIVEN** an inference server is running with 1 GPU
- **AND** client load increases 2x
- **WHEN** administrators scale clients from 8 to 16 containers
- **THEN** the inference server SHALL handle increased load without modification
- **AND** administrators MAY scale server independently (add GPU capacity)
- **AND** clients SHALL continue operating with increased concurrency

### Requirement: OpenAI-Compatible Inference API

GPU inference servers SHALL expose an **OpenAI-compatible HTTP API** for standardized client integration.

**API Standard**: OpenAI Chat Completions API (`/v1/chat/completions`)

**Rationale**:

- Industry-standard API format for ML inference
- Client libraries widely available (httpx, openai-python)
- Provider portability (self-hosted, OpenAI, Anthropic, etc.)
- Standardized request/response format

#### Scenario: Inference server exposes OpenAI-compatible endpoint

- **GIVEN** an inference server is running (e.g., vLLM)
- **WHEN** the server starts
- **THEN** the server SHALL expose POST endpoint `/v1/chat/completions`
- **AND** the endpoint SHALL accept OpenAI-format request body:

  ```json
  {
    "model": "<model-name>",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 4096,
    "temperature": 0.0
  }
  ```

- **AND** the endpoint SHALL return OpenAI-format response with `choices`, `usage`, `id` fields

#### Scenario: Client uses httpx for inference requests

- **GIVEN** a client container needs ML inference
- **WHEN** the client makes HTTP POST to inference server `/v1/chat/completions`
- **THEN** the client SHALL use httpx AsyncClient with connection pooling
- **AND** the client SHALL receive response in OpenAI format
- **AND** the client SHALL parse `choices[0].message.content` for model output

### Requirement: Inference Server Health Checks

GPU inference servers SHALL expose health check endpoints for Kubernetes liveness and readiness probes.

#### Scenario: Health check endpoint for liveness

- **GIVEN** an inference server is running
- **WHEN** Kubernetes liveness probe GETs `/health` endpoint
- **THEN** the server SHALL return HTTP 200 if GPU available and model loaded
- **AND** the server SHALL return HTTP 503 if GPU unavailable or model failed to load
- **AND** the response SHALL complete within 5 seconds

#### Scenario: Health check includes GPU status

- **GIVEN** an inference server is running successfully
- **WHEN** a health check probe GETs `/health`
- **THEN** the response SHALL include JSON with:

  ```json
  {
    "status": "healthy",
    "model": "<model-name>",
    "gpu_memory_used_gb": 18.5,
    "gpu_memory_total_gb": 32.0
  }
  ```

### Requirement: Inference Server Prometheus Metrics

GPU inference servers SHALL expose Prometheus metrics for observability of inference performance and resource usage.

**Metrics Categories**:

- Request metrics: active requests, queue depth, throughput
- Performance metrics: latency (time to first token, per-output token), throughput (tokens/sec)
- Resource metrics: GPU memory, KV cache utilization

#### Scenario: Server exposes request queue metrics

- **GIVEN** an inference server is running
- **WHEN** Prometheus scrapes `/metrics` endpoint
- **THEN** the server SHALL expose gauge `num_requests_running` (active requests)
- **AND** the server SHALL expose gauge `num_requests_waiting` (queued requests)
- **AND** the server SHALL expose counter `request_success_total` (successful requests)
- **AND** the server SHALL expose counter `request_failure_total` (failed requests)

#### Scenario: Server exposes latency metrics

- **GIVEN** an inference server is processing requests
- **WHEN** Prometheus scrapes `/metrics` endpoint
- **THEN** the server SHALL expose histogram `time_to_first_token_seconds` (latency to first token)
- **AND** the server SHALL expose histogram `time_per_output_token_seconds` (token generation speed)
- **AND** the server SHALL expose histogram `e2e_request_latency_seconds` (end-to-end latency)

#### Scenario: Server exposes GPU resource metrics

- **GIVEN** an inference server is running on GPU
- **WHEN** Prometheus scrapes `/metrics` endpoint
- **THEN** the server SHALL expose gauge `gpu_cache_usage_perc` (KV cache utilization percentage)
- **AND** the server SHALL expose gauge `gpu_memory_usage_bytes` (GPU memory in bytes)
- **AND** the server SHALL expose counter `prompt_tokens_total` (total input tokens processed)
- **AND** the server SHALL expose counter `generation_tokens_total` (total output tokens generated)

### Requirement: Client Circuit Breaker Pattern

Client containers calling GPU inference servers SHALL implement **circuit breaker pattern** to prevent cascade failures.

**Circuit States**:

- **CLOSED**: Normal operation, requests allowed
- **OPEN**: Server unhealthy, requests rejected immediately (fail-fast)
- **HALF_OPEN**: Testing server recovery, limited requests allowed

**Transition Logic**:

- CLOSED → OPEN: After N consecutive failures (default: 5)
- OPEN → HALF_OPEN: After timeout period (default: 60 seconds)
- HALF_OPEN → CLOSED: After M consecutive successes (default: 2)

#### Scenario: Circuit breaker opens on repeated failures

- **GIVEN** a client is making requests to inference server
- **WHEN** 5 consecutive requests fail (timeout or server error)
- **THEN** the client's circuit breaker SHALL transition to OPEN state
- **AND** the client SHALL reject subsequent requests immediately without calling server
- **AND** the client SHALL log circuit breaker state change: "Circuit breaker OPENED - inference server unhealthy"
- **AND** the client SHALL emit metric `circuit_breaker_state=2` (OPEN)

#### Scenario: Circuit breaker attempts recovery

- **GIVEN** a client's circuit breaker is OPEN
- **WHEN** 60 seconds have elapsed since last failure
- **THEN** the circuit breaker SHALL transition to HALF_OPEN state
- **AND** the client SHALL allow 1 test request to inference server
- **WHEN** 2 consecutive test requests succeed
- **THEN** the circuit breaker SHALL transition to CLOSED state (normal operation)
- **AND** the client SHALL log: "Circuit breaker CLOSED - inference server recovered"

### Requirement: Client Retry Policy

Client containers calling GPU inference servers SHALL implement **retry policy** with exponential backoff for transient failures.

**Retry Configuration**:

- **Max Attempts**: 3 (initial attempt + 2 retries)
- **Backoff**: Exponential with jitter (4s, 16s, 64s)
- **Retry Conditions**: Network errors, timeouts, HTTP 5xx (server errors)
- **No Retry**: HTTP 4xx (client errors, permanent failures)

#### Scenario: Client retries on timeout

- **GIVEN** a client makes inference request to server
- **WHEN** the request times out after 300 seconds
- **THEN** the client SHALL retry the request after 4 seconds (1st retry)
- **WHEN** the 1st retry also times out
- **THEN** the client SHALL retry after 16 seconds (2nd retry)
- **WHEN** the 2nd retry succeeds
- **THEN** the client SHALL return successful response

#### Scenario: Client does not retry on client error

- **GIVEN** a client makes inference request with invalid input
- **WHEN** the server returns HTTP 400 Bad Request
- **THEN** the client SHALL NOT retry the request
- **AND** the client SHALL raise validation error to caller

### Requirement: Distributed Tracing for Inference Requests

The system SHALL propagate **OpenTelemetry trace context** from clients to inference servers for distributed tracing.

#### Scenario: Client injects trace context into HTTP headers

- **GIVEN** a client is processing a request with trace ID `abc123`
- **WHEN** the client makes HTTP request to inference server
- **THEN** the client SHALL inject trace context into headers: `traceparent`, `tracestate`
- **AND** the client SHALL create span named `<service>.inference.<method>` (e.g., `mineru.vllm.chat_completion`)
- **AND** the span SHALL have attributes: `model`, `max_tokens`, `prompt_tokens`, `completion_tokens`

#### Scenario: Traces show end-to-end latency breakdown

- **GIVEN** a client made inference request that completed successfully
- **WHEN** viewing trace in Jaeger or Zipkin
- **THEN** the trace SHALL show parent span (client request processing)
- **AND** the trace SHALL show child span (inference server processing)
- **AND** the trace SHALL show latency breakdown: network time, queue time, inference time
- **AND** the trace SHALL enable root cause analysis for slow requests

## MODIFIED Requirements

### Requirement: GPU Service Fail-Fast Policy

GPU services SHALL fail fast if GPU resources are unavailable, with **split-container exceptions** for client services.

**Original Policy**: All GPU services must check GPU availability on startup and exit if unavailable.

**Updated Policy**:

- **Inference Server Containers**: MUST check GPU availability, exit if unavailable (fail-fast)
- **Client Containers**: MUST NOT check GPU availability (do not need GPU)
- **Client Startup**: MUST check inference server connectivity, exit if server unreachable

#### Scenario: Inference server fails fast without GPU

- **GIVEN** an inference server container (e.g., vLLM) is starting
- **WHEN** GPU is not available (`torch.cuda.is_available()` returns False)
- **THEN** the server SHALL log error: "GPU not available, refusing to start"
- **AND** the server SHALL exit with non-zero exit code
- **AND** the server SHALL NOT attempt CPU fallback

#### Scenario: Client container does not check GPU

- **GIVEN** a client container (e.g., MinerU worker) is starting
- **WHEN** the container initializes
- **THEN** the client SHALL NOT check GPU availability
- **AND** the client SHALL NOT import torch or CUDA libraries
- **AND** the client SHALL check inference server connectivity instead

#### Scenario: Client fails fast if inference server unreachable

- **GIVEN** a client container is starting
- **WHEN** the client attempts to connect to inference server
- **AND** the server is unreachable (connection refused or timeout)
- **THEN** the client SHALL log error: "Inference server unreachable at <url>"
- **AND** the client SHALL exit with non-zero exit code
- **AND** the client SHALL NOT enter ready state

### Requirement: GPU Resource Allocation

GPU resource requests SHALL be allocated to **inference server containers only**, not client containers, in split-container architectures.

**Original**: GPU services request GPU resources via `nvidia.com/gpu` in Kubernetes.

**Updated**:

- **Inference Servers**: Request `nvidia.com/gpu: 1` (or fractional)
- **Clients**: Do NOT request GPU resources (CPU and memory only)

#### Scenario: Inference server requests GPU in Kubernetes

- **GIVEN** an inference server deployment manifest
- **WHEN** the deployment is applied to Kubernetes
- **THEN** the pod spec SHALL include `resources.requests.nvidia.com/gpu: 1`
- **AND** the pod SHALL be scheduled only on GPU nodes
- **AND** the pod SHALL have GPU device visible via `nvidia-smi`

#### Scenario: Client does not request GPU in Kubernetes

- **GIVEN** a client container deployment manifest
- **WHEN** the deployment is applied to Kubernetes
- **THEN** the pod spec SHALL NOT include `resources.requests.nvidia.com/gpu`
- **AND** the pod SHALL be scheduled on any node (CPU-only or GPU nodes)
- **AND** the pod SHALL request CPU (e.g., `2 cores`) and memory (e.g., `4Gi`) only

## REMOVED Requirements

None (no existing requirements removed, only new requirements added and existing ones modified)

## RENAMED Requirements

None (no requirements renamed in this change)
