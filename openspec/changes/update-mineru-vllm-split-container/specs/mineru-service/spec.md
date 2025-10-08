# Spec Delta: MinerU Service (Split-Container Architecture)

## MODIFIED Requirements

### Requirement: MinerU Service Architecture

The MinerU service SHALL operate as a **lightweight HTTP client** that invokes a separate vLLM server for vision-language model inference, rather than loading models in-process.

**Architecture Pattern**: Split-container client/server model

**Components**:

1. **vLLM Server Container**: Dedicated GPU service running vision-language model (Qwen2.5-VL-7B-Instruct)
2. **MinerU Worker Containers**: Stateless CPU-bound workers that call vLLM server via HTTP

#### Scenario: Worker starts without GPU

- **GIVEN** a MinerU worker container is deployed
- **WHEN** the worker starts
- **THEN** the worker SHALL NOT require GPU resources (`nvidia.com/gpu` not requested)
- **AND** the worker SHALL start in <5 seconds (no model loading)
- **AND** the worker SHALL connect to vLLM server at configured URL

#### Scenario: Worker calls vLLM for VLM inference

- **GIVEN** a MinerU worker is processing a PDF
- **WHEN** the worker needs vision-language model inference
- **THEN** the worker SHALL make HTTP POST request to vLLM server `/v1/chat/completions` endpoint
- **AND** the request SHALL use OpenAI-compatible API format (messages with text + image_url)
- **AND** the worker SHALL receive inference results in OpenAI response format

#### Scenario: Multiple workers share single vLLM server

- **GIVEN** 8 MinerU workers are running
- **AND** 1 vLLM server is running
- **WHEN** multiple workers make concurrent VLM requests
- **THEN** the vLLM server SHALL batch requests using continuous batching
- **AND** all workers SHALL receive responses successfully
- **AND** GPU utilization SHALL exceed 85% (efficient batching)

### Requirement: MinerU CLI Backend Configuration

The MinerU service SHALL invoke the `mineru` CLI with **HTTP client backend** (`-b vlm-http-client`) pointing to the vLLM server URL.

#### Scenario: CLI invocation with HTTP client backend

- **GIVEN** a MinerU worker is configured with vLLM server URL `http://vllm-server:8000`
- **WHEN** the worker invokes the mineru CLI to process a PDF
- **THEN** the CLI command SHALL include `-b vlm-http-client -u http://vllm-server:8000`
- **AND** the CLI SHALL NOT include GPU-specific flags like `--gpu-memory-utilization`
- **AND** the worker SHALL NOT set `CUDA_VISIBLE_DEVICES` environment variable

#### Scenario: Worker fails fast if vLLM server unreachable

- **GIVEN** a MinerU worker is starting
- **WHEN** the worker attempts to connect to vLLM server
- **AND** the vLLM server is unreachable
- **THEN** the worker SHALL log error message with vLLM server URL
- **AND** the worker SHALL exit with non-zero exit code
- **AND** the worker SHALL NOT enter ready state

### Requirement: HTTP Client with Resilience

The MinerU service SHALL implement an HTTP client for vLLM communication with connection pooling, retry logic, and circuit breaker patterns.

#### Scenario: Connection pool reuse

- **GIVEN** a MinerU worker has established connection pool to vLLM server
- **WHEN** the worker makes multiple VLM requests
- **THEN** the worker SHALL reuse persistent HTTP connections (keep-alive)
- **AND** the connection pool SHALL support 10 max connections, 5 keepalive connections
- **AND** the worker SHALL NOT re-establish TCP handshake for each request

#### Scenario: Retry on transient failures

- **GIVEN** a MinerU worker makes VLM request to vLLM server
- **WHEN** the request fails with timeout or network error
- **THEN** the worker SHALL retry up to 3 times
- **AND** the worker SHALL use exponential backoff (4s, 16s, 64s with jitter)
- **AND** the worker SHALL NOT retry on HTTP 4xx errors (client error, permanent)

#### Scenario: Circuit breaker opens on repeated failures

- **GIVEN** a MinerU worker is making VLM requests
- **WHEN** 5 consecutive requests fail
- **THEN** the worker's circuit breaker SHALL transition to OPEN state
- **AND** the worker SHALL reject subsequent requests immediately without calling vLLM
- **AND** the worker SHALL log circuit breaker state change with failure count

#### Scenario: Circuit breaker recovery

- **GIVEN** a MinerU worker's circuit breaker is OPEN
- **WHEN** 60 seconds have elapsed since last failure
- **THEN** the circuit breaker SHALL transition to HALF_OPEN state
- **AND** the worker SHALL allow test request to vLLM server
- **WHEN** 2 consecutive test requests succeed
- **THEN** the circuit breaker SHALL transition to CLOSED state (normal operation)

## ADDED Requirements

### Requirement: vLLM Server Service

The system SHALL provide a **vLLM server service** that exposes an OpenAI-compatible HTTP API for vision-language model inference.

**Model**: `Qwen/Qwen2.5-VL-7B-Instruct` (7B parameter vision-language model)

**Endpoint**: `/v1/chat/completions` (OpenAI-compatible)

**Resource Requirements**:

- **GPU**: 1x NVIDIA RTX 5090 (32GB VRAM)
- **VRAM**: 16-24GB (model weights + KV cache)
- **CPU**: 4-8 cores
- **Memory**: 32GB RAM

#### Scenario: vLLM server starts and loads model

- **GIVEN** vLLM server container is deployed
- **WHEN** the server starts
- **THEN** the server SHALL load Qwen2.5-VL-7B-Instruct model from Hugging Face
- **AND** the server SHALL allocate 92% of GPU memory (29GB on RTX 5090)
- **AND** the server SHALL expose health check at `/health` endpoint
- **AND** the server SHALL start successfully within 60 seconds

#### Scenario: vLLM server handles chat completion request

- **GIVEN** vLLM server is running
- **WHEN** a client POSTs to `/v1/chat/completions` with multimodal messages (text + image)
- **THEN** the server SHALL perform VLM inference on GPU
- **AND** the server SHALL return OpenAI-compatible response with generated text
- **AND** the response SHALL include usage statistics (prompt_tokens, completion_tokens)
- **AND** the request SHALL complete within 5 seconds (P95 latency target)

#### Scenario: vLLM server uses continuous batching

- **GIVEN** vLLM server is running
- **WHEN** 4 concurrent chat completion requests arrive within 1 second
- **THEN** the server SHALL batch the requests for GPU processing
- **AND** the server SHALL process requests more efficiently than sequential processing
- **AND** GPU utilization SHALL exceed 85%

#### Scenario: vLLM server exposes Prometheus metrics

- **GIVEN** vLLM server is running
- **WHEN** Prometheus scrapes `/metrics` endpoint
- **THEN** the server SHALL expose metrics including:
  - `vllm:num_requests_running` (active requests)
  - `vllm:time_to_first_token_seconds` (latency)
  - `vllm:gpu_cache_usage_perc` (KV cache utilization)
  - `vllm:prompt_tokens_total` (total input tokens)
  - `vllm:generation_tokens_total` (total output tokens)

### Requirement: vLLM Server Health Check

The vLLM server SHALL expose a health check endpoint for liveness and readiness probes.

#### Scenario: Health check endpoint returns status

- **GIVEN** vLLM server is running
- **WHEN** a health check probe GETs `/health`
- **THEN** the server SHALL return HTTP 200 if healthy
- **AND** the response SHALL include JSON with model name and GPU memory usage
- **AND** the response SHALL be returned within 1 second

#### Scenario: Health check fails if GPU unavailable

- **GIVEN** vLLM server is starting
- **WHEN** GPU is not available (CUDA_VISIBLE_DEVICES empty or GPU offline)
- **THEN** health check SHALL return HTTP 503 Service Unavailable
- **AND** the server SHALL log error message indicating GPU unavailable
- **AND** the server SHALL NOT accept inference requests

### Requirement: Worker Metrics for vLLM Client

The MinerU service SHALL expose Prometheus metrics tracking vLLM HTTP client performance and circuit breaker state.

#### Scenario: Worker exposes vLLM request latency

- **GIVEN** a MinerU worker is making vLLM requests
- **WHEN** Prometheus scrapes worker `/metrics` endpoint
- **THEN** the worker SHALL expose histogram `mineru_vllm_request_duration_seconds`
- **AND** the histogram SHALL have labels: `worker_id`, `status` (success, timeout, error)
- **AND** the histogram SHALL have buckets: [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0] seconds

#### Scenario: Worker exposes circuit breaker state

- **GIVEN** a MinerU worker has a circuit breaker for vLLM client
- **WHEN** Prometheus scrapes worker `/metrics` endpoint
- **THEN** the worker SHALL expose gauge `mineru_vllm_circuit_breaker_state`
- **AND** the gauge value SHALL be 0 (CLOSED), 1 (HALF_OPEN), or 2 (OPEN)
- **AND** the gauge SHALL update immediately when circuit breaker state changes

#### Scenario: Worker exposes vLLM client failure counter

- **GIVEN** a MinerU worker makes vLLM requests that fail
- **WHEN** Prometheus scrapes worker `/metrics` endpoint
- **THEN** the worker SHALL expose counter `mineru_vllm_client_failures_total`
- **AND** the counter SHALL have labels: `worker_id`, `error_type` (timeout, network, http_4xx, http_5xx)
- **AND** the counter SHALL increment for each failed request

### Requirement: Distributed Tracing for vLLM Calls

The MinerU service SHALL propagate OpenTelemetry trace context from worker to vLLM server for distributed tracing.

#### Scenario: Worker propagates trace context to vLLM

- **GIVEN** a MinerU worker is processing a PDF with trace ID `abc123`
- **WHEN** the worker makes HTTP request to vLLM server
- **THEN** the worker SHALL inject trace context into HTTP headers (`traceparent`, `tracestate`)
- **AND** the vLLM span SHALL appear as child of worker span in trace visualization
- **AND** the trace SHALL show end-to-end latency breakdown (worker + vLLM inference)

#### Scenario: Worker records vLLM request span

- **GIVEN** a MinerU worker makes vLLM chat completion request
- **WHEN** the request is being traced
- **THEN** the worker SHALL create span named `mineru.vllm.chat_completion`
- **AND** the span SHALL have attributes: `vllm.model`, `vllm.max_tokens`, `vllm.prompt_tokens`, `vllm.completion_tokens`
- **AND** the span SHALL record exceptions if request fails

## REMOVED Requirements

### Requirement: In-Process vLLM Engine

**Removed**: The requirement for MinerU workers to load vLLM models in-process using `-b vlm-vllm-engine` backend.

**Reason**: Split-container architecture separates inference serving from PDF processing for better resource utilization, operational hygiene, and scalability.

**Migration**:

- Existing workers using in-process engine SHALL be migrated to HTTP client backend during deployment
- Configuration key `mineru.workers.backend` SHALL change from `vlm-vllm-engine` to `vlm-http-client`
- Workers SHALL no longer request GPU resources in Kubernetes deployments

### Requirement: GPU Resource Allocation per Worker

**Removed**: The requirement for each MinerU worker to reserve GPU resources (`nvidia.com/gpu: 0.25`) for in-process model inference.

**Reason**: Workers no longer perform GPU computation; GPU is centralized in vLLM server.

**Migration**:

- Kubernetes deployment specs SHALL remove `resources.requests.nvidia.com/gpu` from worker containers
- GPU resources SHALL be requested only by vLLM server container
- Worker count can increase from 4 to 8+ since GPU is no longer limiting factor

### Requirement: Worker Startup Model Loading

**Removed**: The requirement for MinerU workers to load VLM model weights during startup (60-second initialization).

**Reason**: Workers no longer load models; vLLM server loads model once on startup.

**Migration**:

- Worker startup time SHALL decrease from 60 seconds to <5 seconds
- Kubernetes readiness probes SHALL use shorter `initialDelaySeconds` (5s instead of 60s)
- Workers SHALL check vLLM server connectivity instead of GPU availability

## RENAMED Requirements

None (no requirements renamed in this change)
