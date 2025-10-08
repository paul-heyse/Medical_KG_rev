# Spec Delta: Orchestration (Split-Container Worker Management)

## MODIFIED Requirements

### Requirement: Worker Pool Configuration

The orchestration system SHALL manage worker pools with configuration optimized for split-container architecture, where workers are **CPU-bound** rather than **GPU-bound**.

**Key Changes**:

- **Worker Count**: Increased from 4 to 8-12 workers (no longer limited by GPU)
- **Resource Requests**: CPU and memory only (no GPU)
- **Startup Time**: <5 seconds (no model loading)
- **Scaling Strategy**: Scale based on CPU and Kafka consumer lag, not GPU availability

#### Scenario: Worker pool scales based on CPU utilization

- **GIVEN** a worker pool is deployed with 8 workers
- **WHEN** Kafka consumer lag exceeds 100 messages for 5 minutes
- **THEN** the orchestration system SHALL scale workers to 12 replicas
- **AND** workers SHALL NOT require additional GPU resources (CPU-bound)
- **AND** new workers SHALL start within 10 seconds (<5s each + scheduling)

#### Scenario: Worker pool scales independently of inference server

- **GIVEN** a worker pool is deployed with 8 workers
- **AND** a vLLM inference server is deployed with 1 GPU
- **WHEN** administrators scale workers from 8 to 16
- **THEN** workers SHALL scale successfully without modifying inference server
- **AND** inference server SHALL handle increased load via request batching
- **AND** workers SHALL share single hot model in inference server

### Requirement: Worker Startup Health Checks

Worker containers SHALL perform **inference server connectivity checks** on startup instead of GPU availability checks.

**Original**: Workers check `torch.cuda.is_available()` on startup.

**Updated**: Workers check HTTP connectivity to inference server `/health` endpoint on startup.

#### Scenario: Worker checks inference server connectivity on startup

- **GIVEN** a worker container is starting
- **WHEN** the worker initializes
- **THEN** the worker SHALL make HTTP GET request to `<INFERENCE_SERVER_URL>/health`
- **AND** the worker SHALL wait up to 10 seconds for response
- **WHEN** the server returns HTTP 200
- **THEN** the worker SHALL enter ready state
- **WHEN** the server is unreachable or returns error
- **THEN** the worker SHALL exit with code 1 and log error message

#### Scenario: Worker startup faster without model loading

- **GIVEN** a worker container is starting
- **WHEN** the worker initializes (no model weights to load)
- **THEN** the worker SHALL reach ready state within 5 seconds
- **AND** Kubernetes readiness probe SHALL pass after 5 seconds
- **AND** the worker SHALL be added to service load balancer pool

### Requirement: Worker Environment Configuration

Worker containers SHALL be configured with environment variables for inference server connectivity instead of GPU settings.

**Original Environment Variables**:

- `CUDA_VISIBLE_DEVICES`: GPU device assignment
- `GPU_MEMORY_FRACTION`: GPU memory limit per worker

**New Environment Variables**:

- `INFERENCE_SERVER_URL`: Inference server base URL (e.g., `http://vllm-server:8000`)
- `INFERENCE_SERVER_MODEL`: Model name for validation
- `HTTP_CLIENT_TIMEOUT`: Request timeout in seconds (default: 300)
- `HTTP_CLIENT_RETRIES`: Max retry attempts (default: 3)
- `CIRCUIT_BREAKER_ENABLED`: Enable circuit breaker (default: true)

#### Scenario: Worker reads inference server URL from environment

- **GIVEN** a worker container is deployed with env var `INFERENCE_SERVER_URL=http://vllm-server:8000`
- **WHEN** the worker initializes HTTP client
- **THEN** the worker SHALL use `http://vllm-server:8000` as base URL for inference requests
- **AND** the worker SHALL resolve DNS `vllm-server` to Kubernetes service ClusterIP

#### Scenario: Worker uses default settings if env vars not set

- **GIVEN** a worker container is deployed without `INFERENCE_SERVER_URL` env var
- **WHEN** the worker initializes
- **THEN** the worker SHALL use default URL `http://vllm-server:8000`
- **AND** the worker SHALL log warning: "Using default inference server URL"

### Requirement: Worker Deployment Strategy

The orchestration system SHALL support **rolling deployment** for workers with minimal disruption, leveraging fast startup times.

**Strategy**:

- **RollingUpdate**: Deploy new workers gradually, drain old workers
- **MaxSurge**: 25% (2 extra workers during rollout for 8-worker pool)
- **MaxUnavailable**: 0% (always maintain capacity)

**Rationale**: Fast worker startup (<5s) enables zero-downtime deployments with surge capacity.

#### Scenario: Rolling deployment of workers with zero downtime

- **GIVEN** a worker pool with 8 workers is deployed (version v1)
- **WHEN** administrators deploy new version v2 with rolling update strategy
- **THEN** the orchestration system SHALL create 2 new v2 workers (maxSurge: 25%)
- **AND** the system SHALL wait for v2 workers to become ready (<10s)
- **AND** the system SHALL terminate 2 old v1 workers
- **AND** the system SHALL repeat until all 8 workers are v2
- **AND** at least 8 workers SHALL be available throughout rollout (maxUnavailable: 0)

#### Scenario: Worker drains in-flight requests before termination

- **GIVEN** a worker is processing 3 PDFs (in-flight requests)
- **WHEN** the orchestration system sends termination signal (SIGTERM)
- **THEN** the worker SHALL stop consuming new messages from Kafka
- **AND** the worker SHALL complete processing of 3 in-flight PDFs
- **AND** the worker SHALL ACK Kafka messages for completed PDFs
- **WHEN** all in-flight requests are completed
- **THEN** the worker SHALL exit gracefully

## ADDED Requirements

### Requirement: Inference Server Service Discovery

The orchestration system SHALL provide **Kubernetes Service** for inference server with DNS-based discovery for workers.

**Service Type**: `ClusterIP` (internal cluster access only)

**Service Name**: Standardized naming (e.g., `vllm-server`, `embedding-server`)

#### Scenario: Workers discover inference server via Kubernetes Service

- **GIVEN** an inference server Kubernetes Service named `vllm-server` is deployed
- **AND** workers are configured with `INFERENCE_SERVER_URL=http://vllm-server:8000`
- **WHEN** a worker makes HTTP request to inference server
- **THEN** the worker SHALL resolve DNS name `vllm-server` to Service ClusterIP
- **AND** Kubernetes Service SHALL load balance requests to inference server pods (if multiple)
- **AND** the worker SHALL successfully reach inference server without hardcoded IP

#### Scenario: Service health check filters unhealthy pods

- **GIVEN** an inference server Service with 2 pods (A and B)
- **WHEN** pod A fails health check (returns HTTP 503)
- **THEN** the Service SHALL remove pod A from endpoints
- **AND** workers making requests SHALL only reach healthy pod B
- **WHEN** pod A recovers (health check returns HTTP 200)
- **THEN** the Service SHALL add pod A back to endpoints

### Requirement: Worker-to-Inference-Server Network Policy

The orchestration system SHALL deploy **Kubernetes NetworkPolicy** to restrict network access between workers and inference server.

**Policy**:

- Inference server SHALL accept ingress traffic ONLY from worker pods
- Inference server SHALL NOT be accessible from external networks or other namespaces
- Workers SHALL have egress access to inference server on port 8000

#### Scenario: Network policy restricts inference server access

- **GIVEN** a NetworkPolicy is deployed for inference server
- **WHEN** a worker pod makes HTTP request to inference server on port 8000
- **THEN** the request SHALL succeed (worker is allowed)
- **WHEN** a pod from different namespace attempts to access inference server
- **THEN** the request SHALL be blocked by NetworkPolicy
- **AND** the request SHALL timeout without reaching inference server

#### Scenario: Network policy allows worker egress to inference server

- **GIVEN** a NetworkPolicy is deployed for workers
- **WHEN** a worker pod makes HTTP request to `vllm-server:8000`
- **THEN** the request SHALL be allowed by egress rules
- **WHEN** a worker pod attempts to access external internet (e.g., `google.com`)
- **THEN** the request MAY be allowed or blocked based on general cluster policy (not specific to this NetworkPolicy)

### Requirement: Worker Metrics for Orchestration

The orchestration system SHALL collect **worker-level metrics** for scaling decisions and operational monitoring.

**Metrics for Autoscaling**:

- Kafka consumer lag (messages behind)
- Worker CPU utilization
- Worker memory usage
- Inference request queue depth (per worker)

#### Scenario: Orchestration monitors Kafka consumer lag

- **GIVEN** workers are consuming from Kafka topic `pdf.parse.requests.v1`
- **WHEN** Prometheus scrapes metrics
- **THEN** the orchestration system SHALL expose metric `kafka_consumergroup_lag{topic="pdf.parse.requests.v1", group="mineru-workers"}`
- **AND** the metric SHALL show number of messages workers are behind

#### Scenario: HorizontalPodAutoscaler scales workers based on lag

- **GIVEN** an HPA is configured with target: `kafka_consumergroup_lag < 50`
- **WHEN** consumer lag exceeds 50 messages for 2 minutes
- **THEN** the HPA SHALL scale workers up (increase replicas by 25%)
- **WHEN** consumer lag drops below 50 messages
- **THEN** the HPA SHALL scale workers down after cooldown period (5 minutes)

### Requirement: Inference Server Dedicated Node Pool (Optional)

The orchestration system SHALL support deployment of inference servers to **dedicated GPU node pool** with taints and tolerations for isolation when configured.

**Use Case**: Separate GPU workloads from CPU workloads for cost optimization and resource isolation.

**Configuration**:

- **Node Taint**: `nvidia.com/gpu=present:NoSchedule` (only pods with matching toleration can schedule)
- **Node Label**: `accelerator=nvidia-rtx-5090` (for node selection)
- **Inference Server Toleration**: Match taint to allow scheduling on GPU nodes
- **Worker Toleration**: None (workers stay on CPU nodes)

#### Scenario: Inference server scheduled on GPU node pool

- **GIVEN** GPU nodes are tainted with `nvidia.com/gpu=present:NoSchedule`
- **AND** inference server pod has toleration for taint
- **WHEN** inference server pod is deployed
- **THEN** the pod SHALL be scheduled on GPU node (ignores taint)
- **AND** the pod SHALL have access to GPU device

#### Scenario: Workers scheduled on CPU node pool

- **GIVEN** GPU nodes are tainted with `nvidia.com/gpu=present:NoSchedule`
- **AND** worker pods have NO toleration for taint
- **WHEN** worker pods are deployed
- **THEN** the pods SHALL be scheduled on CPU-only nodes
- **AND** the pods SHALL NOT be scheduled on GPU nodes (respect taint)

### Requirement: Graceful Inference Server Restart

The orchestration system SHALL support **zero-downtime restart** of inference server by coordinating with workers via circuit breakers.

**Procedure**:

1. Deploy new inference server pod (rolling update)
2. Wait for new pod to become ready (health check passes)
3. Terminate old pod with grace period (60 seconds)
4. Workers detect old pod unavailability, circuit breaker opens
5. Workers retry requests, discover new pod via Service
6. Circuit breaker closes when requests succeed

#### Scenario: Inference server rolling restart with worker resilience

- **GIVEN** workers are processing PDFs with inference server pod A
- **WHEN** administrators deploy new inference server pod B (rolling update)
- **THEN** pod B SHALL start and become ready (health check passes)
- **AND** Kubernetes Service SHALL add pod B to endpoints
- **WHEN** administrators terminate pod A
- **THEN** workers' requests to pod A SHALL fail (connection refused)
- **AND** workers' circuit breakers SHALL open after 5 failures
- **AND** workers SHALL retry requests after 60 seconds
- **WHEN** workers retry
- **THEN** Kubernetes Service SHALL route requests to pod B
- **AND** workers' circuit breakers SHALL close after 2 successes

## REMOVED Requirements

None (no existing requirements removed, only new requirements added and existing ones modified)

## RENAMED Requirements

None (no requirements renamed in this change)
