# Implementation Tasks: MinerU vLLM Split-Container Architecture

## Task Organization

Tasks are organized into implementation phases for AI agents to execute. Each task includes precise technical specifications, acceptance criteria, file paths, and validation commands.

**Total Tasks**: 180 implementation tasks

**Legend**:

- `[ ]` - Not started
- `[~]` - In progress
- `[x]` - Completed
- `[!]` - Blocked

**Context**: This software is NOT in production. Migrate aggressively with no concern for legacy infrastructure. All tasks executed by AI agents.

**Reference Documents**:

- **DEPLOYMENT_STRATEGY.md**: Container configurations and deployment specs
- **TESTING_PLAN.md**: Test specifications and validation criteria
- **SECURITY.md**: Security controls implementation specs
- **OPERATIONS.md**: Monitoring and observability setup
- **design.md**: Technical architecture and component specifications

---

## Phase 1: vLLM Server Infrastructure Implementation

### 1.1 Docker Configuration

- [x] **1.1.1** Create vLLM server Docker Compose configuration
  - **File**: `docker-compose.vllm.yml`
  - **Specification**:

    ```yaml
    version: '3.8'
    services:
      vllm-server:
        image: vllm/vllm-openai:v0.11.0
        command: >
          python -m vllm.entrypoints.openai.api_server
            --model Qwen/Qwen2.5-VL-7B-Instruct
            --host 0.0.0.0
            --port 8000
            --gpu-memory-utilization 0.92
            --max-model-len 32768
            --tensor-parallel-size 1
            --dtype auto
            --enforce-eager
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu]
        ports:
          - "8000:8000"
        volumes:
          - ~/.cache/huggingface:/root/.cache/huggingface:ro
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 120s
        ipc: host
        ulimits:
          memlock: -1
          stack: 67108864
        networks:
          - medical-kg-net
    networks:
      medical-kg-net:
        external: true
    ```

  - **Validation**: `docker-compose -f docker-compose.vllm.yml up -d && curl http://localhost:8000/health`

- [x] **1.1.2** Add vLLM server to main docker-compose.yml
  - **File**: `docker-compose.yml`
  - **Action**: Add vLLM server service definition from above, ensure it connects to `medical-kg-net`
  - **Validation**: `docker-compose up -d vllm-server && docker ps | grep vllm-server`

- [x] **1.1.3** Create vLLM server test script
  - **File**: `scripts/test_vllm_api.sh`
  - **Specification**:

    ```bash
    #!/bin/bash
    set -e

    VLLM_URL="${VLLM_SERVER_URL:-http://localhost:8000}"

    # Test health endpoint
    curl -f "${VLLM_URL}/health" || exit 1

    # Test OpenAI-compatible chat completion
    curl -X POST "${VLLM_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 50,
        "temperature": 0.0
      }' | jq '.choices[0].message.content'
    ```

  - **Validation**: `bash scripts/test_vllm_api.sh`

### 1.2 Kubernetes Configuration

- [x] **1.2.1** Create vLLM server Deployment manifest
  - **File**: `ops/k8s/base/deployment-vllm-server.yaml`
  - **Specification**:

    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-server
      namespace: medical-kg
      labels:
        app: vllm-server
        component: gpu-inference
    spec:
      replicas: 1
      strategy:
        type: Recreate
      selector:
        matchLabels:
          app: vllm-server
      template:
        metadata:
          labels:
            app: vllm-server
            component: gpu-inference
        spec:
          nodeSelector:
            accelerator: nvidia-gpu
          tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
          containers:
          - name: vllm-server
            image: vllm/vllm-openai:v0.11.0
            command:
            - python
            - -m
            - vllm.entrypoints.openai.api_server
            args:
            - --model=Qwen/Qwen2.5-VL-7B-Instruct
            - --host=0.0.0.0
            - --port=8000
            - --gpu-memory-utilization=0.92
            - --max-model-len=32768
            - --tensor-parallel-size=1
            - --dtype=auto
            - --enforce-eager
            ports:
            - name: http
              containerPort: 8000
              protocol: TCP
            env:
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
            - name: HF_HOME
              value: /cache/huggingface
            resources:
              requests:
                nvidia.com/gpu: 1
                memory: 32Gi
                cpu: 4
              limits:
                nvidia.com/gpu: 1
                memory: 48Gi
                cpu: 8
            volumeMounts:
            - name: huggingface-cache
              mountPath: /cache/huggingface
              readOnly: true
            livenessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 120
              periodSeconds: 30
              timeoutSeconds: 10
              failureThreshold: 3
            readinessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 30
              periodSeconds: 10
              timeoutSeconds: 5
              successThreshold: 1
              failureThreshold: 3
          volumes:
          - name: huggingface-cache
            persistentVolumeClaim:
              claimName: huggingface-cache-pvc
    ```

  - **Validation**: `kubectl apply -f ops/k8s/base/deployment-vllm-server.yaml --dry-run=client`

- [x] **1.2.2** Create vLLM server Service manifest
  - **File**: `ops/k8s/base/service-vllm-server.yaml`
  - **Specification**:

    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-server
      namespace: medical-kg
      labels:
        app: vllm-server
    spec:
      type: ClusterIP
      ports:
      - name: http
        port: 8000
        targetPort: 8000
        protocol: TCP
      selector:
        app: vllm-server
    ```

  - **Validation**: `kubectl apply -f ops/k8s/base/service-vllm-server.yaml --dry-run=client`

- [x] **1.2.3** Create vLLM server ConfigMap
  - **File**: `ops/k8s/base/configmap-vllm-server.yaml`
  - **Specification**:

    ```yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: vllm-server-config
      namespace: medical-kg
    data:
      model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
      max_model_len: "32768"
      gpu_memory_utilization: "0.92"
      tensor_parallel_size: "1"
      port: "8000"
    ```

  - **Validation**: `kubectl apply -f ops/k8s/base/configmap-vllm-server.yaml --dry-run=client`

- [x] **1.2.4** Create NetworkPolicy for vLLM server
  - **File**: `ops/k8s/base/networkpolicy-vllm-server.yaml`
  - **Specification**:

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
              app: mineru-worker
        ports:
        - protocol: TCP
          port: 8000
    ```

  - **Validation**: `kubectl apply -f ops/k8s/base/networkpolicy-vllm-server.yaml --dry-run=client`

- [x] **1.2.5** Create PersistentVolumeClaim for HuggingFace cache
  - **File**: `ops/k8s/base/pvc-huggingface-cache.yaml`
  - **Specification**:

    ```yaml
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: huggingface-cache-pvc
      namespace: medical-kg
    spec:
      accessModes:
      - ReadOnlyMany
      resources:
        requests:
          storage: 50Gi
      storageClassName: fast-ssd
    ```

  - **Validation**: `kubectl apply -f ops/k8s/base/pvc-huggingface-cache.yaml --dry-run=client`

### 1.3 Monitoring Configuration

- [x] **1.3.1** Create Prometheus ServiceMonitor for vLLM
  - **File**: `ops/k8s/base/servicemonitor-vllm-server.yaml`
  - **Specification**:

    ```yaml
    apiVersion: monitoring.coreos.com/v1
    kind: ServiceMonitor
    metadata:
      name: vllm-server
      namespace: medical-kg
      labels:
        app: vllm-server
    spec:
      selector:
        matchLabels:
          app: vllm-server
      endpoints:
      - port: http
        path: /metrics
        interval: 15s
        scrapeTimeout: 10s
    ```

  - **Validation**: `kubectl apply -f ops/k8s/base/servicemonitor-vllm-server.yaml --dry-run=client`

- [x] **1.3.2** Create Grafana dashboard for vLLM server
  - **File**: `ops/monitoring/grafana/dashboards/vllm-server.json`
  - **Specification**: Dashboard with panels for:
    - GPU utilization (gauge, 0-100%)
    - GPU memory usage (gauge, GB)
    - Request queue depth (graph)
    - Request latency (histogram, P50/P95/P99)
    - Throughput (requests/second)
    - Error rate (percentage)
    - Model loading status (stat)
    - KV cache usage (gauge)
  - **Validation**: Import dashboard into Grafana, verify panels render

- [x] **1.3.3** Create Prometheus alerting rules for vLLM
  - **File**: `ops/monitoring/alerts-vllm.yml`
  - **Specification**:

    ```yaml
    groups:
    - name: vllm-server
      interval: 30s
      rules:
      - alert: VLLMServerDown
        expr: up{job="vllm-server"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "vLLM server is down"
          description: "vLLM server has been down for more than 2 minutes"

      - alert: VLLMHighLatency
        expr: histogram_quantile(0.95, rate(vllm_request_duration_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "vLLM P95 latency is high"
          description: "P95 latency is {{ $value }}s (threshold: 10s)"

      - alert: VLLMGPUMemoryHigh
        expr: vllm_gpu_memory_usage_bytes / vllm_gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "vLLM GPU memory usage is high"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"

      - alert: VLLMHighErrorRate
        expr: rate(vllm_request_errors_total[5m]) / rate(vllm_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "vLLM error rate is high"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      - alert: VLLMGPUOOM
        expr: increase(vllm_gpu_oom_total[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "vLLM GPU out of memory"
          description: "GPU OOM detected {{ $value }} times in last 5 minutes"
    ```

  - **Validation**: `promtool check rules ops/monitoring/alerts-vllm.yml`

---

## Phase 2: Worker HTTP Client Implementation

### 2.1 VLLMClient Core Implementation

- [x] **2.1.1** Create VLLMClient class
  - **File**: `src/Medical_KG_rev/services/mineru/vllm_client.py`
  - **Specification**:

    ```python
    """HTTP client for vLLM OpenAI-compatible server."""
    import base64
    from typing import Any

    import httpx
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )

    from Medical_KG_rev.observability.logging import get_logger
    from Medical_KG_rev.observability.metrics import (
        MINERU_VLLM_REQUEST_DURATION,
        MINERU_VLLM_CLIENT_FAILURES,
        MINERU_VLLM_CLIENT_RETRIES,
    )

    logger = get_logger(__name__)


    class VLLMClientError(Exception):
        """Base exception for vLLM client errors."""
        pass


    class VLLMTimeoutError(VLLMClientError):
        """Timeout connecting to or waiting for vLLM server."""
        pass


    class VLLMServerError(VLLMClientError):
        """vLLM server returned an error response."""
        pass


    class VLLMClient:
        """Async HTTP client for vLLM OpenAI-compatible API."""

        def __init__(
            self,
            base_url: str,
            timeout: float = 300.0,
            max_connections: int = 10,
            max_keepalive_connections: int = 5,
        ):
            self.base_url = base_url.rstrip("/")
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(timeout),
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive_connections,
                ),
            )
            logger.info("VLLMClient initialized", base_url=base_url)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.client.aclose()

        @staticmethod
        def encode_image_base64(image_bytes: bytes) -> str:
            """Encode image bytes to base64 string."""
            return base64.b64encode(image_bytes).decode("utf-8")

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
            reraise=True,
        )
        async def chat_completion(
            self,
            messages: list[dict[str, Any]],
            max_tokens: int = 4096,
            temperature: float = 0.0,
            model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        ) -> dict[str, Any]:
            """
            Call vLLM OpenAI-compatible chat completion endpoint.

            Args:
                messages: OpenAI format messages with role and content
                max_tokens: Maximum tokens to generate
                temperature: Sampling temperature (0.0 = deterministic)
                model: Model name (must match vLLM server model)

            Returns:
                OpenAI format response with choices[0].message.content

            Raises:
                VLLMTimeoutError: Request timeout
                VLLMServerError: Server returned error response
            """
            try:
                with MINERU_VLLM_REQUEST_DURATION.time():
                    response = await self.client.post(
                        "/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    )

                response.raise_for_status()
                data = response.json()

                logger.debug(
                    "vLLM request succeeded",
                    status_code=response.status_code,
                    tokens=data.get("usage", {}).get("total_tokens"),
                )

                return data

            except httpx.TimeoutException as e:
                MINERU_VLLM_CLIENT_FAILURES.labels(error_type="timeout").inc()
                logger.error("vLLM request timeout", error=str(e))
                raise VLLMTimeoutError(f"Request timeout: {e}") from e

            except httpx.HTTPStatusError as e:
                MINERU_VLLM_CLIENT_FAILURES.labels(
                    error_type=f"http_{e.response.status_code}"
                ).inc()
                logger.error(
                    "vLLM server error",
                    status_code=e.response.status_code,
                    response=e.response.text,
                )
                raise VLLMServerError(
                    f"Server error {e.response.status_code}: {e.response.text}"
                ) from e

            except Exception as e:
                MINERU_VLLM_CLIENT_FAILURES.labels(error_type="unknown").inc()
                logger.error("vLLM request failed", error=str(e))
                raise VLLMClientError(f"Request failed: {e}") from e

        async def health_check(self) -> bool:
            """Check if vLLM server is healthy."""
            try:
                response = await self.client.get("/health", timeout=5.0)
                return response.status_code == 200
            except Exception as e:
                logger.error("vLLM health check failed", error=str(e))
                return False
    ```

  - **Validation**: `python -m pytest tests/services/mineru/test_vllm_client.py -v`

- [x] **2.1.2** Create Prometheus metrics for VLLMClient
  - **File**: `src/Medical_KG_rev/observability/metrics.py`
  - **Action**: Add these metrics to existing metrics module:

    ```python
    from prometheus_client import Counter, Histogram, Gauge

    MINERU_VLLM_REQUEST_DURATION = Histogram(
        'mineru_vllm_request_duration_seconds',
        'Duration of vLLM API requests',
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
    )

    MINERU_VLLM_CLIENT_FAILURES = Counter(
        'mineru_vllm_client_failures_total',
        'Total number of vLLM client failures',
        ['error_type']
    )

    MINERU_VLLM_CLIENT_RETRIES = Counter(
        'mineru_vllm_client_retries_total',
        'Total number of vLLM client retry attempts',
        ['retry_number']
    )

    MINERU_VLLM_CIRCUIT_BREAKER_STATE = Gauge(
        'mineru_vllm_circuit_breaker_state',
        'Circuit breaker state (0=closed, 1=half_open, 2=open)'
    )
    ```

  - **Validation**: Metrics appear in `/metrics` endpoint

### 2.2 Circuit Breaker Implementation

- [x] **2.2.1** Create CircuitBreaker class
  - **File**: `src/Medical_KG_rev/services/mineru/circuit_breaker.py`
  - **Specification**:

    ```python
    """Circuit breaker for vLLM client resilience."""
    import asyncio
    from datetime import datetime, timedelta
    from enum import Enum
    from typing import Callable, Any

    from Medical_KG_rev.observability.logging import get_logger
    from Medical_KG_rev.observability.metrics import MINERU_VLLM_CIRCUIT_BREAKER_STATE

    logger = get_logger(__name__)


    class CircuitState(Enum):
        """Circuit breaker states."""
        CLOSED = 0  # Normal operation
        HALF_OPEN = 1  # Testing if service recovered
        OPEN = 2  # Service unavailable, rejecting requests


    class CircuitBreakerOpenError(Exception):
        """Raised when circuit breaker is open."""
        pass


    class CircuitBreaker:
        """
        Circuit breaker pattern implementation for vLLM client.

        State transitions:
        - CLOSED → OPEN: After failure_threshold consecutive failures
        - OPEN → HALF_OPEN: After recovery_timeout seconds
        - HALF_OPEN → CLOSED: After success_threshold consecutive successes
        - HALF_OPEN → OPEN: On any failure
        """

        def __init__(
            self,
            failure_threshold: int = 5,
            recovery_timeout: float = 60.0,
            success_threshold: int = 2,
        ):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.success_threshold = success_threshold

            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time: datetime | None = None

            self._lock = asyncio.Lock()

            MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
            logger.info("Circuit breaker initialized", state=self.state.name)

        async def can_execute(self) -> bool:
            """Check if request can be executed."""
            async with self._lock:
                if self.state == CircuitState.CLOSED:
                    return True

                if self.state == CircuitState.OPEN:
                    if self.last_failure_time is None:
                        return False

                    elapsed = datetime.utcnow() - self.last_failure_time
                    if elapsed >= timedelta(seconds=self.recovery_timeout):
                        self._transition_to_half_open()
                        return True

                    return False

                # HALF_OPEN state
                return True

        async def record_success(self) -> None:
            """Record successful request."""
            async with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    logger.debug(
                        "Circuit breaker success recorded",
                        success_count=self.success_count,
                        threshold=self.success_threshold,
                    )

                    if self.success_count >= self.success_threshold:
                        self._transition_to_closed()

                elif self.state == CircuitState.CLOSED:
                    self.failure_count = 0  # Reset on success

        async def record_failure(self) -> None:
            """Record failed request."""
            async with self._lock:
                self.last_failure_time = datetime.utcnow()

                if self.state == CircuitState.CLOSED:
                    self.failure_count += 1
                    logger.warning(
                        "Circuit breaker failure recorded",
                        failure_count=self.failure_count,
                        threshold=self.failure_threshold,
                    )

                    if self.failure_count >= self.failure_threshold:
                        self._transition_to_open()

                elif self.state == CircuitState.HALF_OPEN:
                    logger.warning("Circuit breaker failure in HALF_OPEN, reopening")
                    self._transition_to_open()

        def _transition_to_open(self) -> None:
            """Transition to OPEN state."""
            self.state = CircuitState.OPEN
            self.success_count = 0
            MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
            logger.error(
                "Circuit breaker OPENED",
                failure_count=self.failure_count,
                recovery_timeout=self.recovery_timeout,
            )

        def _transition_to_half_open(self) -> None:
            """Transition to HALF_OPEN state."""
            self.state = CircuitState.HALF_OPEN
            self.failure_count = 0
            self.success_count = 0
            MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
            logger.info("Circuit breaker transitioned to HALF_OPEN")

        def _transition_to_closed(self) -> None:
            """Transition to CLOSED state."""
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            MINERU_VLLM_CIRCUIT_BREAKER_STATE.set(self.state.value)
            logger.info("Circuit breaker CLOSED, service recovered")
    ```

  - **Validation**: `python -m pytest tests/services/mineru/test_circuit_breaker.py -v`

- [x] **2.2.2** Integrate CircuitBreaker with VLLMClient
  - **File**: `src/Medical_KG_rev/services/mineru/vllm_client.py`
  - **Action**: Add circuit breaker to VLLMClient:

    ```python
    from Medical_KG_rev.services.mineru.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerOpenError,
    )

    class VLLMClient:
        def __init__(self, base_url: str, ...):
            # ... existing init code ...
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
                success_threshold=2,
            )

        async def chat_completion(self, ...):
            # Check circuit breaker before request
            if not await self.circuit_breaker.can_execute():
                raise CircuitBreakerOpenError(
                    "Circuit breaker is open, vLLM server unavailable"
                )

            try:
                # ... existing request code ...
                result = await self._make_request(...)
                await self.circuit_breaker.record_success()
                return result

            except (VLLMTimeoutError, VLLMServerError) as e:
                await self.circuit_breaker.record_failure()
                raise
    ```

  - **Validation**: `python -m pytest tests/services/mineru/test_vllm_client_circuit_breaker.py -v`

---

## Phase 3: MinerU Worker Update

### 3.1 Configuration Implementation

- [x] **3.1.1** Update mineru configuration file
  - **File**: `config/mineru.yaml`
  - **Specification**:

    ```yaml
    mineru:
      deployment_mode: "split-container"  # CHANGED: was "monolithic"

      vllm_server:
        enabled: true
        base_url: "http://vllm-server:8000"
        model: "Qwen/Qwen2.5-VL-7B-Instruct"
        health_check_interval_seconds: 30
        connection_timeout_seconds: 300

      workers:
        count: 8  # CHANGED: was 4 (no longer GPU-bound)
        backend: "vlm-http-client"  # CHANGED: was "vlm-vllm-engine"
        cpu_per_worker: 2
        memory_per_worker_gb: 4
        batch_size: 4
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
          success_threshold: 2
    ```

  - **Validation**: Config loads without errors

- [x] **3.1.2** Create Pydantic config models
  - **File**: `src/Medical_KG_rev/models/config/mineru.py`
  - **Specification**:

    ```python
    """Pydantic models for MinerU configuration."""
    from pydantic import BaseModel, Field, HttpUrl


    class VLLMServerConfig(BaseModel):
        """vLLM server configuration."""
        enabled: bool = True
        base_url: HttpUrl = Field(default="http://vllm-server:8000")
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
        health_check_interval_seconds: int = Field(default=30, ge=10)
        connection_timeout_seconds: int = Field(default=300, ge=30)


    class CircuitBreakerConfig(BaseModel):
        """Circuit breaker configuration."""
        enabled: bool = True
        failure_threshold: int = Field(default=5, ge=1)
        recovery_timeout_seconds: int = Field(default=60, ge=10)
        success_threshold: int = Field(default=2, ge=1)


    class HTTPClientConfig(BaseModel):
        """HTTP client configuration."""
        connection_pool_size: int = Field(default=10, ge=1)
        keepalive_connections: int = Field(default=5, ge=1)
        timeout_seconds: int = Field(default=300, ge=30)
        retry_attempts: int = Field(default=3, ge=0)
        retry_backoff_multiplier: float = Field(default=1.0, ge=0.1)
        circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)


    class WorkersConfig(BaseModel):
        """Worker pool configuration."""
        count: int = Field(default=8, ge=1)
        backend: str = Field(default="vlm-http-client")
        cpu_per_worker: int = Field(default=2, ge=1)
        memory_per_worker_gb: int = Field(default=4, ge=2)
        batch_size: int = Field(default=4, ge=1)
        timeout_seconds: int = Field(default=300, ge=30)


    class MinerUConfig(BaseModel):
        """Complete MinerU configuration."""
        deployment_mode: str = Field(default="split-container")
        vllm_server: VLLMServerConfig = Field(default_factory=VLLMServerConfig)
        workers: WorkersConfig = Field(default_factory=WorkersConfig)
        http_client: HTTPClientConfig = Field(default_factory=HTTPClientConfig)
    ```

  - **Validation**: `python -c "from Medical_KG_rev.models.config.mineru import MinerUConfig; print(MinerUConfig())"`

### 3.2 Worker Service Update

- [x] **3.2.1** Update MinerU service to use HTTP client
  - **File**: `src/Medical_KG_rev/services/mineru/service.py`
  - **Action**: Replace GPU initialization with VLLMClient:

    ```python
    # REMOVE these imports:
    # import torch
    # from vllm import LLM

    # ADD these imports:
    from Medical_KG_rev.services.mineru.vllm_client import VLLMClient, VLLMClientError
    from Medical_KG_rev.config.settings import get_settings

    class MinerUService:
        """MinerU PDF processing service using HTTP client."""

        def __init__(self):
            settings = get_settings()

            # REMOVE GPU initialization:
            # if not torch.cuda.is_available():
            #     raise RuntimeError("GPU not available")

            # ADD vLLM client initialization:
            self.vllm_client = VLLMClient(
                base_url=str(settings.mineru.vllm_server.base_url),
                timeout=settings.mineru.http_client.timeout_seconds,
                max_connections=settings.mineru.http_client.connection_pool_size,
                max_keepalive_connections=settings.mineru.http_client.keepalive_connections,
            )

            # Check vLLM server connectivity on startup
            if not asyncio.run(self.vllm_client.health_check()):
                logger.error("vLLM server health check failed on startup")
                raise RuntimeError("vLLM server unavailable")

            logger.info("MinerU service initialized with HTTP client")
    ```

  - **Validation**: Service starts without errors

- [x] **3.2.2** Update MinerU CLI invocation
  - **File**: `src/Medical_KG_rev/services/mineru/cli_wrapper.py`
  - **Action**: Update subprocess command:

    ```python
    def build_mineru_command(
        input_dir: Path,
        output_dir: Path,
        vllm_server_url: str,
    ) -> list[str]:
        """Build MinerU CLI command for split-container mode."""
        return [
            "mineru",
            "--pdf-path", str(input_dir),
            "--output-dir", str(output_dir),
            "--backend", "vlm-http-client",  # CHANGED: was "vlm-vllm-engine"
            "--vllm-url", vllm_server_url,  # NEW: vLLM server URL
            # REMOVE GPU flags:
            # "--gpu-memory-utilization", "0.90",
            # "--device", "cuda:0",
        ]
    ```

  - **Validation**: MinerU CLI runs successfully with new flags

- [x] **3.2.3** Remove GPU availability check
  - **File**: `src/Medical_KG_rev/services/mineru/service.py`
  - **Action**: Remove all `torch.cuda.is_available()` checks and GPU initialization code
  - **Validation**: Service starts on CPU-only machine

### 3.3 Worker Deployment Update

- [x] **3.3.1** Update worker Kubernetes deployment
  - **File**: `ops/k8s/base/deployment-mineru-workers.yaml`
  - **Specification**:

    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: mineru-workers
      namespace: medical-kg
    spec:
      replicas: 8  # CHANGED: was 4
      selector:
        matchLabels:
          app: mineru-worker
      template:
        metadata:
          labels:
            app: mineru-worker
        spec:
          # REMOVE GPU node selector and tolerations
          containers:
          - name: mineru
            image: ghcr.io/your-org/mineru-worker:split-container
            env:
            - name: VLLM_SERVER_URL
              value: "http://vllm-server:8000"
            - name: MINERU_BACKEND
              value: "vlm-http-client"
            - name: KAFKA_BOOTSTRAP_SERVERS
              value: "kafka:9092"
            resources:
              requests:
                cpu: 2000m  # CHANGED: was 1000m
                memory: 4Gi  # CHANGED: was 2Gi
                # REMOVE: nvidia.com/gpu: 0.25
              limits:
                cpu: 4000m
                memory: 8Gi
            livenessProbe:
              httpGet:
                path: /health
                port: 8080
              initialDelaySeconds: 30
              periodSeconds: 30
            readinessProbe:
              httpGet:
                path: /ready
                port: 8080
              initialDelaySeconds: 10
              periodSeconds: 10
    ```

  - **Validation**: `kubectl apply -f ops/k8s/base/deployment-mineru-workers.yaml --dry-run=client`

- [x] **3.3.2** Update worker Docker image
  - **File**: `ops/docker/Dockerfile.mineru-worker`
  - **Specification**: Remove vLLM and GPU dependencies:

    ```dockerfile
    FROM python:3.12-slim

    WORKDIR /app

    # Install only CPU dependencies (NO GPU libraries)
    RUN pip install --no-cache-dir \
        mineru[cpu] \
        httpx \
        tenacity \
        pydantic \
        prometheus-client

    # REMOVE: vllm torch cuda-python

    COPY src/ /app/src/
    COPY config/ /app/config/

    ENV PYTHONPATH=/app

    CMD ["python", "-m", "Medical_KG_rev.services.mineru.worker"]
    ```

  - **Validation**: `docker build -f ops/docker/Dockerfile.mineru-worker -t mineru-worker:test .`

---

## Phase 4: Testing Implementation

### 4.1 Unit Tests

- [x] **4.1.1** Create VLLMClient unit tests
  - **File**: `tests/services/mineru/test_vllm_client.py`
  - **Specification**:

    ```python
    """Unit tests for VLLMClient."""
    import pytest
    from unittest.mock import AsyncMock, patch
    import httpx

    from Medical_KG_rev.services.mineru.vllm_client import (
        VLLMClient,
        VLLMTimeoutError,
        VLLMServerError,
    )


    @pytest.mark.asyncio
    async def test_vllm_client_init():
        """Test VLLMClient initialization."""
        client = VLLMClient(base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"
        assert client.client is not None
        await client.client.aclose()


    @pytest.mark.asyncio
    async def test_chat_completion_success(respx_mock):
        """Test successful chat completion request."""
        respx_mock.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "4"}}
                    ],
                    "usage": {"total_tokens": 20},
                },
            )
        )

        async with VLLMClient(base_url="http://localhost:8000") as client:
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "What is 2+2?"}]
            )
            assert response["choices"][0]["message"]["content"] == "4"


    @pytest.mark.asyncio
    async def test_chat_completion_timeout(respx_mock):
        """Test timeout handling."""
        respx_mock.post("http://localhost:8000/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        async with VLLMClient(base_url="http://localhost:8000") as client:
            with pytest.raises(VLLMTimeoutError):
                await client.chat_completion(
                    messages=[{"role": "user", "content": "test"}]
                )


    @pytest.mark.asyncio
    async def test_chat_completion_server_error(respx_mock):
        """Test server error handling."""
        respx_mock.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        async with VLLMClient(base_url="http://localhost:8000") as client:
            with pytest.raises(VLLMServerError):
                await client.chat_completion(
                    messages=[{"role": "user", "content": "test"}]
                )


    @pytest.mark.asyncio
    async def test_health_check_success(respx_mock):
        """Test health check success."""
        respx_mock.get("http://localhost:8000/health").mock(
            return_value=httpx.Response(200)
        )

        async with VLLMClient(base_url="http://localhost:8000") as client:
            healthy = await client.health_check()
            assert healthy is True
    ```

  - **Validation**: `pytest tests/services/mineru/test_vllm_client.py -v`

- [x] **4.1.2** Create CircuitBreaker unit tests
  - **File**: `tests/services/mineru/test_circuit_breaker.py`
  - **Specification**: Tests for all state transitions (CLOSED→OPEN, OPEN→HALF_OPEN, HALF_OPEN→CLOSED, HALF_OPEN→OPEN)
  - **Validation**: `pytest tests/services/mineru/test_circuit_breaker.py -v`

### 4.2 Integration Tests

- [x] **4.2.1** Create vLLM integration test
  - **File**: `tests/integration/test_vllm_integration.py`
  - **Specification**:

    ```python
    """Integration tests with real vLLM server."""
    import pytest
    import asyncio

    from Medical_KG_rev.services.mineru.vllm_client import VLLMClient


    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_vllm_chat_completion():
        """Test with real vLLM server (requires Docker Compose)."""
        async with VLLMClient(base_url="http://localhost:8000") as client:
            # Verify health first
            healthy = await client.health_check()
            assert healthy, "vLLM server not healthy"

            # Test chat completion
            response = await client.chat_completion(
                messages=[
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                max_tokens=50,
                temperature=0.0,
            )

            assert "choices" in response
            assert len(response["choices"]) > 0
            content = response["choices"][0]["message"]["content"]
            assert "Paris" in content


    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_requests():
        """Test multiple concurrent requests."""
        async with VLLMClient(base_url="http://localhost:8000") as client:
            tasks = [
                client.chat_completion(
                    messages=[{"role": "user", "content": f"What is {i}+{i}?"}],
                    max_tokens=20,
                )
                for i in range(1, 9)
            ]

            responses = await asyncio.gather(*tasks)
            assert len(responses) == 8
            assert all("choices" in r for r in responses)
    ```

  - **Validation**: `pytest tests/integration/test_vllm_integration.py -m integration -v`

### 4.3 End-to-End Tests

- [x] **4.3.1** Create E2E PDF processing test
  - **File**: `tests/integration/test_pdf_e2e.py`
  - **Specification**: Process sample PDF through full pipeline (worker → vLLM → output), validate structure
  - **Validation**: `pytest tests/integration/test_pdf_e2e.py -m e2e -v`

- [x] **4.3.2** Create baseline comparison test
  - **File**: `tests/integration/test_quality_baseline.py`
  - **Specification**: Process 20 PDFs, compare outputs with baseline, assert >95% similarity
  - **Validation**: `pytest tests/integration/test_quality_baseline.py -v`

### 4.4 Performance Tests

- [x] **4.4.1** Create vLLM load test script
  - **File**: `tests/performance/vllm_load_test.py`
  - **Specification**:

    ```python
    """Load test for vLLM server."""
    import asyncio
    import time
    from statistics import mean, quantiles

    from Medical_KG_rev.services.mineru.vllm_client import VLLMClient


    async def single_request(client: VLLMClient, request_id: int) -> float:
        """Make single request and return duration."""
        start = time.time()
        await client.chat_completion(
            messages=[{"role": "user", "content": f"Request {request_id}"}],
            max_tokens=100,
        )
        return time.time() - start


    async def load_test(
        concurrency: int = 10,
        total_requests: int = 100,
    ):
        """Run load test with specified concurrency."""
        async with VLLMClient(base_url="http://localhost:8000") as client:
            durations = []

            for batch_start in range(0, total_requests, concurrency):
                batch_end = min(batch_start + concurrency, total_requests)
                tasks = [
                    single_request(client, i)
                    for i in range(batch_start, batch_end)
                ]
                batch_durations = await asyncio.gather(*tasks)
                durations.extend(batch_durations)

            # Calculate statistics
            p50, p95, p99 = quantiles(durations, n=100)[49::46]

            print(f"\nLoad Test Results:")
            print(f"  Concurrency: {concurrency}")
            print(f"  Total Requests: {total_requests}")
            print(f"  Mean Latency: {mean(durations):.2f}s")
            print(f"  P50 Latency: {p50:.2f}s")
            print(f"  P95 Latency: {p95:.2f}s")
            print(f"  P99 Latency: {p99:.2f}s")

            # Assertions
            assert p95 < 10.0, f"P95 latency {p95:.2f}s exceeds 10s threshold"


    if __name__ == "__main__":
        asyncio.run(load_test(concurrency=10, total_requests=100))
    ```

  - **Validation**: `python tests/performance/vllm_load_test.py`

---

## Phase 5: Deployment Execution

### 5.1 Local Development Deployment

- [x] **5.1.1** Deploy vLLM server locally
  - **Command**: `docker-compose -f docker-compose.vllm.yml up -d`
  - **Validation**: `curl http://localhost:8000/health`

- [x] **5.1.2** Update workers in main docker-compose
  - **Command**: `docker-compose up -d --build mineru-worker`
  - **Validation**: `docker-compose ps | grep mineru-worker`

- [x] **5.1.3** Run smoke test
  - **Command**: `bash scripts/test_vllm_api.sh && pytest tests/integration/test_pdf_e2e.py -m e2e`
  - **Validation**: All tests pass

### 5.2 Kubernetes Deployment

- [ ] **5.2.1** Apply Kubernetes manifests
  - **Commands**:

    ```bash
    kubectl apply -f ops/k8s/base/pvc-huggingface-cache.yaml
    kubectl apply -f ops/k8s/base/configmap-vllm-server.yaml
    kubectl apply -f ops/k8s/base/deployment-vllm-server.yaml
    kubectl apply -f ops/k8s/base/service-vllm-server.yaml
    kubectl apply -f ops/k8s/base/networkpolicy-vllm-server.yaml
    kubectl apply -f ops/k8s/base/deployment-mineru-workers.yaml
    kubectl apply -f ops/k8s/base/servicemonitor-vllm-server.yaml
    ```

  - **Validation**: `kubectl get pods -n medical-kg | grep -E 'vllm-server|mineru-worker'`

- [ ] **5.2.2** Verify vLLM server is running
  - **Command**: `kubectl exec -n medical-kg deployment/mineru-workers -- curl -f http://vllm-server:8000/health`
  - **Validation**: HTTP 200 response

- [ ] **5.2.3** Process test PDF through pipeline
  - **Command**: Submit test PDF via Kafka, monitor worker logs
  - **Validation**: PDF processed successfully, output in MinIO

### 5.3 Monitoring Setup

- [ ] **5.3.1** Import Grafana dashboards
  - **Command**: `curl -X POST http://grafana:3000/api/dashboards/import -d @ops/monitoring/grafana/dashboards/vllm-server.json`
  - **Validation**: Dashboard appears in Grafana UI

- [x] **5.3.2** Apply Prometheus alert rules
  - **Command**: `kubectl apply -f ops/monitoring/alerts-vllm.yml`
  - **Validation**: `promtool check rules ops/monitoring/alerts-vllm.yml`

- [ ] **5.3.3** Verify metrics collection
  - **Command**: Query Prometheus for `vllm_` and `mineru_vllm_` metrics
  - **Validation**: Metrics present with recent timestamps

---

## Phase 6: Validation and Optimization

### 6.1 System Validation

- [ ] **6.1.1** Run full integration test suite
  - **Command**: `pytest tests/integration/ -m integration -v`
  - **Validation**: All tests pass

- [ ] **6.1.2** Run performance benchmark
  - **Command**: `python tests/performance/vllm_load_test.py`
  - **Validation**: P95 latency <10s, no errors

- [x] **6.1.3** Run quality comparison test
  - **Command**: `pytest tests/integration/test_quality_baseline.py -v`
  - **Validation**: ≥95% similarity with baseline

### 6.2 Performance Optimization

- [ ] **6.2.1** Tune vLLM batch size
  - **Action**: Experiment with `--max-num-batched-tokens` values (8192, 16384, 32768)
  - **Validation**: Measure throughput improvement

- [ ] **6.2.2** Optimize worker connection pool
  - **Action**: Adjust `connection_pool_size` in config (5, 10, 20)
  - **Validation**: Measure latency improvement

- [ ] **6.2.3** Tune circuit breaker thresholds
  - **Action**: Adjust `failure_threshold` and `recovery_timeout` based on observed patterns
  - **Validation**: Reduced false positives in circuit opening

### 6.3 Security Hardening

- [ ] **6.3.1** Run container vulnerability scan
  - **Command**: `trivy image --severity HIGH,CRITICAL ghcr.io/your-org/mineru-worker:split-container`
  - **Validation**: No HIGH/CRITICAL vulnerabilities

- [ ] **6.3.2** Test NetworkPolicy enforcement
  - **Command**: Attempt vLLM access from unauthorized pod
  - **Validation**: Connection rejected

- [ ] **6.3.3** Verify RBAC permissions
  - **Command**: `kubectl auth can-i --as=system:serviceaccount:medical-kg:vllm-server list secrets -n medical-kg`
  - **Validation**: Permission denied (should not have secret access)

---

## Phase 7: Documentation

### 7.1 Architecture Documentation

- [x] **7.1.1** Update architecture documentation
  - **File**: `docs/gpu-microservices.md`
  - **Action**: Add "vLLM Split-Container Architecture" section with diagrams
  - **Validation**: Documentation builds without errors

- [x] **7.1.2** Create deployment guide
  - **File**: `docs/devops/vllm-deployment.md`
  - **Content**: Prerequisites, Docker deployment, K8s deployment, configuration, troubleshooting
  - **Validation**: Follow guide, verify all steps work

### 7.2 Operational Documentation

- [x] **7.2.1** Create runbook: vLLM Server Restart
  - **File**: `docs/runbooks/vllm-server-restart.md`
  - **Specification**:

    ```markdown
    # Runbook: vLLM Server Restart

    ## Purpose
    Restart vLLM server with zero downtime

    ## Prerequisites
    - kubectl access to medical-kg namespace
    - No active high-priority PDF processing jobs

    ## Procedure

    1. Check current vLLM server status
       ```bash
       kubectl get pods -n medical-kg -l app=vllm-server
       kubectl logs -n medical-kg deployment/vllm-server --tail=50
       ```

    2. Scale workers to reduce load

       ```bash
       kubectl scale deployment/mineru-workers --replicas=4 -n medical-kg
       ```

    3. Restart vLLM server pod

       ```bash
       kubectl rollout restart deployment/vllm-server -n medical-kg
       kubectl rollout status deployment/vllm-server -n medical-kg
       ```

    4. Verify health

       ```bash
       kubectl exec -n medical-kg deployment/mineru-workers -- \
         curl -f http://vllm-server:8000/health
       ```

    5. Scale workers back up

       ```bash
       kubectl scale deployment/mineru-workers --replicas=8 -n medical-kg
       ```

    6. Monitor for 10 minutes

       ```bash
       watch -n 10 'kubectl get pods -n medical-kg | grep -E "vllm|mineru"'
       ```

    ## Validation

    - vLLM server pod in Running state
    - Health endpoint returns 200
    - Workers successfully processing PDFs
    - No circuit breaker open alerts

    ## Rollback

    If issues occur:

    ```bash
    kubectl rollout undo deployment/vllm-server -n medical-kg
    ```

    ```
  - **Validation**: Execute runbook in staging environment

- [x] **7.2.2** Create troubleshooting guide
  - **File**: `docs/troubleshooting/vllm-connectivity.md`
  - **Content**: Common issues, diagnostics, resolution steps
  - **Validation**: Guide covers all failure scenarios

### 7.3 API Documentation

- [x] **7.3.1** Update API documentation
  - **File**: `docs/api/mineru-service.md`
  - **Action**: Document vLLM HTTP client usage, error codes, retry behavior
  - **Validation**: API docs accurate and complete

---

## Phase 8: Cleanup

### 8.1 Remove Legacy Code

- [x] **8.1.1** Remove monolithic worker code
  - **Action**: Delete old GPU initialization code, vLLM engine imports
  - **Files**: Search codebase for `torch.cuda`, `vllm.LLM`, old backend references
  - **Validation**: `git grep -l "torch.cuda" src/` returns no results

- [x] **8.1.2** Remove legacy configuration
  - **Action**: Delete old `mineru.yaml` monolithic settings
  - **Validation**: No references to `vlm-vllm-engine` backend

- [x] **8.1.3** Remove legacy Docker images
  - **Command**: `docker rmi ghcr.io/your-org/mineru-worker:monolithic`
  - **Validation**: Old image deleted from registry

### 8.2 Final Validation

- [ ] **8.2.1** Run complete test suite
  - **Command**: `pytest tests/ -v --cov=Medical_KG_rev`
  - **Validation**: All tests pass, coverage >80%

- [ ] **8.2.2** Validate OpenSpec change
  - **Command**: `openspec validate update-mineru-vllm-split-container --strict`
  - **Validation**: No validation errors

- [ ] **8.2.3** System health check
  - **Command**: Run health check script across all components
  - **Validation**: All components healthy, metrics flowing

---

## Summary

**Total Tasks**: 180 implementation tasks

**Implementation Phases**:

1. **Phase 1**: vLLM Server Infrastructure (18 tasks)
2. **Phase 2**: Worker HTTP Client (8 tasks)
3. **Phase 3**: Worker Update (9 tasks)
4. **Phase 4**: Testing (10 tasks)
5. **Phase 5**: Deployment (10 tasks)
6. **Phase 6**: Validation & Optimization (9 tasks)
7. **Phase 7**: Documentation (8 tasks)
8. **Phase 8**: Cleanup (6 tasks)

**Key Specifications**:

- vLLM server: Qwen2.5-VL-7B-Instruct, 92% GPU memory, 32K context
- Workers: 8 replicas, 2 CPU cores, 4GB RAM each (no GPU)
- Circuit breaker: 5 failure threshold, 60s recovery timeout
- Connection pool: 10 max connections, 5 keepalive
- Retry: 3 attempts, exponential backoff (4-60s)

**Validation Commands**:

- `docker-compose up -d && curl http://localhost:8000/health`
- `kubectl get pods -n medical-kg | grep -E 'vllm|mineru'`
- `pytest tests/ -v`
- `python tests/performance/vllm_load_test.py`

**Success Criteria**:

- ✅ All tests pass
- ✅ P95 latency <10s
- ✅ Quality ≥95% baseline
- ✅ No HIGH/CRITICAL vulnerabilities
- ✅ Workers start <5s
- ✅ 8 workers running concurrently

**Context**: Software not in production, aggressive migration, no legacy infrastructure retention required.

---

**Document Version**: 3.0 (AI Agent Implementation Specifications)
**Last Updated**: 2025-10-08
**Target**: AI Programming Agents
