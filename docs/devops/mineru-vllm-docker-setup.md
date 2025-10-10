# MinerU + vLLM Docker Setup Guide

This guide provides comprehensive instructions for running MinerU with vLLM in Docker, as designed in the codebase architecture.

## Architecture Overview

The system uses a **split-container architecture**:

- **vLLM Server Container**: GPU-accelerated inference server running Qwen/Qwen2.5-VL-7B-Instruct
- **MinerU Worker Container**: CPU-based workers that connect to vLLM via HTTP
- **Network**: Both containers communicate over a Docker bridge network

```
┌─────────────────────────────────────────────────────────┐
│                  Docker Network                          │
│                (medical-kg-network)                      │
│                                                          │
│  ┌──────────────────────┐    ┌────────────────────────┐ │
│  │   vLLM Server        │    │   MinerU Worker        │ │
│  │                      │    │                        │ │
│  │  - GPU Inference     │◄───│  - CPU Processing      │ │
│  │  - Qwen2.5-VL-7B    │    │  - HTTP Client         │ │
│  │  - OpenAI API        │    │  - PDF Processing      │ │
│  │  - Port: 8000        │    │  - Batch Operations    │ │
│  └──────────────────────┘    └────────────────────────┘ │
│           ▲                                              │
│           │ GPU Access                                   │
│           │                                              │
└───────────┼──────────────────────────────────────────────┘
            │
    ┌───────▼────────┐
    │  NVIDIA GPU    │
    │   (CUDA)       │
    └────────────────┘
```

## Prerequisites

### System Requirements

- **OS**: Linux with kernel 5.x+ (for Docker GPU support)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 24GB+ VRAM (for Qwen2.5-VL-7B-Instruct)
- **Disk**: 50GB free space (for models and containers)
- **CUDA**: Version 11.8 or newer

### Software Requirements

1. **Docker** (20.10+)

   ```bash
   docker --version
   ```

2. **Docker Compose** (V2)

   ```bash
   docker compose version
   ```

3. **NVIDIA Container Toolkit**

   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

### Installation Links

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Configuration

### Environment Variables

The MinerU worker is configured via environment variables with the prefix `MK_MINERU__`:

```yaml
environment:
  # vLLM Server Connection
  MK_MINERU__VLLM_SERVER__BASE_URL: http://vllm-server:8000

  # Worker Configuration
  MK_MINERU__WORKERS__BACKEND: vlm-http-client
  MK_MINERU__WORKERS__COUNT: 8

  # HTTP Client Settings
  MK_MINERU__HTTP_CLIENT__CONNECTION_POOL_SIZE: 10
  MK_MINERU__HTTP_CLIENT__KEEPALIVE_CONNECTIONS: 5
  MK_MINERU__HTTP_CLIENT__RETRY_ATTEMPTS: 3
  MK_MINERU__HTTP_CLIENT__TIMEOUT_SECONDS: 300
```

### Configuration Files

- **`config/mineru.yaml`**: MinerU service configuration
- **`docker-compose.yml`**: Service orchestration
- **`ops/docker/Dockerfile.mineru-worker`**: MinerU worker image definition

## Deployment Steps

### 1. Validate Setup

Run the validation script to check prerequisites:

```bash
./scripts/validate_docker_setup.sh
```

This checks:

- Docker installation
- GPU support
- Required files
- Configuration syntax
- Available resources

### 2. Start Services

Start vLLM server and MinerU worker:

```bash
# Start all services
docker compose up -d

# Or start specific services
docker compose up -d vllm-server mineru-worker
```

### 3. Monitor Startup

Watch the logs to ensure services start correctly:

```bash
# Follow all logs
docker compose logs -f

# Follow specific service
docker compose logs -f vllm-server
docker compose logs -f mineru-worker
```

**Expected startup sequence:**

1. **vLLM Server** (2-5 minutes):

   ```
   Loading model Qwen/Qwen2.5-VL-7B-Instruct...
   Model loaded successfully
   Uvicorn running on http://0.0.0.0:8000
   ```

2. **MinerU Worker** (30-60 seconds):

   ```
   MinerU service imported successfully
   VLLMClient initialized
   Worker ready to process requests
   ```

### 4. Verify Integration

Run the integration test:

```bash
python scripts/test_mineru_vllm_integration.py
```

This tests:

- Configuration loading
- vLLM server connectivity
- Health checks
- Client initialization
- Chat completion (optional)

## Service Management

### Check Service Status

```bash
# List running containers
docker compose ps

# Check service health
docker compose ps --format json | jq '.[] | {name: .Name, health: .Health}'
```

### View Logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs vllm-server
docker compose logs mineru-worker

# Follow logs (real-time)
docker compose logs -f mineru-worker

# Last N lines
docker compose logs --tail=100 vllm-server
```

### Restart Services

```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart mineru-worker

# Restart with rebuild
docker compose up -d --build mineru-worker
```

### Stop Services

```bash
# Stop all services
docker compose stop

# Stop specific service
docker compose stop mineru-worker

# Stop and remove containers
docker compose down

# Stop, remove containers and volumes
docker compose down -v
```

## Troubleshooting

### vLLM Server Issues

**Problem**: vLLM server fails to start

```bash
# Check GPU availability
nvidia-smi

# Check vLLM logs
docker compose logs vllm-server

# Common issues:
# - Insufficient GPU memory: Reduce --gpu-memory-utilization
# - Model download failed: Check internet connection and HuggingFace access
# - CUDA version mismatch: Update NVIDIA drivers
```

**Problem**: vLLM server OOM (Out of Memory)

```yaml
# In docker-compose.yml, adjust GPU memory:
command: >
  python -m vllm.entrypoints.openai.api_server \
    --gpu-memory-utilization 0.85  # Reduce from 0.92
    --max-model-len 16384          # Reduce from 32768
```

### MinerU Worker Issues

**Problem**: MinerU cannot connect to vLLM

```bash
# Check network connectivity
docker compose exec mineru-worker curl http://vllm-server:8000/health

# Verify DNS resolution
docker compose exec mineru-worker nslookup vllm-server

# Check network
docker network inspect medical-kg-network
```

**Problem**: Import errors in MinerU worker

```bash
# Rebuild with no cache
docker compose build --no-cache mineru-worker

# Check Python environment
docker compose exec mineru-worker python -c "import Medical_KG_rev.services.mineru"
```

### Network Issues

**Problem**: Services cannot communicate

```bash
# Verify network exists
docker network ls | grep medical-kg-network

# Recreate network
docker network rm medical-kg-network
docker compose up -d
```

### Performance Issues

**Problem**: Slow inference

1. Check GPU utilization:

   ```bash
   nvidia-smi -l 1
   ```

2. Adjust worker settings in `config/mineru.yaml`:

   ```yaml
   workers:
     count: 4  # Reduce if CPU-bound
     batch_size: 2  # Reduce if memory-bound
   ```

3. Tune vLLM parameters:

   ```yaml
   command: >
     --tensor-parallel-size 2  # Use multiple GPUs
     --max-num-batched-tokens 8192  # Adjust batch size
   ```

## Testing

### Manual Health Check

```bash
# vLLM health endpoint
curl http://localhost:8000/health

# vLLM models endpoint
curl http://localhost:8000/v1/models
```

### Integration Test

```bash
# Run comprehensive test
python scripts/test_mineru_vllm_integration.py

# Expected output:
# ✓ Settings loaded successfully
# ✓ vLLM server is reachable
# ✓ vLLM server reports healthy status
# ✓ VLLMClient initialized successfully
# ✓ Chat completion successful
```

### Processing Test

```bash
# Test PDF processing (when integrated)
python -c "
from Medical_KG_rev.services.mineru import MineruProcessor
processor = MineruProcessor()
# Add test PDF processing here
"
```

## Monitoring

### Resource Usage

```bash
# Container stats
docker stats vllm-server mineru-worker

# GPU monitoring
watch -n 1 nvidia-smi
```

### Metrics

- **Prometheus metrics**: `http://localhost:9090`
- **vLLM metrics**: Available at `/metrics` endpoint
- **Application metrics**: Via Prometheus client in MinerU worker

### Logs

```bash
# Structured logs with timestamps
docker compose logs -f --timestamps mineru-worker

# Search logs
docker compose logs mineru-worker | grep ERROR

# Export logs
docker compose logs --no-color > docker-logs.txt
```

## Best Practices

1. **Always validate setup** before deployment
2. **Monitor resource usage** during operation
3. **Keep models cached** in HuggingFace cache directory
4. **Use health checks** to detect issues early
5. **Scale workers** based on CPU availability
6. **Tune GPU memory** based on available VRAM
7. **Enable circuit breakers** to prevent cascading failures

## Advanced Configuration

### Multi-GPU Setup

```yaml
vllm-server:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0', '1']  # Use GPUs 0 and 1
            capabilities: [gpu]
  command: >
    --tensor-parallel-size 2  # Parallel across 2 GPUs
```

### Production Tuning

```yaml
# In config/mineru.yaml
mineru:
  workers:
    count: 16  # More workers for high throughput
    batch_size: 8  # Larger batches

  http_client:
    connection_pool_size: 20  # More connections
    retry_attempts: 5  # More retries
    circuit_breaker:
      enabled: true
      failure_threshold: 10
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [MinerU Documentation](https://github.com/opendatalab/MinerU)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

## Support

For issues and questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review logs: `docker compose logs`
3. Run validation: `./scripts/validate_docker_setup.sh`
4. Run integration test: `python scripts/test_mineru_vllm_integration.py`
