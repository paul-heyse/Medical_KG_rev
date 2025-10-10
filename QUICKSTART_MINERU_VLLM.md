# MinerU + vLLM Quick Start Guide

Get up and running with MinerU and vLLM in Docker in under 10 minutes.

## Prerequisites

- Docker with GPU support
- NVIDIA GPU with 24GB+ VRAM
- 50GB free disk space

## Quick Start

### 1. Validate Setup (2 minutes)

```bash
./scripts/validate_docker_setup.sh
```

✅ All checks should pass or show warnings only

### 2. Start Services (3-5 minutes)

```bash
# Start both vLLM and MinerU
docker compose up -d vllm-server mineru-worker

# Watch logs
docker compose logs -f vllm-server mineru-worker
```

Wait for:

- vLLM: `"Uvicorn running on http://0.0.0.0:8000"`
- MinerU: `"MinerU service imported successfully"`

### 3. Test Integration (30 seconds)

```bash
python scripts/test_mineru_vllm_integration.py
```

Expected output:

```
✓ Settings loaded successfully
✓ vLLM server is reachable
✓ vLLM server reports healthy status
✓ VLLMClient initialized successfully
✓ All critical tests passed!
```

## Common Commands

### Service Management

```bash
# Status
docker compose ps

# Logs
docker compose logs -f mineru-worker

# Restart
docker compose restart mineru-worker

# Stop
docker compose down
```

### Health Checks

```bash
# vLLM health
curl http://localhost:8000/health

# Container health
docker compose ps --format json | jq '.[] | {name: .Name, health: .Health}'
```

### GPU Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Container stats
docker stats vllm-server mineru-worker
```

## Troubleshooting

### vLLM won't start

```bash
# Check GPU
nvidia-smi

# Check logs
docker compose logs vllm-server

# Restart
docker compose restart vllm-server
```

### MinerU can't connect

```bash
# Test connectivity
docker compose exec mineru-worker curl http://vllm-server:8000/health

# Restart
docker compose restart mineru-worker
```

### Out of memory

Edit `docker-compose.yml`:

```yaml
--gpu-memory-utilization 0.85  # Reduce from 0.92
```

## Configuration

### Environment Variables

Located in `docker-compose.yml`:

```yaml
environment:
  MK_MINERU__VLLM_SERVER__BASE_URL: http://vllm-server:8000
  MK_MINERU__WORKERS__BACKEND: vlm-http-client
  MK_MINERU__WORKERS__COUNT: 8
  MK_MINERU__HTTP_CLIENT__CONNECTION_POOL_SIZE: 10
```

### Configuration File

Edit `config/mineru.yaml`:

```yaml
mineru:
  workers:
    count: 8
    batch_size: 4

  http_client:
    timeout_seconds: 300
    retry_attempts: 3
```

## Architecture

```
vLLM Server (GPU) ←─ HTTP ─→ MinerU Worker (CPU)
     :8000                         :N/A
       ↓                             ↓
   GPU Inference              PDF Processing
   Qwen2.5-VL-7B             Batch Operations
   OpenAI API                HTTP Client
```

## Full Documentation

- **Setup Guide**: `docs/devops/mineru-vllm-docker-setup.md`
- **Summary**: `MINERU_VLLM_SETUP_SUMMARY.md`
- **Validation**: `scripts/validate_docker_setup.sh`
- **Testing**: `scripts/test_mineru_vllm_integration.py`

## Support

If you encounter issues:

1. ✅ Run validation: `./scripts/validate_docker_setup.sh`
2. ✅ Check logs: `docker compose logs`
3. ✅ Test integration: `python scripts/test_mineru_vllm_integration.py`
4. ✅ Review: `docs/devops/mineru-vllm-docker-setup.md`

## Next Steps

- [ ] Test with real PDF documents
- [ ] Configure worker scaling
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Deploy to Kubernetes (see `ops/k8s/`)

---

**Ready to Go?** Run `./scripts/validate_docker_setup.sh` to begin!
