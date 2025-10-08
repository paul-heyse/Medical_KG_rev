# MinerU Vision-Language Deployment

This document describes how the MinerU PDF parsing service integrates with the dedicated
vLLM vision-language model when operating in split-container mode.

## Architecture Overview

```
┌──────────────────────────────────────────┐
│             MinerU Workers               │
│  • 8 CPU containers (2 vCPU / 4GiB)      │
│  • mineru CLI in HTTP client mode        │
│  • Connect to vLLM via Service DNS       │
└──────────────────────────────────────────┘
                    │
                    │  HTTP (OpenAI-compatible)
                    ▼
┌──────────────────────────────────────────┐
│                vLLM Server               │
│  • Model: Qwen/Qwen2.5-VL-7B-Instruct    │
│  • GPU memory utilisation target: 92 %   │
│  • Exposes /v1/chat/completions + /health│
└──────────────────────────────────────────┘
```

### Key Characteristics

- **Split responsibility** – MinerU workers no longer load GPU models. They stream requests to the
  vLLM OpenAI-compatible API and focus exclusively on PDF orchestration.
- **Hot model sharing** – A single vLLM instance batches requests from all workers, dramatically
  improving GPU utilisation and throughput.
- **Resilience** – The worker HTTP client provides connection pooling, retries, and circuit-breaker
  protection. Metrics are exported for end-to-end visibility.

## Local Development

### Start the vLLM Server

```bash
docker compose -f docker-compose.vllm.yml up -d vllm-server
curl http://localhost:8000/health
```

### Start MinerU Workers

```bash
docker compose up -d mineru-worker
```

Workers require the following environment variables:

- `VLLM_SERVER_URL` – defaults to `http://vllm-server:8000`
- `MINERU_BACKEND` – must be `vlm-http-client`

## Kubernetes Deployment

The base manifests under `ops/k8s/base/` deploy the vLLM server and the CPU-only MinerU worker
pool. Apply them via Kustomize or plain `kubectl`:

```bash
kubectl apply -k ops/k8s/base
```

Resources:

- `deployment-vllm-server.yaml` – GPU-backed vLLM instance (RTX 5090)
- `deployment-mineru-workers.yaml` – 8 replicas, no GPU requests
- `networkpolicy-vllm-server.yaml` – restricts access to MinerU workers only
- `servicemonitor-vllm-server.yaml` – scrapes Prometheus metrics from the vLLM pod

## Health Monitoring

- **vLLM** – `GET /health` returns HTTP 200 when the model is ready.
- **Prometheus** – scrape `/metrics` from both vLLM and worker pods for
  `mineru_vllm_*` and `vllm_*` series.
- **Alerts** – `ops/monitoring/alerts-vllm.yml` defines high latency and circuit breaker alerts.

## Graceful Shutdown

Scale the worker deployment to zero before restarting the vLLM server to avoid open requests:

```bash
kubectl scale deployment/mineru-workers --replicas=0 -n medical-kg
kubectl rollout restart deployment/vllm-server -n medical-kg
kubectl scale deployment/mineru-workers --replicas=8 -n medical-kg
```

Monitor pod status and Prometheus metrics throughout the restart to confirm healthy recovery.
