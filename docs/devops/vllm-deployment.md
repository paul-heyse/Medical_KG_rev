# vLLM Split-Container Deployment Guide

This guide walks through provisioning the vLLM server and MinerU workers across development and
production environments.

## Prerequisites

- Kubernetes cluster with NVIDIA GPU nodes (RTX 5090 or comparable)
- NVIDIA device plugin and drivers installed
- Container registry access for MinerU worker image
- Prometheus and Grafana stack deployed (for metrics + dashboards)

## 1. Local Validation

1. Launch the vLLM server locally:
   ```bash
   docker compose -f docker-compose.vllm.yml up -d vllm-server
   curl http://localhost:8000/health
   ```
2. Start a MinerU worker and run the smoke test:
   ```bash
   docker compose up -d mineru-worker
   bash scripts/test_vllm_api.sh
   ```
3. Run targeted unit tests:
   ```bash
   PYTHONPATH=src pytest tests/services/mineru -k vllm -v
   ```

## 2. Build Images

1. Build and push the worker image:
   ```bash
   docker build -f ops/docker/Dockerfile.mineru-worker -t ghcr.io/your-org/mineru-worker:split-container .
   docker push ghcr.io/your-org/mineru-worker:split-container
   ```
2. Pull the official `vllm/vllm-openai` image and tag the approved version.

## 3. Deploy to Kubernetes

Apply the manifests in the recommended order:

```bash
kubectl apply -f ops/k8s/base/pvc-huggingface-cache.yaml
kubectl apply -f ops/k8s/base/configmap-vllm-server.yaml
kubectl apply -f ops/k8s/base/deployment-vllm-server.yaml
kubectl apply -f ops/k8s/base/service-vllm-server.yaml
kubectl apply -f ops/k8s/base/networkpolicy-vllm-server.yaml
kubectl apply -f ops/k8s/base/deployment-mineru-workers.yaml
kubectl apply -f ops/k8s/base/servicemonitor-vllm-server.yaml
```

Verify:

```bash
kubectl get pods -n medical-kg | grep -E 'vllm|mineru'
```

## 4. Post-Deployment Validation

1. Execute the smoke script within a worker pod:
   ```bash
   kubectl exec -n medical-kg deployment/mineru-workers -- bash /app/scripts/test_vllm_api.sh
   ```
2. Confirm metrics ingestion by querying Prometheus for `mineru_vllm_request_duration_seconds`.
3. Import the Grafana dashboard:
   ```bash
   curl -X POST http://grafana:3000/api/dashboards/import \
     -H "Content-Type: application/json" \
     -d @ops/monitoring/grafana/dashboards/vllm-server.json
   ```

## 5. Rollback Strategy

- `kubectl rollout undo deployment/vllm-server -n medical-kg`
- `kubectl rollout undo deployment/mineru-workers -n medical-kg`

Always scale workers to zero before rolling back the vLLM deployment to avoid request failures.

## 6. Operational Checklist

- [ ] vLLM `/health` endpoint returns 200
- [ ] MinerU worker readiness probes are green
- [ ] Prometheus is scraping `vllm_*` and `mineru_vllm_*` series
- [ ] Grafana dashboard updated with live data
- [ ] Alertmanager receiving notifications from `alerts-vllm.yml`

Keep this guide alongside the runbook to accelerate operational tasks and incident response.
