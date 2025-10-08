# Embeddings Service Operations Runbook

This runbook fulfils task 10.3.1 of the
`add-embeddings-representation` change.  It is intended for the SRE team that
operates the GPU-only embedding stack.

## 1. vLLM Service Lifecycle

1. Build the image:
   ```bash
   docker compose build vllm-embedding
   ```
2. Deploy via Kubernetes (see manifests under `ops/k8s`):
   ```bash
   kubectl apply -k ops/k8s/overlays/production
   ```
3. Health check endpoint: `GET /health` should return `{"status":"healthy"}`.
4. Embedding endpoint: `POST /v1/embeddings` with namespace-qualified models
   (e.g. `single_vector.qwen3.4096.v1`).

## 2. GPU Troubleshooting

| Symptom | Checks | Resolution |
|---------|--------|------------|
| Pod fails to start | `kubectl describe pod` → look for `nvidia.com/gpu` scheduling errors | Ensure node has GPU label and drivers installed. |
| 503 GPU unavailable | `scripts.embedding.verify_environment` reports `available: false` | Reboot node, reseat drivers, or cordon and drain affected node. |
| OOM | vLLM logs mention CUDA OOM | Reduce batch size via `GPU_MEMORY_UTILIZATION` env var and redeploy. |

## 3. FAISS Index Management

1. Initial bootstrap via `scripts/vector_store/bootstrap_faiss.py` (ensures HNSW
   index and metadata in object storage).
2. Incremental updates handled by `VectorStoreService.upsert`; monitor the
   `embedding.vector_store.upserted` metric.
3. Rebuild procedure:
   - Pause ingestion pipeline.
   - Delete FAISS PVC (`kubectl delete pvc faiss-data`).
   - Re-run bootstrap script.
   - Resume ingestion.

## 4. OpenSearch Rank Features

- Index templates defined in `config/embedding/namespaces/*.yaml` include the
  `rank_features` mapping.  Apply template updates using the
  `scripts/opensearch/apply_rank_features.py` helper.
- Validate with:
  ```bash
  curl -u "$USER:$PASS" https://opensearch/_mapping | jq '.properties'
  ```

## 5. Monitoring & Alerting

- **Dashboards** – Grafana folder `Embeddings & Representation`.  Panels:
  latency (FAISS P95 < 50ms, OpenSearch P95 < 200ms), throughput, and GPU
  utilisation.
- **Alerts** – Prometheus rules located in `ops/monitoring/embeddings.rules.yaml`:
  - `EmbeddingGpuUnavailable` – triggered when health endpoint returns non-200.
  - `EmbeddingThroughputLow` – triggered when throughput < 500 emb/sec.
  - `EmbeddingLatencyHigh` – triggered when FAISS P95 > 80ms for 5 minutes.

## 6. Emergency Procedures

1. **Rollback** – Scale down vLLM deployment and scale up the legacy
   `sentence-transformers` worker (available in `ops/k8s/overlays/rollback`).
2. **Restart** – `kubectl rollout restart deployment vllm-embedding`.
3. **Purge cache** – Delete `.vllm_cache` PVC to flush stale KV cache.

## 7. Pre-Deployment Checklist (Task 11.1.3)

- [x] Unit and integration tests green (`pytest tests/embeddings -q`).
- [x] `scripts/detect_dangling_imports.py` returns success.
- [x] GPU fail-fast path validated via `scripts.embedding.verify_environment`.
- [x] Dashboards and alerts reviewed with operations team.
- [x] Runbook reviewed and linked from on-call handbook.

## 8. Staging & Production Rollout (Tasks 11.2 & 11.3)

1. Deploy to staging overlay and run smoke tests (`tests/smoke/embed_smoke.py`).
2. Run storage migration job to rebuild FAISS and OpenSearch indices.
3. Promote manifests to production overlay, monitor for 24–48 hours.
4. Capture performance deltas and lessons learned in post-deployment report.

This document should be attached to the change record during CAB review and
kept up-to-date with future enhancements.
