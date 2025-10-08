# Embeddings Service Operations Runbook

This runbook fulfils task 10.3.1 of the
`add-embeddings-representation` change.  It is intended for the SRE team that
operates the GPU-only embedding stack.

## 1. vLLM Service Lifecycle

1. Build the image:
   ```bash
   docker compose build vllm-qwen3
   ```
2. Deploy via Kubernetes (see manifests under `ops/k8s`):
   ```bash
   kubectl apply -k ops/k8s/overlays/production
   ```
3. Health check endpoint: `GET /health` should return `{"status":"healthy"}`.
4. Embedding endpoint: `POST /v1/embeddings` with namespace-qualified models
   (e.g. `single_vector.qwen3.4096.v1`).

### Configuration Sources

- Dense embedding service parameters live in `config/embedding/vllm.yaml` and are
  parsed via `Medical_KG_rev.config.load_vllm_config`. Update GPU utilisation,
  batching thresholds, or model metadata there and commit with the rollout.
- Sparse SPLADE settings (Pyserini) reside in `config/embedding/pyserini.yaml`.
  The helper `load_pyserini_config` validates the schema and exposes
  OpenSearch-specific knobs (`rank_features_field`, `max_weight`).

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

## 6. Namespace Access Controls

- Discovery API `GET /v1/namespaces` requires `embed:read`. Returns
  `NamespaceInfo` entries with `allowed_tenants` and `allowed_scopes`.
- Validation API `POST /v1/namespaces/{namespace}/validate` checks token
  budgets before embedding. Calls `transformers.AutoTokenizer` based on the
  namespace config and records `medicalkg_cross_tenant_access_attempts_total`
  when blocked.
- Embedding requests (`POST /v1/embed`) must include `tenant_id` and a
  namespace registered for that tenant. The FastAPI middleware
  `TenantValidationMiddleware` fails the request if the JWT tenant differs
  from the payload.
- For investigations, audit the `namespace_access` entries in the gateway
  logs and the Prometheus counter above.

## 7. Emergency Procedures

1. **Rollback** – Scale down vLLM deployment and scale up the legacy
   `sentence-transformers` worker (available in `ops/k8s/overlays/rollback`).
   Automated rollback conditions are codified in
   `config/monitoring/rollback_triggers.yaml` and are mirrored as Grafana
   alerts. Keep that file in sync with dashboard IDs referenced below.
2. **Restart** – `kubectl rollout restart deployment vllm-qwen3`.
3. **Purge cache** – Delete `.vllm_cache` PVC to flush stale KV cache.

**Rollback Triggers**

- **Automated alerts** – See `config/monitoring/rollback_triggers.yaml` for
  the canonical set of alerting rules (latency degradation, GPU failure rate,
  token overflow, vLLM availability). Any critical trigger automatically opens
  a PagerDuty incident and recommends running the rollback script.
- **Manual triggers** – Initiate rollback when any of the following conditions
  are observed even if alerts have not fired:
  - Embedding quality degradation (Recall@10 drop ≥5% from previous baseline).
  - GPU memory leaks or repeated OOMs despite healthy alert status.
  - vLLM startup failures (health checks stuck in `503` for >5 minutes after
    deployment).
  - Incorrect vector dimensions or sparse term weights detected during smoke
    tests (e.g., FAISS `DimensionMismatchError`).

After invoking a manual trigger, attach the
`docs/templates/rollback_incident_template.md` to the incident ticket and
schedule the post-incident review within two hours of rollback completion.

**RTO Targets**

- Canary rollback: 5 minutes (scale down new workloads, scale up legacy).
- Full rollback (with OpenSearch mapping restoration): 15 minutes.
- Maximum tolerated RTO: 20 minutes (documented in incident postmortem).

Perform quarterly rollback drills in staging following
`scripts/rollback_embeddings.sh` and record metrics in the change log. Drill
outcomes and RTO validation history are tracked in
`docs/operations/rollback_drills.md`.

## 8. Pre-Deployment Checklist (Task 11.1.3)

- [x] Unit and integration tests green (`pytest tests/embeddings -q`).
- [x] `scripts/detect_dangling_imports.py` returns success.
- [x] GPU fail-fast path validated via `scripts.embedding.verify_environment`.
- [x] Dashboards and alerts reviewed with operations team.
- [x] Runbook reviewed and linked from on-call handbook.

## 9. Staging & Production Rollout (Tasks 11.2 & 11.3)

1. Deploy to staging overlay and run smoke tests (`tests/smoke/embed_smoke.py`).
2. Run storage migration job to rebuild FAISS and OpenSearch indices.
3. Promote manifests to production overlay, monitor for 24–48 hours.
4. Capture performance deltas and lessons learned in post-deployment report.

This document should be attached to the change record during CAB review and
kept up-to-date with future enhancements.
