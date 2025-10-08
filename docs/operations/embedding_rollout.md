# Embedding Service Rollout Guide

This guide captures the staged rollout procedure for the vLLM/Pyserini embedding stack and documents the validation signals required to complete the OpenSpec deployment tasks.

## 1. Prerequisites

- Install Kubernetes CLI (`kubectl`) with access to the target cluster.
- Ensure the vLLM image has been published to the container registry referenced by `ops/k8s/base/deployment-vllm-embedding.yaml`.
- Prometheus and Grafana must be configured using the manifests in `ops/monitoring/` so the new metrics appear during validation.

## 2. Deploy to Staging

Run the helper script which validates `kubectl` availability and applies the staging overlay:

```bash
python scripts/embedding/deploy.py staging --dry-run  # sanity check
python scripts/embedding/deploy.py staging
```

Monitor the deployment:

1. `kubectl rollout status deployment/vllm-embedding -n embeddings`
2. `kubectl get pods -n embeddings` to verify GPU scheduling (node selector + tolerations enforced).
3. Inspect Grafana dashboard **Embeddings & Representation** for:
   - `medicalkg_embedding_duration_seconds` P95 under 500ms.
   - Throughput increasing as jobs execute.
   - Cache ratio trending towards 0.8+ after warm-up.

Smoke test using the gateway REST endpoint with the new namespace parameter and confirm responses include the namespace metadata.

## 3. Storage Migration

Staging storage validation prior to production deployment:

1. Trigger the background job to rebuild the FAISS index using the orchestration worker (documented in `docs/guides/embedding_catalog.md`).
2. Confirm FAISS file creation in the persistent volume and execute a retrieval QA query verifying expected Recall@10 values.
3. Run the sparse expansion job to populate the OpenSearch `rank_features` field and validate using the `write_sparse_embeddings` smoke test in `tests/services/embedding/test_embedding_vector_store.py`.

## 4. Deploy to Production

Execute the deployment script without the dry-run flag:

```bash
python scripts/embedding/deploy.py production
```

Watch the Grafana dashboard for the following acceptance criteria over the first 24 hours:

- Throughput ≥ 1000 embeddings/sec (`medicalkg_embeddings_generated_total`).
- GPU utilisation between 60%–80% on average.
- Cache hit ratio ≥ 70% (`medicalkg_embedding_cache_hits_total`).
- No sustained growth in `medicalkg_embedding_failures_total` (bursts should resolve after retries).

In addition, stream CloudEvents from the topic `embedding.events.v1` using the Kafka tooling bundled with the orchestration change set to confirm `embedding.started`, `embedding.completed`, and `embedding.failed` appear for each batch. Events conform to the following schema:

```json
{
  "specversion": "1.0",
  "type": "com.medical-kg.embedding.completed",
  "source": "services.embedding.worker",
  "subject": "single_vector.qwen3.4096.v1",
  "id": "<uuid>",
  "time": "2025-10-07T14:30:00Z",
  "datacontenttype": "application/json",
  "correlationid": "<correlation>",
  "data": {
    "tenant_id": "tenant-a",
    "namespace": "single_vector.qwen3.4096.v1",
    "provider": "vllm",
    "duration_ms": 245.7,
    "embeddings_generated": 128,
    "cache_hits": 96,
    "cache_misses": 32
  }
}
```

## 5. Post-Deployment Report

Document the rollout in the operations journal with:

- Observed throughput and latency deltas versus the legacy stack.
- Cache warm-up time and steady-state ratio.
- Any anomalies captured via CloudEvents or Prometheus alerts.
- Confirmation that CPU fallbacks remained at zero (vLLM job failures emit `error_type="GpuNotAvailableError"`).

Share the report with the platform operations team and attach Grafana snapshots for traceability.
