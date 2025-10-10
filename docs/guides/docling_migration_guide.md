# Docling Migration Guide

This guide documents the steps required to migrate from the deprecated MinerU pipeline to the Docling vision-language model (VLM)
implementation.

## 1. Prerequisites

- CUDA 12.1+ runtime with at least 24GB VRAM per GPU
- Access to Gemma3 12B checkpoint (configured via `DOCLING_VLM_MODEL_NAME` and `DOCLING_VLM_MODEL_PATH`)
- Updated Docker images with Docling dependencies baked in (`docling[vlm]`, `transformers`, `torch`)

## 2. Configuration Changes

1. Set the feature flag:
   ```bash
   export PDF_PROCESSING_BACKEND=docling_vlm
   ```
2. Configure Docling environment variables:
   ```bash
   export DOCLING_VLM_MODEL_PATH=/models/gemma3-12b
   export DOCLING_VLM_MODEL_NAME=google/gemma-3-12b-it
   export DOCLING_VLM_GPU_MEMORY_FRACTION=0.95
   ```
3. Update `config/docling_vlm.yaml` (or environment specific overrides) with batch size, timeout, and retry settings.

## 3. Deployment Checklist

1. Roll out updated containers and Kubernetes manifests (`ops/k8s/docling-vlm-deployment.yaml`).
2. Warm the model cache by running `python scripts/download_gemma3.py` in each GPU node.
3. Verify health endpoints:
   - REST: `GET /health/docling`
   - gRPC: `grpcurl -plaintext localhost:50051 medicalkg.gateway.v1.MineruService.ProcessPdf`
4. Run regression suites:
   ```bash
   pytest tests/services/parsing/test_docling_vlm_service.py -q
   pytest tests/integration/test_docling_vlm_pipeline.py -q
   ```
5. Update Grafana dashboards with new metrics (`docling_vlm_*`).

## 4. Rollback Plan

- Execute `scripts/rollback_to_mineru.sh` to restore the archived MinerU configuration and feature flag.
- Scale down Docling workloads and re-enable MinerU deployments if necessary.
- Monitor `docling_gate_triggered_total` to confirm no new Docling traffic is processed during rollback.

## 5. Post-migration Tasks

- Remove MinerU-specific dashboards and alerts once Docling is stable.
- Archive legacy MinerU documentation under `openspec/changes/archive/`.
- Communicate the change to stakeholders with updated API documentation and troubleshooting links.
