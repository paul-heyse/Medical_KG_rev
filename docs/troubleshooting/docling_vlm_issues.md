# Docling VLM Troubleshooting

## Common Symptoms

| Symptom | Diagnosis Steps | Remediation |
|---------|-----------------|-------------|
| `/health/docling` returns `degraded` with `cache_exists=false` | Verify `DOCLING_VLM_MODEL_PATH` permissions and rerun `scripts/download_gemma3.py`. | Ensure the model cache volume is mounted read/write and rerun the download script. |
| gRPC returns `UNAVAILABLE: Docling VLM unavailable` | GPU not detected or allocated. Check `nvidia-smi` and `DOC-ling` pod tolerations. | Restart the pod after confirming GPU resources are free; adjust `gpu.memory` requests. |
| REST endpoint returns `504 docling-timeout` | Model inference exceeded `DOCLING_VLM_TIMEOUT_SECONDS`. Inspect pipeline logs for oversized PDFs. | Increase timeout and batch size gradually; consider splitting PDFs or disabling warmup prompts. |
| Alert `docling_vlm_processing_seconds > 300` firing | VLM stuck loading weights. | Restart deployment and verify model cache integrity. |

## Diagnostic Commands

```bash
# Inspect Docling Kubernetes deployment
kubectl describe deployment/docling-vlm -n medical-kg

# Tail Docling logs
kubectl logs deployment/docling-vlm -n medical-kg --tail=200

# Check GPU utilisation
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv
```

## Remediation Checklist

1. Confirm feature flag: `PDF_PROCESSING_BACKEND=docling_vlm`.
2. Validate Gemma3 cache permissions and checksum.
3. Ensure Prometheus scraped metrics `docling_vlm_processing_seconds` and `docling_vlm_gpu_memory_mb` are updating.
4. Trigger a test run via `POST /v1/pdf/docling/process` using a small PDF and monitor logs.
