# MinerU Two-Phase Gate Runbook

The PDF ingestion pipeline is split into two phases to enforce GPU-backed
MinerU parsing before chunking and embeddings resume. This runbook explains how
operators can monitor, unblock, and validate the workflow.

## Prerequisites

- MinerU GPU service deployed (`Medical_KG_rev.services.gpu.mineru_service`)
- Job ledger entries with the additional PDF gate fields
- Gateway endpoint `/v1/jobs/{job_id}/postpdf-start` available
- Chunking stack deployed with the new Hugging Face sentence segmenter

## Operational Flow

1. **PDF Download** – Orchestration workers fetch the PDF and update the job
   ledger with `pdf_downloaded=True`.
2. **MinerU Parsing** – GPU service converts PDFs into Markdown/JSON artifacts,
   extracts bounding boxes, and emits table HTML when rectangularization is
   uncertain.
3. **Ledger Update** – Successful MinerU execution sets
   `pdf_ir_ready=True` and stores the `mineru_bbox_map` payload.
4. **Manual Resume** – Operators trigger `/v1/jobs/{job_id}/postpdf-start` once
   the GPU output is verified.
5. **Chunking & Embedding** – Downstream stages pick up the job, load the saved
   MinerU artifacts, and proceed using the profile-specific chunker.

## Triggering `postpdf-start`

```bash
curl -X POST "https://<gateway>/v1/jobs/${JOB_ID}/postpdf-start" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"data":{"type":"postpdf-start","attributes":{}}}'
```

The gateway validates that `pdf_ir_ready` is `true`. If the check fails, the API
returns `409 Conflict` with details. Successful requests mark
`postpdf_start_triggered=True`.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| MinerU job fails with GPU error | Confirm GPU availability (`nvidia-smi`), redeploy MinerU, and retry the job. |
| `postpdf-start` returns 409 | Inspect ledger entry via Dagster UI or Redis CLI to confirm PDF IR readiness. |
| Chunker rejects MinerU output | Run `scripts/check_chunking_dependencies.py` to verify dependencies and ensure the correct profile is selected. |
| Missing tables in chunks | Check MinerU HTML artifacts for `is_unparsed_table=true` and rerun chunking after manual review. |

## Monitoring Checklist

- Grafana dashboard `Medical_KG_Chunking_Quality` – monitor chunk counts,
  failure rates, and sentence segmentation fallback events.
- Prometheus metrics `mineru_gate_triggered_total` and
  `postpdf_start_triggered_total` – confirm jobs move through the two-phase gate.
- Dagster sensor logs – ensure the post-PDF pipeline triggers automatically
  after manual approval.

## Related Guides

- [Chunking & Parsing Runtime](../guides/chunking.md)
- [Chunking Profiles](../guides/chunking-profiles.md)
- [OpenSpec Change Documentation](../../openspec/changes/add-parsing-chunking-normalization/README.md)
