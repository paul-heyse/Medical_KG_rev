# PDF Two-Phase Pipeline with MinerU Gate

The **pdf-two-phase** pipeline orchestrates ingestion of scholarly PDFs using a
two-step topology: metadata acquisition and download in the first phase, and
MinerU IR generation plus downstream enrichment in the second. Execution pauses
at the `gate_pdf_ir_ready` stage until MinerU signals that structured IR is
ready for chunking.

## Pipeline Overview

```yaml
name: pdf-two-phase
version: "2025-03-15"
stages:
  - ingest (OpenAlex metadata adapter)
  - download (PDF acquisition)
  - gate_pdf_ir_ready (ledger-based gate)
  - chunk → embed → index → extract → kg
gates:
  - pdf_ir_ready resumes chunking once MinerU sets `pdf_ir_ready=true`
```

### Phase 1 – Metadata and Download

1. **Ingest** – Uses the OpenAlex adapter to fetch metadata including
   candidate PDF URLs (`best_oa_location`, `primary_location`, etc.).
2. **Download** – Extracts PDF URLs from adapter payloads and downloads the
   first reachable asset. Files are stored under
   `/var/lib/medical-kg/pdfs/{doc_id}.pdf` by default, and the Job Ledger is
   updated with the resolved URL, storage path, checksum, and timestamp.

If no URL can be downloaded, the ledger records the failure, enabling retry
automation or manual intervention.

### Phase 2 – MinerU and Downstream Processing

The pipeline halts at **gate_pdf_ir_ready** until MinerU updates the ledger with
`pdf_ir_ready=true`. Once the gate passes, chunking, embedding, indexing,
extraction, and knowledge-graph stages resume using the MinerU IR output.

## Download Stage Configuration

The download stage is declarative and supports multiple URL extractors:

```yaml
stages:
  - name: download
    type: download
    policy: polite-api
    depends_on: [ingest]
    config:
      storage:
        base_path: /var/lib/medical-kg/pdfs
        filename_template: "{doc_id}.pdf"
      http:
        timeout_seconds: 45
        max_attempts: 3
        user_agent: Medical-KG-Pipeline/1.0
      url_extractors:
        - source: payload
          path: best_oa_location.pdf_url
        - source: payload
          path: primary_location.pdf_url
        - source: payload
          path: locations[].pdf_url
```

Each extractor uses dot-path notation; `[]` expands list values. The stage
deduplicates candidates, downloads using the shared HTTP client (with retry and
backoff), records Prometheus metrics, and writes the file to the configured
storage location.

## Gate Configuration

```yaml
stages:
  - name: gate_pdf_ir_ready
    type: gate
    depends_on: [download]
    policy: default
    config:
      gate: pdf_ir_ready
      field: pdf_ir_ready
      equals: true
      resume_stage: chunk
      timeout_seconds: 1800
      poll_interval_seconds: 15

gates:
  - name: pdf_ir_ready
    resume_stage: chunk
    condition:
      field: pdf_ir_ready
      equals: true
      timeout_seconds: 1800
      poll_interval_seconds: 15
```

The gate stage polls the Job Ledger for the configured field, emitting metrics
for pass, wait, and timeout outcomes. Ledger metadata captures elapsed time and
attempt counts (`gate.pdf_ir_ready.elapsed_seconds`).

## Metrics and Observability

New Prometheus series expose pipeline health:

- `pdf_download_duration_seconds{pipeline,status}` – download latency
- `pdf_download_size_bytes{pipeline}` – PDF sizes
- `pdf_download_events_total{pipeline,status}` – success/failure counts
- `pipeline_gate_wait_seconds{pipeline,gate}` – gate wait durations
- `pipeline_gate_events_total{pipeline,gate,outcome}` – gate outcome counts

These metrics support dashboards that highlight stuck downloads or long MinerU
turnaround times.

## Troubleshooting

- **Download failures** – Inspect ledger metadata (`pdf_download_error`) and
  retry with updated extractor configuration or alternate sources.
- **Gate timeouts** – Ensure MinerU updates the ledger. The gate stage logs
  warnings with the last observed value to aid diagnosis.
- **Storage issues** – The download stage creates the target directory; ensure
  the orchestration worker has write permissions to `/var/lib/medical-kg/pdfs`.
