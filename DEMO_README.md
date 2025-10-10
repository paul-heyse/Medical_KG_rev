# End-to-End MinerU + vLLM Docker Demo

This demo proves that the complete Docker integration works end-to-end by downloading real PDFs from OpenAlex and processing them with MinerU using vLLM for GPU-accelerated inference.

## What This Demo Does

1. **Validates vLLM Server** - Checks that vLLM is running and healthy
2. **Fetches Papers** - Gets random biomedical papers from OpenAlex
3. **Downloads PDFs** - Downloads available PDFs for processing
4. **Processes with MinerU** - Uses MinerU service with vLLM backend to extract structured content
5. **Generates Reports** - Creates detailed JSON reports with processing results

## Prerequisites

### 1. Start Docker Services

```bash
# Start vLLM server (may take 2-5 minutes to load model)
docker compose up -d vllm-server

# Wait for vLLM to be healthy
docker compose logs -f vllm-server
# Wait for: "Uvicorn running on http://0.0.0.0:8000"

# Verify health
curl http://localhost:8000/health
```

### 2. Verify Setup (Optional)

```bash
# Run validation script
./scripts/validate_docker_setup.sh

# Run integration test
python scripts/test_mineru_vllm_integration.py
```

## Running the Demo

### Basic Usage

```bash
# Process 10 papers (default)
python download_and_process_random_papers.py
```

### Advanced Usage

```bash
# Process 20 papers
python download_and_process_random_papers.py --samples 20

# Specify output directory
python download_and_process_random_papers.py --samples 5 --output my_test

# Get help
python download_and_process_random_papers.py --help
```

## Expected Output

The script will show detailed progress:

```
======================================================================
END-TO-END PDF PROCESSING WITH MINERU + VLLM DOCKER
======================================================================
Sample size: 10
Output directory: random_papers_output
Email: paul@heyse.io
vLLM URL: http://localhost:8000
----------------------------------------------------------------------

🏥 STEP 0: Validating vLLM Server Health
----------------------------------------------------------------------
🔍 Checking vLLM server health at http://localhost:8000...
✅ vLLM server is healthy and ready

📚 STEP 1: Fetching Papers from OpenAlex
----------------------------------------------------------------------
🔍 Fetching 10 random papers from OpenAlex...
✅ Fetched 10 papers

📊 STEP 2: Identifying Papers with PDFs
----------------------------------------------------------------------
Papers with PDFs: 7/10

📥 STEP 3: Downloading PDFs
----------------------------------------------------------------------
[1/7] 📥 Downloading PDF: W2162544110.pdf
✅ Downloaded: W2162544110.pdf (1234567 bytes)
...

⚙️  STEP 4: Processing PDFs with MinerU + vLLM
----------------------------------------------------------------------
This demonstrates the full Docker integration:
  • PDFs are processed by MinerU service
  • MinerU connects to vLLM server via HTTP
  • vLLM provides GPU-accelerated inference
----------------------------------------------------------------------

[1/7]
🔄 Processing PDF with MinerU + vLLM: W2162544110.pdf
   📄 Title: A Study of Medical Research...
   📊 PDF size: 1.18 MB
   ⚙️  Sending to MinerU processor...
   ✅ Processing complete in 4.52s
   📊 Extracted: 127 blocks, 3 tables, 5 figures
   💾 Results saved to: W2162544110_processed.json
...

📋 STEP 5: Generating Summary Report
----------------------------------------------------------------------

======================================================================
END-TO-END PROCESSING SUMMARY
======================================================================
✅ vLLM Server:        http://localhost:8000
✅ vLLM Healthy:       True
----------------------------------------------------------------------
Papers fetched:        10
Papers with PDFs:      7
PDFs downloaded:       7
PDFs processed:        7
Processing errors:     0
----------------------------------------------------------------------
Total blocks extracted: 845
Total tables extracted: 21
Total processing time:  31.65s
Avg processing time:    4.52s per PDF
PDF availability rate:  70.0%
Processing success rate: 100.0%
======================================================================
📁 Results saved to: random_papers_output
   • PDFs: random_papers_output/pdfs
   • Processed: random_papers_output/processed
   • Reports: random_papers_output/reports/processing_summary.json
======================================================================

🎉 SUCCESS! End-to-end MinerU + vLLM Docker integration verified!
   ✅ Downloaded PDFs from OpenAlex
   ✅ Processed PDFs with MinerU
   ✅ MinerU connected to vLLM server
   ✅ vLLM provided GPU-accelerated inference
   ✅ Extracted structured content (blocks, tables, etc.)
```

## Output Files

### Directory Structure

```
random_papers_output/
├── pdfs/                     # Downloaded PDF files
│   ├── W2162544110.pdf
│   └── ...
├── processed/                # Processed JSON results
│   ├── W2162544110_processed.json
│   └── ...
└── reports/                  # Summary reports
    └── processing_summary.json
```

### Processed JSON Format

Each processed PDF generates a JSON file with:

```json
{
  "paper_id": "https://openalex.org/W2162544110",
  "title": "A Study of Medical Research...",
  "doi": "10.1234/example",
  "pdf_size_mb": 1.18,
  "processing_time_seconds": 4.52,
  "document_blocks": 127,
  "tables_extracted": 3,
  "figures_extracted": 5,
  "processing_metadata": {
    "worker_id": "worker-1",
    "vllm_backend": true,
    "backend_url": "http://localhost:8000"
  },
  "success": true,
  "timestamp": "2025-10-09 14:30:45"
}
```

### Summary Report Format

```json
{
  "timestamp": "2025-10-09 14:35:12",
  "test_type": "end_to_end_mineru_vllm_docker",
  "docker_integration": {
    "vllm_server_url": "http://localhost:8000",
    "vllm_healthy": true,
    "backend_type": "vllm-http-client"
  },
  "results": {
    "papers_fetched": 10,
    "papers_with_pdfs": 7,
    "pdfs_downloaded": 7,
    "pdfs_processed": 7,
    "processing_errors": 0,
    "total_blocks_extracted": 845,
    "total_tables_extracted": 21
  },
  "success_rates": {
    "pdf_availability_rate": 0.7,
    "processing_success_rate": 1.0
  }
}
```

## Troubleshooting

### vLLM Server Not Running

```
❌ Cannot connect to vLLM server: Connection refused
ℹ️  Make sure Docker services are running:
   docker compose up -d vllm-server
```

**Solution**: Start vLLM server and wait for it to be healthy

### vLLM Server Not Healthy

```
❌ vLLM server is not healthy
```

**Solution**: Check logs and wait for model to load:

```bash
docker compose logs vllm-server
# Wait for startup to complete
```

### No PDFs Found

```
⚠️  No papers with PDFs found
```

**Solution**: This is normal - not all papers have PDFs. The script will continue with whatever PDFs it finds.

### Processing Errors

```
❌ Error processing PDF: MinerU processing failed
```

**Solution**: Check MinerU logs:

```bash
# Check if using simulated CLI (fallback mode)
grep "simulated-cli" logs

# Check vLLM connectivity
python scripts/test_mineru_vllm_integration.py
```

## Monitoring

### Watch GPU Usage

```bash
# Monitor GPU during processing
watch -n 1 nvidia-smi
```

### Monitor Container Logs

```bash
# vLLM server logs
docker compose logs -f vllm-server

# All services
docker compose logs -f
```

### Check Container Health

```bash
docker compose ps
```

## What This Proves

This demo verifies that:

1. ✅ **vLLM Server** runs successfully in Docker with GPU
2. ✅ **MinerU Service** can connect to vLLM via HTTP
3. ✅ **PDF Processing** works end-to-end with real documents
4. ✅ **GPU Inference** is provided by vLLM for layout analysis
5. ✅ **Structured Extraction** produces blocks, tables, and figures
6. ✅ **Docker Integration** connects all components correctly

## Next Steps

After verifying the demo works:

1. **Scale Testing**: Process more papers with `--samples 50`
2. **Performance Tuning**: Adjust worker settings in `config/mineru.yaml`
3. **Production Deployment**: Deploy to Kubernetes using `ops/k8s/` manifests
4. **Monitoring**: Set up Prometheus + Grafana dashboards
5. **Integration**: Connect to downstream pipeline components

## Related Documentation

- **Setup Guide**: `docs/devops/mineru-vllm-docker-setup.md`
- **Quick Start**: `QUICKSTART_MINERU_VLLM.md`
- **Summary**: `MINERU_VLLM_SETUP_SUMMARY.md`
- **Validation**: `scripts/validate_docker_setup.sh`
- **Testing**: `scripts/test_mineru_vllm_integration.py`

---

**Questions?** Check the troubleshooting section or review the Docker logs!
