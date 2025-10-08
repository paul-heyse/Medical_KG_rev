# Change Proposal: Add MinerU GPU CLI Integration

## Why

The current in-house PDF processing implementation (`src/Medical_KG_rev/services/mineru/service.py`) is a lightweight stub that only decodes UTF-8 text and infers blocks heuristically. This approach lacks the sophisticated capabilities of the official MinerU library, which provides:

1. **GPU-accelerated layout analysis** with deep learning models for complex document structures
2. **OCR integration** for scanned PDFs and images
3. **Table detection and extraction** preserving cell relationships and formatting
4. **Figure/image extraction** with captions and metadata
5. **Multi-column layout handling** common in scientific papers
6. **Equation recognition** critical for medical/pharmaceutical literature
7. **Reading order detection** for proper text flow reconstruction

Without MinerU's advanced capabilities, we lose critical information from PDFs, particularly:

- Complex tables (dosing schedules, trial results, pharmacokinetic parameters)
- Figures (molecular structures, clinical trial flowcharts, forest plots)
- Mathematical formulas (PK/PD equations, statistical models)
- Multi-column scientific papers with proper reading order

**Problem Statement**: The stub implementation cannot reliably extract structured information from real-world biomedical PDFs (clinical trial documents, drug labels, research papers), severely limiting the quality of downstream chunking, embedding, and retrieval operations.

## What Changes

### Core Changes

1. **Replace stub MinerU service with official `mineru[gpu]` library integration**
   - Add `mineru[gpu]` to `pyproject.toml` and `requirements.txt`
   - Implement CLI-based invocation with subprocess management
   - Support batch processing with configurable parallelism

2. **Implement parallel worker architecture**
   - Multi-GPU support with worker-per-GPU assignment
   - Queue-based job distribution (Kafka topic: `pdf.parse.requests.v1`)
   - Configurable worker pool size and GPU allocation

3. **Structured output processing**
   - Parse MinerU JSON/Markdown output into IR Block objects
   - Extract tables as structured Table objects with cell relationships
   - Extract figures with captions and save to object storage (MinIO/S3)
   - Preserve provenance (MinerU version, model names, processing timestamp)

4. **Downstream pipeline integration**
   - Post-processing stage to convert MinerU output → chunking-ready blocks
   - Table-aware chunking strategies leveraging extracted structure
   - Figure metadata for cross-referencing and citation
   - Enhanced semantic chunking with layout signals

5. **Performance optimization**
   - Batch processing multiple PDFs per worker invocation
   - GPU memory management and OOM handling
   - Retry logic for transient GPU failures
   - Monitoring and alerting for GPU saturation

### Breaking Changes

- **BREAKING**: `MineruProcessor._decode_pdf` and `MineruProcessor._infer_blocks` methods removed
- **BREAKING**: gRPC `ProcessPDFRequest` payload structure changed to support batch processing
- **BREAKING**: `ProcessPDFResponse` now includes structured tables, figures, and equations

### Service Architecture

**Deployment Model**: MinerU runs as a **separate gRPC microservice in Docker** with GPU access, not embedded in the Python application.

**Rationale**:

- **GPU Isolation**: CUDA drivers, model loading, and GPU memory management isolated from application code
- **Fail-Fast Enforcement**: Service-level GPU checks prevent silent CPU fallbacks
- **Resource Control**: Independent scaling and GPU allocation per service
- **Process Isolation**: MinerU CLI subprocess management contained within service boundaries

**Docker Deployment**:

```dockerfile
# docker/mineru/Dockerfile
FROM nvidia/cuda:12.8-runtime-ubuntu22.04

# Install MinerU with GPU support
RUN pip install mineru[gpu]>=2.5.4

# Pre-download models
RUN python -c "from mineru import download_models; download_models()"

# gRPC service entrypoint
CMD ["python", "-m", "Medical_KG_rev.services.mineru.grpc_server"]
```

**Client Integration**: Orchestration pipeline makes gRPC calls to the MinerU service:

```python
# Python client code (orchestration side)
import grpc
from proto import mineru_service_pb2_grpc

channel = grpc.insecure_channel('mineru-service:8003')
stub = mineru_service_pb2_grpc.MinerUServiceStub(channel)

response = stub.ProcessPDF(
    ProcessPDFRequest(pdf_url="s3://bucket/document.pdf")
)
```

**Health Checks**: Service exposes gRPC health check endpoint that verifies GPU availability before accepting work.

### Configuration Changes

New configuration section in `config/settings.yaml`:

```yaml
mineru:
  enabled: true
  version: ">=2.5.4"  # Minimum required version
  cli_command: "mineru"  # Use built-in MinerU CLI (do not build custom CLI)
  model_dir: "/models/mineru"
  workers:
    count: 4  # Default: 4 parallel workers
    vram_per_worker_gb: 7  # 7GB VRAM per worker (RTX 5090: 32GB total)
    batch_size: 8  # PDFs per worker invocation
  cuda:
    version: "12.8"  # Standard CUDA version
    enforce_version_check: true
  cpu:
    enable_multiprocessing: true  # Enable multi-core CPU utilization
    prevent_bottleneck: true  # Optimize CPU thread/core allocation
  output:
    format: "auto"  # "auto", "markdown", "json"
    extract_images: true
    extract_tables: true
    extract_equations: true
  performance:
    timeout_seconds: 300
    retry_attempts: 3
```

## Impact

### Affected Specs

- **mineru-service**: GPU microservice specification
- **pdf-pipeline**: PDF ingestion and processing pipeline

### Affected Code

**Core Implementation**:

- `src/Medical_KG_rev/services/mineru/service.py` - Complete rewrite
- `src/Medical_KG_rev/services/mineru/worker.py` - New parallel worker implementation
- `src/Medical_KG_rev/services/mineru/parser.py` - New MinerU output parser
- `proto/mineru_service.proto` - Updated gRPC definitions

**Downstream Integration**:

- `src/Medical_KG_rev/orchestration/ingestion_pipeline.py` - Update PDF processing stage
- `src/Medical_KG_rev/chunking/adapters/table_aware.py` - New table-aware chunker
- `src/Medical_KG_rev/models/ir.py` - Extended Block and Table models
- `src/Medical_KG_rev/storage/object_store.py` - Figure/image storage

**Configuration & Deployment**:

- `pyproject.toml` - Add `mineru[gpu]` dependency
- `requirements.txt` - Update GPU dependencies
- `docker/mineru/Dockerfile` - Update base image with MinerU
- `ops/k8s/mineru-deployment.yaml` - Update GPU resource requests

**Testing**:

- `tests/services/mineru/test_cli_integration.py` - New CLI integration tests
- `tests/services/mineru/test_parallel_workers.py` - Worker pool tests
- `tests/services/mineru/test_output_parsing.py` - Output parsing tests
- `tests/integration/test_pdf_pipeline_e2e.py` - End-to-end pipeline tests

### Migration Path

1. **Phase 1 - Dependency Addition** (Week 1):
   - Add `mineru[gpu]` to dependencies
   - Update Docker images with MinerU installation
   - Verify GPU compatibility and model downloads

2. **Phase 2 - Core Implementation** (Week 2-3):
   - Implement CLI wrapper and subprocess management
   - Build parallel worker pool with GPU assignment
   - Create MinerU output parser for all formats

3. **Phase 3 - Pipeline Integration** (Week 4):
   - Update ingestion pipeline to use new service
   - Implement post-processing for downstream stages
   - Add table-aware and figure-aware chunking

4. **Phase 4 - Testing & Validation** (Week 5):
   - Comprehensive unit and integration tests
   - Performance benchmarking (throughput, latency, GPU utilization)
   - Quality assessment on sample biomedical PDFs

5. **Phase 5 - Deployment** (Week 6):
   - Staged rollout with A/B testing (old stub vs new MinerU)
   - Monitor error rates, processing times, GPU metrics
   - Full cutover once quality metrics validated

### Rollback Plan

- Feature flag `mineru.enabled: false` reverts to stub implementation
- Docker images tagged with versions for quick rollback
- Kafka DLQ preserves failed PDFs for reprocessing

### Performance Impact

**Expected Improvements**:

- **Quality**: 10-20x improvement in structured data extraction accuracy
- **Throughput**: 50-100 PDFs/hour per GPU (vs ~500/hour for stub, but stub produces low-quality output)
- **Latency**: 30-60 seconds per PDF (acceptable for batch processing)

**Resource Requirements**:

- **GPU**: RTX 5090 (32GB VRAM) - 4 workers @ 7GB VRAM each
- **CUDA**: Version 12.8 (standard)
- **CPU**: Multi-core configuration to prevent bottleneck
- **Storage**: 2-5GB per 1000 PDFs (extracted images, tables)

### Security Considerations

- **Subprocess isolation**: MinerU CLI runs in separate process with resource limits
- **Input validation**: PDF size limits (max 100MB), format verification
- **Output sanitization**: HTML/Markdown sanitization to prevent XSS
- **Tenant isolation**: Separate working directories per tenant
- **Audit logging**: Track all PDF processing with user_id, tenant_id, correlation_id

### Monitoring & Alerting

**New Metrics**:

- `mineru_processing_duration_seconds` (histogram, labels: gpu_id, status)
- `mineru_pdf_pages_processed_total` (counter, labels: gpu_id)
- `mineru_gpu_memory_usage_bytes` (gauge, labels: gpu_id)
- `mineru_worker_queue_depth` (gauge, labels: worker_id)
- `mineru_cli_failures_total` (counter, labels: gpu_id, error_type)
- `mineru_table_extraction_count` (histogram, labels: pdf_type)
- `mineru_figure_extraction_count` (histogram, labels: pdf_type)

**Alerts**:

- GPU utilization > 95% for 10+ minutes
- Worker queue depth > 100 PDFs
- CLI failure rate > 5%
- Average processing time > 120 seconds
- GPU memory OOM events

### Documentation Updates

- **Architecture docs**: Update GPU microservices section with MinerU details
- **Deployment guide**: GPU setup, MinerU model downloads, configuration
- **API docs**: Updated gRPC service definitions and examples
- **Troubleshooting**: Common MinerU CLI errors and resolutions
