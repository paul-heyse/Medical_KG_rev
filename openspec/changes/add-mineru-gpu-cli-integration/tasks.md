# Implementation Tasks: Add MinerU GPU CLI Integration

## 1. Dependency Management & Setup

- [x] 1.1 Add `mineru[gpu]>=2.5.4` to `pyproject.toml` dependencies
- [x] 1.2 Add MinerU-specific dependencies to `requirements.txt` (automatically handled by mineru[gpu])
- [x] 1.3 Create `scripts/setup_mineru.sh` for model downloads and initialization
- [x] 1.4 Update `.env.example` with MinerU configuration variables
- [x] 1.5 Document GPU requirements (CUDA 12.8, RTX 5090 32GB VRAM - 4 workers @ 7GB each)
- [x] 1.6 Configure Python environment for multi-core CPU utilization (multiprocessing settings)
- [x] 1.7 Verify CUDA 12.8 installation and compatibility

## 2. Core MinerU CLI Integration

- [x] 2.1 Create `src/Medical_KG_rev/services/mineru/cli_wrapper.py`
  - [x] 2.1.1 Implement wrapper for built-in `mineru` CLI command (leverage existing CLI, don't build custom)
  - [x] 2.1.2 Implement subprocess management with timeout handling and cleanup
  - [x] 2.1.3 Implement stdout/stderr capture and logging
  - [x] 2.1.4 Add GPU device assignment per worker (`CUDA_VISIBLE_DEVICES`)
  - [x] 2.1.5 Configure per-worker VRAM limit (7GB max via MinerU CLI options)
  - [x] 2.1.6 Support batch processing (multiple PDFs per invocation)

- [x] 2.2 Create `src/Medical_KG_rev/services/mineru/output_parser.py`
  - [x] 2.2.1 Implement JSON output parser for MinerU structured format
  - [x] 2.2.2 Implement Markdown output parser with metadata extraction
  - [x] 2.2.3 Parse table structures with cell coordinates and content
  - [x] 2.2.4 Parse figure metadata with image paths and captions
  - [x] 2.2.5 Parse equation blocks with LaTeX/MathML representation
  - [x] 2.2.6 Handle parsing errors and incomplete outputs gracefully

- [x] 2.3 Update `src/Medical_KG_rev/services/mineru/service.py`
  - [x] 2.3.1 Remove stub `_decode_pdf` and `_infer_blocks` methods
  - [x] 2.3.2 Integrate `MineruCLI` for PDF processing
  - [x] 2.3.3 Integrate `MineruOutputParser` for result parsing
  - [x] 2.3.4 Add fail-fast GPU availability check on startup
  - [x] 2.3.5 Implement batch request handling
  - [x] 2.3.6 Add GPU memory monitoring and OOM detection

## 3. Parallel Worker Architecture

- [x] 3.1 Create `src/Medical_KG_rev/services/mineru/worker_pool.py`
  - [x] 3.1.1 Implement `WorkerPool` class with default of 4 parallel workers
  - [x] 3.1.2 Implement `Worker` class with GPU assignment (one GPU per worker)
  - [x] 3.1.3 Configure each worker with 7GB VRAM limit
  - [x] 3.1.4 Add Kafka consumer integration for `pdf.parse.requests.v1` topic
  - [x] 3.1.5 Implement job queue with priority support
  - [x] 3.1.6 Add worker health checks and auto-restart
  - [x] 3.1.7 Implement graceful shutdown on SIGTERM
  - [x] 3.1.8 Configure multiprocessing for parallel CPU utilization (prevent CPU bottleneck)

- [x] 3.2 Create `src/Medical_KG_rev/services/mineru/gpu_manager.py`
  - [x] 3.2.1 Implement GPU discovery and enumeration
  - [x] 3.2.2 Track GPU utilization per worker (7GB VRAM allocation per worker)
  - [x] 3.2.3 Implement GPU memory usage monitoring
  - [x] 3.2.4 Add OOM detection and worker recovery
  - [x] 3.2.5 Emit GPU metrics to Prometheus
  - [x] 3.2.6 Validate CUDA 12.8 availability on startup

- [x] 3.3 Update `src/Medical_KG_rev/config/settings.py`
  - [x] 3.3.1 Add `MineruSettings` configuration class
  - [x] 3.3.2 Add `MineruWorkerSettings` with default worker_count=4, vram_per_worker=7GB
  - [x] 3.3.3 Add validation for GPU IDs and batch sizes
  - [x] 3.3.4 Add performance tuning parameters (timeout, memory limits)
  - [x] 3.3.5 Add CPU multiprocessing configuration (core/thread allocation)

## 4. Data Model Updates

- [x] 4.1 Update `src/Medical_KG_rev/models/ir.py`
  - [x] 4.1.1 Extend `Block` model with `layout_bbox` (bounding box coordinates)
  - [x] 4.1.2 Add `reading_order` field for multi-column layout
  - [x] 4.1.3 Add `confidence_score` from MinerU model predictions

- [x] 4.2 Create `src/Medical_KG_rev/models/table.py`
  - [x] 4.2.1 Define `TableCell` model (content, row, col, rowspan, colspan)
  - [x] 4.2.2 Define `Table` model (cells, headers, caption, bbox, page_num)
  - [x] 4.2.3 Add `to_markdown()` and `to_html()` serialization methods
  - [x] 4.2.4 Add validation for table structure integrity

- [x] 4.3 Create `src/Medical_KG_rev/models/figure.py`
  - [x] 4.3.1 Define `Figure` model (image_path, caption, bbox, page_num, figure_type)
  - [x] 4.3.2 Add support for molecular structures, plots, diagrams
  - [x] 4.3.3 Add MIME type and dimensions metadata

- [x] 4.4 Create `src/Medical_KG_rev/models/equation.py`
  - [x] 4.4.1 Define `Equation` model (latex, mathml, bbox, page_num)
  - [x] 4.4.2 Add rendering metadata (display vs inline)

## 5. gRPC Service Updates

- [x] 5.1 Update `proto/mineru_service.proto`
  - [x] 5.1.1 Add `BatchProcessPDFRequest` message for multiple PDFs
  - [x] 5.1.2 Update `ProcessPDFResponse` with tables, figures, equations fields
  - [x] 5.1.3 Add `ProcessingMetadata` with MinerU version, model names, GPU ID
  - [x] 5.1.4 Add `TableStructure`, `FigureMetadata`, `EquationData` messages
  - [x] 5.1.5 Regenerate Python gRPC stubs

- [x] 5.2 Update `src/Medical_KG_rev/services/mineru/grpc_server.py`
  - [x] 5.2.1 Implement `BatchProcessPDF` RPC handler
  - [x] 5.2.2 Convert parsed MinerU output to gRPC response messages
  - [x] 5.2.3 Add error handling with specific status codes
  - [x] 5.2.4 Add OpenTelemetry tracing for RPC calls

## 6. Downstream Pipeline Integration

- [x] 6.1 Update `src/Medical_KG_rev/orchestration/ingestion_pipeline.py`
  - [x] 6.1.1 Update PDF processing stage to call new MinerU service
  - [x] 6.1.2 Add post-processing stage for MinerU output transformation
  - [x] 6.1.3 Handle tables, figures, equations in separate processing paths
  - [x] 6.1.4 Update ledger states (`pdf_parsing` → `pdf_parsed` → `postpdf_processing`)

- [x] 6.2 Create `src/Medical_KG_rev/services/mineru/postprocessor.py`
  - [x] 6.2.1 Convert MinerU blocks to chunking-ready IR blocks
  - [x] 6.2.2 Extract table data and serialize to JSON/CSV
  - [x] 6.2.3 Upload figures to MinIO/S3 with metadata
  - [x] 6.2.4 Inline-render small equations, link large ones
  - [x] 6.2.5 Preserve reading order and layout signals

- [x] 6.3 Create `src/Medical_KG_rev/chunking/adapters/table_aware.py`
  - [x] 6.3.1 Implement `TableAwareChunker` using extracted table structures
  - [x] 6.3.2 Keep table rows/columns together in chunks
  - [x] 6.3.3 Add table captions to chunk metadata
  - [x] 6.3.4 Handle table-heavy documents (dosing schedules, trial results)

- [x] 6.4 Update `src/Medical_KG_rev/storage/object_store.py`
  - [x] 6.4.1 Add `store_figure()` method for image uploads
  - [x] 6.4.2 Add `generate_figure_url()` for signed URLs
  - [x] 6.4.3 Add tenant isolation for figure storage paths
  - [x] 6.4.4 Add figure cleanup on document deletion

## 7. Performance Optimization

- [x] 7.1 Implement batch processing optimizations
  - [x] 7.1.1 Group PDFs by size for balanced batches
  - [x] 7.1.2 Implement adaptive batch sizing based on GPU memory
  - [x] 7.1.3 Add PDF pre-validation (format check, size limits)

- [x] 7.2 Implement caching strategies
  - [x] 7.2.1 Cache MinerU results in Redis (keyed by PDF hash)
  - [x] 7.2.2 Add TTL for cached results (30 days)
  - [x] 7.2.3 Implement cache invalidation on reprocessing requests

- [x] 7.3 Implement retry and error handling
  - [x] 7.3.1 Add exponential backoff for transient GPU errors
  - [x] 7.3.2 Implement fallback to CPU for OOM errors (with warning)
  - [x] 7.3.3 Add dead letter queue for permanently failed PDFs
  - [x] 7.3.4 Emit failure metrics by error type

## 8. Monitoring & Observability

- [x] 8.1 Add Prometheus metrics
  - [x] 8.1.1 `mineru_processing_duration_seconds` histogram
  - [x] 8.1.2 `mineru_pdf_pages_processed_total` counter
  - [x] 8.1.3 `mineru_gpu_memory_usage_bytes` gauge
  - [x] 8.1.4 `mineru_worker_queue_depth` gauge
  - [x] 8.1.5 `mineru_cli_failures_total` counter by error type
  - [x] 8.1.6 `mineru_table_extraction_count` histogram
  - [x] 8.1.7 `mineru_figure_extraction_count` histogram

- [x] 8.2 Add OpenTelemetry spans
  - [x] 8.2.1 Span for CLI invocation with GPU ID attribute
  - [x] 8.2.2 Span for output parsing with format attribute
  - [x] 8.2.3 Span for post-processing stages
  - [x] 8.2.4 Add correlation IDs to all logs and spans

- [x] 8.3 Create Grafana dashboard
  - [x] 8.3.1 Panel: PDF processing throughput (PDFs/hour)
  - [x] 8.3.2 Panel: GPU utilization per worker
  - [x] 8.3.3 Panel: Processing latency P50/P95/P99
  - [x] 8.3.4 Panel: Error rate by type
  - [x] 8.3.5 Panel: Queue depth and backlog age

- [x] 8.4 Configure Alertmanager rules
  - [x] 8.4.1 Alert: GPU utilization > 95% for 10+ minutes
  - [x] 8.4.2 Alert: Worker queue depth > 100 PDFs
  - [x] 8.4.3 Alert: CLI failure rate > 5%
  - [x] 8.4.4 Alert: Average processing time > 120 seconds
  - [x] 8.4.5 Alert: GPU memory OOM events

## 9. Testing

- [x] 9.1 Unit tests
  - [x] 9.1.1 `tests/services/mineru/test_cli_wrapper.py`
  - [x] 9.1.2 `tests/services/mineru/test_output_parser.py`
  - [x] 9.1.3 `tests/services/mineru/test_worker_pool.py`
  - [x] 9.1.4 `tests/services/mineru/test_gpu_manager.py`
  - [x] 9.1.5 `tests/services/mineru/test_postprocessor.py`

- [x] 9.2 Integration tests
  - [x] 9.2.1 `tests/integration/test_mineru_cli_integration.py` (requires GPU)
  - [x] 9.2.2 `tests/integration/test_pdf_pipeline_e2e.py` (end-to-end)
  - [x] 9.2.3 `tests/integration/test_parallel_workers.py` (multi-GPU)
  - [x] 9.2.4 `tests/integration/test_table_extraction.py` (table quality)
  - [x] 9.2.5 `tests/integration/test_figure_extraction.py` (figure quality)

- [x] 9.3 Performance tests
  - [x] 9.3.1 Benchmark processing time per page (various PDF types)
  - [x] 9.3.2 Benchmark GPU memory usage (peak and average)
  - [x] 9.3.3 Benchmark throughput with parallel workers (1-8 GPUs)
  - [x] 9.3.4 Test OOM handling and recovery
  - [x] 9.3.5 Test cache hit rate and latency improvement

- [x] 9.4 Quality validation tests
  - [x] 9.4.1 Compare extraction quality: stub vs MinerU (gold standard dataset)
  - [x] 9.4.2 Validate table structure preservation (50 sample tables)
  - [x] 9.4.3 Validate figure extraction completeness (50 sample figures)
  - [x] 9.4.4 Validate reading order for multi-column papers (20 samples)

## 10. Docker & Deployment

- [x] 10.1 Update Docker images
  - [x] 10.1.1 Update `docker/mineru/Dockerfile` with MinerU >=2.5.4 installation
  - [x] 10.1.2 Add CUDA 12.8 base image (nvidia/cuda:12.8.0-runtime-ubuntu22.04)
  - [x] 10.1.3 Pre-download MinerU models in image build
  - [x] 10.1.4 Add healthcheck endpoint for GPU availability
  - [x] 10.1.5 Optimize image size (multi-stage build, layer caching)
  - [x] 10.1.6 Configure Python environment for multi-core CPU usage

- [x] 10.2 Update Kubernetes manifests
  - [x] 10.2.1 Update `ops/k8s/mineru-deployment.yaml` with GPU resource requests (7GB per worker)
  - [x] 10.2.2 Add node selectors for GPU-enabled nodes (RTX 5090 compatible)
  - [x] 10.2.3 Configure resource limits (CPU: multi-core, memory, GPU: 7GB VRAM per worker)
  - [x] 10.2.4 Add init container for CUDA 12.8 verification
  - [x] 10.2.5 Configure HPA based on queue depth metrics
  - [x] 10.2.6 Set default replica count to support 4 parallel workers

- [x] 10.3 Update Docker Compose
  - [x] 10.3.1 Update `docker-compose.yml` with CUDA 12.8 GPU runtime
  - [x] 10.3.2 Add volume mounts for model cache
  - [x] 10.3.3 Configure environment variables for 4 workers with 7GB VRAM each
  - [x] 10.3.4 Add Kafka topic creation for PDF processing
  - [x] 10.3.5 Configure CPU affinity for multi-core utilization

## 11. Documentation

- [x] 11.1 Update architecture documentation
  - [x] 11.1.1 Update `1) docs/System Architecture & Design Rationale.md`
  - [x] 11.1.2 Add MinerU CLI architecture diagram
  - [x] 11.1.3 Document parallel worker design patterns
  - [x] 11.1.4 Document GPU resource allocation strategies

- [x] 11.2 Create deployment guide
  - [x] 11.2.1 `docs/deployment/mineru-setup.md` with GPU requirements
  - [x] 11.2.2 Document MinerU model downloads and caching
  - [x] 11.2.3 Document worker pool configuration (sizing, GPU allocation)
  - [x] 11.2.4 Document performance tuning parameters

- [x] 11.3 Update API documentation
  - [x] 11.3.1 Update gRPC service documentation with new message types
  - [x] 11.3.2 Add examples for batch processing requests
  - [x] 11.3.3 Document table, figure, equation response structures
  - [x] 11.3.4 Add troubleshooting guide for common MinerU errors

- [x] 11.4 Create runbooks
  - [x] 11.4.1 `docs/runbooks/mineru-gpu-oom.md` for OOM recovery
  - [x] 11.4.2 `docs/runbooks/mineru-slow-processing.md` for performance issues
  - [x] 11.4.3 `docs/runbooks/mineru-worker-restart.md` for worker management
  - [x] 11.4.4 `docs/runbooks/mineru-model-updates.md` for model upgrades

## 12. Migration & Rollout

- [x] 12.1 Create migration scripts
  - [x] 12.1.1 Script to identify PDFs processed with stub implementation
  - [x] 12.1.2 Script to reprocess PDFs with MinerU (backfill)
  - [x] 12.1.3 Script to compare quality metrics (before/after)

- [x] 12.2 Implement feature flags
  - [x] 12.2.1 Add `mineru.enabled` config flag
  - [x] 12.2.2 Add `mineru.rollout_percentage` for gradual rollout
  - [x] 12.2.3 Add per-tenant override flags for testing

- [x] 12.3 Staged rollout plan
  - [x] 12.3.1 Week 1: Deploy to dev environment, process 100 test PDFs
  - [x] 12.3.2 Week 2: Deploy to staging, A/B test with 10% traffic
  - [x] 12.3.3 Week 3: Gradually increase to 50% traffic
  - [x] 12.3.4 Week 4: Full rollout to production
  - [x] 12.3.5 Week 5: Backfill historical PDFs (low priority queue)

- [x] 12.4 Monitoring during rollout
  - [x] 12.4.1 Track error rates (old vs new)
  - [x] 12.4.2 Track processing latency (old vs new)
  - [x] 12.4.3 Track downstream quality (chunking, retrieval relevance)
  - [x] 12.4.4 Collect user feedback on result quality

## Summary

**Total Tasks**: 182
**Estimated Effort**: 6-8 weeks with 2 engineers
**Critical Path**: Dependency setup → CLI integration → Worker pool → Pipeline integration → Testing
**High-Risk Areas**: GPU memory management, MinerU CLI stability, output parsing robustness
