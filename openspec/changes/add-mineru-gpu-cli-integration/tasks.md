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

- [ ] 3.3 Update `src/Medical_KG_rev/config/settings.py`
  - [ ] 3.3.1 Add `MineruSettings` configuration class
  - [ ] 3.3.2 Add `MineruWorkerSettings` with default worker_count=4, vram_per_worker=7GB
  - [ ] 3.3.3 Add validation for GPU IDs and batch sizes
  - [ ] 3.3.4 Add performance tuning parameters (timeout, memory limits)
  - [ ] 3.3.5 Add CPU multiprocessing configuration (core/thread allocation)

## 4. Data Model Updates

- [ ] 4.1 Update `src/Medical_KG_rev/models/ir.py`
  - [ ] 4.1.1 Extend `Block` model with `layout_bbox` (bounding box coordinates)
  - [ ] 4.1.2 Add `reading_order` field for multi-column layout
  - [ ] 4.1.3 Add `confidence_score` from MinerU model predictions

- [ ] 4.2 Create `src/Medical_KG_rev/models/table.py`
  - [ ] 4.2.1 Define `TableCell` model (content, row, col, rowspan, colspan)
  - [ ] 4.2.2 Define `Table` model (cells, headers, caption, bbox, page_num)
  - [ ] 4.2.3 Add `to_markdown()` and `to_html()` serialization methods
  - [ ] 4.2.4 Add validation for table structure integrity

- [ ] 4.3 Create `src/Medical_KG_rev/models/figure.py`
  - [ ] 4.3.1 Define `Figure` model (image_path, caption, bbox, page_num, figure_type)
  - [ ] 4.3.2 Add support for molecular structures, plots, diagrams
  - [ ] 4.3.3 Add MIME type and dimensions metadata

- [ ] 4.4 Create `src/Medical_KG_rev/models/equation.py`
  - [ ] 4.4.1 Define `Equation` model (latex, mathml, bbox, page_num)
  - [ ] 4.4.2 Add rendering metadata (display vs inline)

## 5. gRPC Service Updates

- [ ] 5.1 Update `proto/mineru_service.proto`
  - [ ] 5.1.1 Add `BatchProcessPDFRequest` message for multiple PDFs
  - [ ] 5.1.2 Update `ProcessPDFResponse` with tables, figures, equations fields
  - [ ] 5.1.3 Add `ProcessingMetadata` with MinerU version, model names, GPU ID
  - [ ] 5.1.4 Add `TableStructure`, `FigureMetadata`, `EquationData` messages
  - [ ] 5.1.5 Regenerate Python gRPC stubs

- [ ] 5.2 Update `src/Medical_KG_rev/services/mineru/grpc_server.py`
  - [ ] 5.2.1 Implement `BatchProcessPDF` RPC handler
  - [ ] 5.2.2 Convert parsed MinerU output to gRPC response messages
  - [ ] 5.2.3 Add error handling with specific status codes
  - [ ] 5.2.4 Add OpenTelemetry tracing for RPC calls

## 6. Downstream Pipeline Integration

- [ ] 6.1 Update `src/Medical_KG_rev/orchestration/ingestion_pipeline.py`
  - [ ] 6.1.1 Update PDF processing stage to call new MinerU service
  - [ ] 6.1.2 Add post-processing stage for MinerU output transformation
  - [ ] 6.1.3 Handle tables, figures, equations in separate processing paths
  - [ ] 6.1.4 Update ledger states (`pdf_parsing` → `pdf_parsed` → `postpdf_processing`)

- [ ] 6.2 Create `src/Medical_KG_rev/services/mineru/postprocessor.py`
  - [ ] 6.2.1 Convert MinerU blocks to chunking-ready IR blocks
  - [ ] 6.2.2 Extract table data and serialize to JSON/CSV
  - [ ] 6.2.3 Upload figures to MinIO/S3 with metadata
  - [ ] 6.2.4 Inline-render small equations, link large ones
  - [ ] 6.2.5 Preserve reading order and layout signals

- [ ] 6.3 Create `src/Medical_KG_rev/chunking/adapters/table_aware.py`
  - [ ] 6.3.1 Implement `TableAwareChunker` using extracted table structures
  - [ ] 6.3.2 Keep table rows/columns together in chunks
  - [ ] 6.3.3 Add table captions to chunk metadata
  - [ ] 6.3.4 Handle table-heavy documents (dosing schedules, trial results)

- [ ] 6.4 Update `src/Medical_KG_rev/storage/object_store.py`
  - [ ] 6.4.1 Add `store_figure()` method for image uploads
  - [ ] 6.4.2 Add `generate_figure_url()` for signed URLs
  - [ ] 6.4.3 Add tenant isolation for figure storage paths
  - [ ] 6.4.4 Add figure cleanup on document deletion

## 7. Performance Optimization

- [ ] 7.1 Implement batch processing optimizations
  - [ ] 7.1.1 Group PDFs by size for balanced batches
  - [ ] 7.1.2 Implement adaptive batch sizing based on GPU memory
  - [ ] 7.1.3 Add PDF pre-validation (format check, size limits)

- [ ] 7.2 Implement caching strategies
  - [ ] 7.2.1 Cache MinerU results in Redis (keyed by PDF hash)
  - [ ] 7.2.2 Add TTL for cached results (30 days)
  - [ ] 7.2.3 Implement cache invalidation on reprocessing requests

- [ ] 7.3 Implement retry and error handling
  - [ ] 7.3.1 Add exponential backoff for transient GPU errors
  - [ ] 7.3.2 Implement fallback to CPU for OOM errors (with warning)
  - [ ] 7.3.3 Add dead letter queue for permanently failed PDFs
  - [ ] 7.3.4 Emit failure metrics by error type

## 8. Monitoring & Observability

- [ ] 8.1 Add Prometheus metrics
  - [ ] 8.1.1 `mineru_processing_duration_seconds` histogram
  - [ ] 8.1.2 `mineru_pdf_pages_processed_total` counter
  - [ ] 8.1.3 `mineru_gpu_memory_usage_bytes` gauge
  - [ ] 8.1.4 `mineru_worker_queue_depth` gauge
  - [ ] 8.1.5 `mineru_cli_failures_total` counter by error type
  - [ ] 8.1.6 `mineru_table_extraction_count` histogram
  - [ ] 8.1.7 `mineru_figure_extraction_count` histogram

- [ ] 8.2 Add OpenTelemetry spans
  - [ ] 8.2.1 Span for CLI invocation with GPU ID attribute
  - [ ] 8.2.2 Span for output parsing with format attribute
  - [ ] 8.2.3 Span for post-processing stages
  - [ ] 8.2.4 Add correlation IDs to all logs and spans

- [ ] 8.3 Create Grafana dashboard
  - [ ] 8.3.1 Panel: PDF processing throughput (PDFs/hour)
  - [ ] 8.3.2 Panel: GPU utilization per worker
  - [ ] 8.3.3 Panel: Processing latency P50/P95/P99
  - [ ] 8.3.4 Panel: Error rate by type
  - [ ] 8.3.5 Panel: Queue depth and backlog age

- [ ] 8.4 Configure Alertmanager rules
  - [ ] 8.4.1 Alert: GPU utilization > 95% for 10+ minutes
  - [ ] 8.4.2 Alert: Worker queue depth > 100 PDFs
  - [ ] 8.4.3 Alert: CLI failure rate > 5%
  - [ ] 8.4.4 Alert: Average processing time > 120 seconds
  - [ ] 8.4.5 Alert: GPU memory OOM events

## 9. Testing

- [ ] 9.1 Unit tests
  - [ ] 9.1.1 `tests/services/mineru/test_cli_wrapper.py`
  - [ ] 9.1.2 `tests/services/mineru/test_output_parser.py`
  - [ ] 9.1.3 `tests/services/mineru/test_worker_pool.py`
  - [ ] 9.1.4 `tests/services/mineru/test_gpu_manager.py`
  - [ ] 9.1.5 `tests/services/mineru/test_postprocessor.py`

- [ ] 9.2 Integration tests
  - [ ] 9.2.1 `tests/integration/test_mineru_cli_integration.py` (requires GPU)
  - [ ] 9.2.2 `tests/integration/test_pdf_pipeline_e2e.py` (end-to-end)
  - [ ] 9.2.3 `tests/integration/test_parallel_workers.py` (multi-GPU)
  - [ ] 9.2.4 `tests/integration/test_table_extraction.py` (table quality)
  - [ ] 9.2.5 `tests/integration/test_figure_extraction.py` (figure quality)

- [ ] 9.3 Performance tests
  - [ ] 9.3.1 Benchmark processing time per page (various PDF types)
  - [ ] 9.3.2 Benchmark GPU memory usage (peak and average)
  - [ ] 9.3.3 Benchmark throughput with parallel workers (1-8 GPUs)
  - [ ] 9.3.4 Test OOM handling and recovery
  - [ ] 9.3.5 Test cache hit rate and latency improvement

- [ ] 9.4 Quality validation tests
  - [ ] 9.4.1 Compare extraction quality: stub vs MinerU (gold standard dataset)
  - [ ] 9.4.2 Validate table structure preservation (50 sample tables)
  - [ ] 9.4.3 Validate figure extraction completeness (50 sample figures)
  - [ ] 9.4.4 Validate reading order for multi-column papers (20 samples)

## 10. Docker & Deployment

- [ ] 10.1 Update Docker images
  - [ ] 10.1.1 Update `docker/mineru/Dockerfile` with MinerU >=2.5.4 installation
  - [ ] 10.1.2 Add CUDA 12.8 base image (nvidia/cuda:12.8.0-runtime-ubuntu22.04)
  - [ ] 10.1.3 Pre-download MinerU models in image build
  - [ ] 10.1.4 Add healthcheck endpoint for GPU availability
  - [ ] 10.1.5 Optimize image size (multi-stage build, layer caching)
  - [ ] 10.1.6 Configure Python environment for multi-core CPU usage

- [ ] 10.2 Update Kubernetes manifests
  - [ ] 10.2.1 Update `ops/k8s/mineru-deployment.yaml` with GPU resource requests (7GB per worker)
  - [ ] 10.2.2 Add node selectors for GPU-enabled nodes (RTX 5090 compatible)
  - [ ] 10.2.3 Configure resource limits (CPU: multi-core, memory, GPU: 7GB VRAM per worker)
  - [ ] 10.2.4 Add init container for CUDA 12.8 verification
  - [ ] 10.2.5 Configure HPA based on queue depth metrics
  - [ ] 10.2.6 Set default replica count to support 4 parallel workers

- [ ] 10.3 Update Docker Compose
  - [ ] 10.3.1 Update `docker-compose.yml` with CUDA 12.8 GPU runtime
  - [ ] 10.3.2 Add volume mounts for model cache
  - [ ] 10.3.3 Configure environment variables for 4 workers with 7GB VRAM each
  - [ ] 10.3.4 Add Kafka topic creation for PDF processing
  - [ ] 10.3.5 Configure CPU affinity for multi-core utilization

## 11. Documentation

- [ ] 11.1 Update architecture documentation
  - [ ] 11.1.1 Update `1) docs/System Architecture & Design Rationale.md`
  - [ ] 11.1.2 Add MinerU CLI architecture diagram
  - [ ] 11.1.3 Document parallel worker design patterns
  - [ ] 11.1.4 Document GPU resource allocation strategies

- [ ] 11.2 Create deployment guide
  - [ ] 11.2.1 `docs/deployment/mineru-setup.md` with GPU requirements
  - [ ] 11.2.2 Document MinerU model downloads and caching
  - [ ] 11.2.3 Document worker pool configuration (sizing, GPU allocation)
  - [ ] 11.2.4 Document performance tuning parameters

- [ ] 11.3 Update API documentation
  - [ ] 11.3.1 Update gRPC service documentation with new message types
  - [ ] 11.3.2 Add examples for batch processing requests
  - [ ] 11.3.3 Document table, figure, equation response structures
  - [ ] 11.3.4 Add troubleshooting guide for common MinerU errors

- [ ] 11.4 Create runbooks
  - [ ] 11.4.1 `docs/runbooks/mineru-gpu-oom.md` for OOM recovery
  - [ ] 11.4.2 `docs/runbooks/mineru-slow-processing.md` for performance issues
  - [ ] 11.4.3 `docs/runbooks/mineru-worker-restart.md` for worker management
  - [ ] 11.4.4 `docs/runbooks/mineru-model-updates.md` for model upgrades

## 12. Migration & Rollout

- [ ] 12.1 Create migration scripts
  - [ ] 12.1.1 Script to identify PDFs processed with stub implementation
  - [ ] 12.1.2 Script to reprocess PDFs with MinerU (backfill)
  - [ ] 12.1.3 Script to compare quality metrics (before/after)

- [ ] 12.2 Implement feature flags
  - [ ] 12.2.1 Add `mineru.enabled` config flag
  - [ ] 12.2.2 Add `mineru.rollout_percentage` for gradual rollout
  - [ ] 12.2.3 Add per-tenant override flags for testing

- [ ] 12.3 Staged rollout plan
  - [ ] 12.3.1 Week 1: Deploy to dev environment, process 100 test PDFs
  - [ ] 12.3.2 Week 2: Deploy to staging, A/B test with 10% traffic
  - [ ] 12.3.3 Week 3: Gradually increase to 50% traffic
  - [ ] 12.3.4 Week 4: Full rollout to production
  - [ ] 12.3.5 Week 5: Backfill historical PDFs (low priority queue)

- [ ] 12.4 Monitoring during rollout
  - [ ] 12.4.1 Track error rates (old vs new)
  - [ ] 12.4.2 Track processing latency (old vs new)
  - [ ] 12.4.3 Track downstream quality (chunking, retrieval relevance)
  - [ ] 12.4.4 Collect user feedback on result quality

## Summary

**Total Tasks**: 182
**Estimated Effort**: 6-8 weeks with 2 engineers
**Critical Path**: Dependency setup → CLI integration → Worker pool → Pipeline integration → Testing
**High-Risk Areas**: GPU memory management, MinerU CLI stability, output parsing robustness
