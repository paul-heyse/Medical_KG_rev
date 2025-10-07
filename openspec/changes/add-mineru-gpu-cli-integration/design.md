# Technical Design: MinerU GPU CLI Integration

## Context

The current PDF processing implementation in `src/Medical_KG_rev/services/mineru/service.py` is a lightweight stub that cannot extract structured information from complex biomedical PDFs. This design describes the integration of the official MinerU library (v2.5.4+) using its built-in CLI with parallel GPU workers.

**Key Constraints**:

- Must use MinerU's built-in `mineru` CLI command (building custom CLI is impractical)
- Hardware: RTX 5090 (32GB VRAM) with CUDA 12.8
- Default configuration: 4 parallel workers @ 7GB VRAM each
- Must prevent CPU bottleneck through multiprocessing optimization
- No vLLM integration (use standard GPU pipeline only)

## Goals / Non-Goals

### Goals

1. **Leverage official MinerU capabilities**: Use the battle-tested `mineru` CLI for PDF processing
2. **Parallel GPU processing**: Achieve high throughput with 4 concurrent workers
3. **Resource optimization**: Efficient VRAM allocation (7GB per worker) and CPU utilization
4. **Structured data extraction**: Tables, figures, equations with provenance
5. **Fail-fast operation**: Validate CUDA 12.8 and GPU availability on startup
6. **Seamless downstream integration**: Convert MinerU outputs to IR blocks for chunking pipeline

### Non-Goals

- Custom CLI implementation (use MinerU's existing CLI)
- vLLM integration for embeddings/generation
- CPU-only fallback (GPU-only with fail-fast policy)
- Multi-language support (English-first)
- Real-time processing (batch-oriented design)

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    Kafka Topic: pdf.parse.requests.v1           │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Worker Pool Manager                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Job Queue (Priority-based)                              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────┬───────────┬───────────┬───────────┬───────────────────┘
          │           │           │           │
  ┌───────▼───┐ ┌─────▼────┐ ┌───▼──────┐ ┌──▼────────┐
  │ Worker 1  │ │ Worker 2 │ │ Worker 3 │ │ Worker 4  │
  │ GPU:0     │ │ GPU:0    │ │ GPU:0    │ │ GPU:0     │
  │ 7GB VRAM  │ │ 7GB VRAM │ │ 7GB VRAM │ │ 7GB VRAM  │
  └─────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘
        │            │            │             │
        │     Subprocess: mineru CLI            │
        │     (CUDA 12.8, isolated)             │
        │                                       │
        └───────────────┬───────────────────────┘
                        │
                        ▼
          ┌─────────────────────────────┐
          │  MinerU Output Parser        │
          │  (JSON/Markdown → IR Blocks) │
          └──────────────┬───────────────┘
                         │
         ┌───────────────┴────────────────┐
         │                                │
         ▼                                ▼
┌─────────────────┐            ┌──────────────────┐
│ Post-Processor  │            │ Object Storage   │
│ (Blocks, Tables)│            │ (Figures, Images)│
└────────┬────────┘            └──────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  Chunking Service                        │
│  (Table-aware, Figure-aware strategies)  │
└──────────────────────────────────────────┘
```

## Key Design Decisions

### Decision 1: Use MinerU's Built-in CLI

**Rationale**:

- MinerU provides a well-tested `mineru` command-line interface
- Building a custom CLI would duplicate functionality and introduce maintenance burden
- The official CLI handles model loading, GPU initialization, and output formatting

**Implementation**:

- Wrap `mineru` command in subprocess with proper environment variables
- Use `CUDA_VISIBLE_DEVICES` to assign specific GPU to each worker
- Pass MinerU CLI arguments for output format, VRAM limits, batch processing

**Alternatives Considered**:

- **Custom Python API wrapper**: Would require deep understanding of MinerU internals; rejected due to complexity
- **Direct library import**: MinerU's internal APIs are not stable; rejected for maintainability

### Decision 2: 4 Workers @ 7GB VRAM Each

**Rationale**:

- RTX 5090 has 32GB VRAM total
- 4 workers × 7GB = 28GB allocated, leaving 4GB for system overhead
- Prevents OOM while maximizing parallelism
- Each worker can handle typical biomedical PDFs (10-50 pages)

**Configuration**:

```yaml
workers:
  count: 4
  vram_per_worker_gb: 7
```

**Implementation**:

- MinerU CLI accepts VRAM limit via command-line argument
- Worker pool distributes jobs round-robin to available workers
- GPU memory monitoring triggers alerts if usage exceeds 90%

**Alternatives Considered**:

- **8 workers @ 3.5GB**: Too many context switches, lower per-worker performance
- **2 workers @ 14GB**: Underutilizes GPU, lower throughput

### Decision 3: Multi-core CPU Optimization

**Rationale**:

- MinerU's post-processing (layout analysis, table parsing) is CPU-intensive
- Without multiprocessing, CPU becomes bottleneck even with GPU acceleration
- Python's GIL limits single-process performance

**Implementation**:

- Use `multiprocessing.Pool` for parallel worker processes
- Each worker process runs on separate CPU cores (no manual affinity)
- Configure `OMP_NUM_THREADS` and `MKL_NUM_THREADS` for numpy/scipy optimization

```python
import multiprocessing as mp
import os

os.environ['OMP_NUM_THREADS'] = '4'  # Per worker
os.environ['MKL_NUM_THREADS'] = '4'

worker_pool = mp.Pool(processes=4, maxtasksperchild=100)
```

**Alternatives Considered**:

- **Threading**: Ineffective due to GIL
- **Manual CPU affinity**: OS scheduler is more efficient; rejected

### Decision 4: Subprocess Isolation for MinerU CLI

**Rationale**:

- MinerU CLI runs as separate process with resource limits
- Prevents GPU memory leaks from affecting other services
- Enables timeout enforcement and graceful termination

**Implementation**:

```python
import subprocess
import os

def invoke_mineru(pdf_path: str, output_dir: str, gpu_id: int, vram_gb: int) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        'mineru',
        '--input', pdf_path,
        '--output', output_dir,
        '--format', 'json',
        '--vram-limit', f'{vram_gb}G',
        '--extract-tables',
        '--extract-figures',
        '--extract-equations'
    ]

    return subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        check=True
    )
```

**Error Handling**:

- `subprocess.TimeoutExpired`: Move PDF to DLQ, log timeout
- `subprocess.CalledProcessError`: Retry with exponential backoff (3 attempts)
- GPU OOM: Reduce batch size dynamically

### Decision 5: JSON Output Format with Structured Parsing

**Rationale**:

- MinerU's JSON output is machine-readable and comprehensive
- Contains bounding boxes, confidence scores, layout hierarchy
- Easier to parse than Markdown for structured extraction

**Output Structure** (MinerU JSON schema):

```json
{
  "doc_id": "...",
  "pages": [
    {
      "page_num": 1,
      "blocks": [
        {
          "type": "text",
          "bbox": [x1, y1, x2, y2],
          "text": "...",
          "confidence": 0.95
        },
        {
          "type": "table",
          "bbox": [x1, y1, x2, y2],
          "cells": [...],
          "caption": "..."
        },
        {
          "type": "figure",
          "bbox": [x1, y1, x2, y2],
          "image_path": "...",
          "caption": "..."
        }
      ]
    }
  ]
}
```

**Parser Implementation**:

```python
class MineruOutputParser:
    def parse_json(self, json_path: Path) -> ParsedDocument:
        with open(json_path) as f:
            data = json.load(f)

        blocks = []
        tables = []
        figures = []

        for page in data['pages']:
            for block in page['blocks']:
                if block['type'] == 'text':
                    blocks.append(self._parse_text_block(block, page['page_num']))
                elif block['type'] == 'table':
                    tables.append(self._parse_table(block, page['page_num']))
                elif block['type'] == 'figure':
                    figures.append(self._parse_figure(block, page['page_num']))

        return ParsedDocument(
            blocks=blocks,
            tables=tables,
            figures=figures,
            provenance=self._build_provenance(data)
        )
```

### Decision 6: Worker Pool with Kafka Integration

**Rationale**:

- Decouples PDF ingestion from processing
- Enables horizontal scaling and fault tolerance
- Kafka provides durable job queue with replay capability

**Worker Lifecycle**:

1. Worker starts, validates CUDA 12.8 and GPU availability
2. Subscribes to `pdf.parse.requests.v1` Kafka topic
3. Consumes message (PDF reference + metadata)
4. Downloads PDF from MinIO/S3
5. Invokes MinerU CLI via subprocess
6. Parses output and converts to IR blocks
7. Uploads results and publishes to `pdf.parse.results.v1`
8. ACKs Kafka message

**Fault Tolerance**:

- Worker crash: Kafka message not ACKed, auto-retried
- GPU OOM: Caught by subprocess, worker restarts clean
- Transient failures: Exponential backoff (4s, 16s, 64s)

## Data Flow

### Input: PDF Document

```python
@dataclass
class PDFProcessingRequest:
    job_id: str
    doc_id: str
    tenant_id: str
    pdf_url: str  # MinIO/S3 signed URL
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Processing Stages

1. **PDF Download** (5-10s):
   - Fetch from object storage
   - Validate format and size (< 100MB)
   - Store in worker temp directory

2. **MinerU CLI Invocation** (20-50s):
   - Run `mineru` command with GPU assignment
   - Monitor GPU memory usage
   - Capture stdout/stderr for logging

3. **Output Parsing** (2-5s):
   - Parse JSON/Markdown output
   - Extract blocks, tables, figures, equations
   - Validate structure and completeness

4. **Post-Processing** (5-10s):
   - Convert to IR Block objects
   - Upload figures to object storage
   - Serialize tables to structured format
   - Add provenance metadata

5. **Result Publishing** (1-2s):
   - Publish to `pdf.parse.results.v1` Kafka topic
   - Update job ledger state
   - Emit metrics

### Output: Parsed Document

```python
@dataclass
class ParsedDocument:
    doc_id: str
    blocks: list[Block]
    tables: list[Table]
    figures: list[Figure]
    equations: list[Equation]
    provenance: ProcessingProvenance

@dataclass
class ProcessingProvenance:
    mineru_version: str
    model_names: dict[str, str]  # layout, table, figure models
    gpu_id: int
    worker_id: str
    processing_time_seconds: float
    timestamp: datetime
```

## Performance Characteristics

### Expected Throughput

| PDF Type | Pages | Processing Time | Throughput (per worker) |
|----------|-------|-----------------|-------------------------|
| Simple text | 5-10 | 15-25s | 144-240 PDFs/hour |
| Scientific paper | 10-20 | 30-50s | 72-120 PDFs/hour |
| Complex tables | 20-50 | 50-90s | 40-72 PDFs/hour |

**Total throughput (4 workers)**: 160-960 PDFs/hour depending on complexity

### Resource Usage

- **GPU Memory**: 7GB per worker (28GB total)
- **CPU**: ~80% utilization across all cores
- **Disk I/O**: 50-100 MB/s for PDF read + image write
- **Network**: 10-20 MB/s for Kafka + object storage

## Monitoring & Observability

### Prometheus Metrics

```python
mineru_processing_duration_seconds = Histogram(
    'mineru_processing_duration_seconds',
    'PDF processing duration',
    labelnames=['gpu_id', 'status'],
    buckets=[10, 20, 30, 60, 120, 300]
)

mineru_gpu_memory_usage_bytes = Gauge(
    'mineru_gpu_memory_usage_bytes',
    'GPU memory usage',
    labelnames=['gpu_id', 'worker_id']
)

mineru_worker_queue_depth = Gauge(
    'mineru_worker_queue_depth',
    'Number of queued PDFs',
    labelnames=['worker_id']
)

mineru_cli_failures_total = Counter(
    'mineru_cli_failures_total',
    'CLI invocation failures',
    labelnames=['gpu_id', 'error_type']
)

mineru_table_extraction_count = Histogram(
    'mineru_table_extraction_count',
    'Tables extracted per PDF',
    buckets=[0, 1, 2, 5, 10, 20]
)
```

### OpenTelemetry Spans

- `mineru.cli.invoke` - CLI subprocess execution
- `mineru.output.parse` - Output parsing
- `mineru.postprocess` - IR conversion
- `mineru.upload.figures` - Figure storage

### Alerting Rules

```yaml
- alert: MinerUHighQueueDepth
  expr: mineru_worker_queue_depth > 100
  for: 10m
  annotations:
    summary: "MinerU queue depth exceeds 100 PDFs"

- alert: MinerUGPUUtilizationHigh
  expr: mineru_gpu_memory_usage_bytes / (7 * 1024^3) > 0.95
  for: 10m
  annotations:
    summary: "GPU VRAM usage exceeds 95%"

- alert: MinerUHighFailureRate
  expr: rate(mineru_cli_failures_total[5m]) > 0.05
  annotations:
    summary: "CLI failure rate exceeds 5%"
```

## Risks & Trade-offs

### Risk 1: MinerU CLI Stability

**Mitigation**:

- Pin to specific version (2.5.4+) in requirements
- Comprehensive integration tests with real PDFs
- Monitor CLI crash rate and update version as needed

### Risk 2: GPU Memory Leaks

**Mitigation**:

- Run MinerU CLI in isolated subprocess (automatic cleanup)
- Worker restart after N successful jobs (clear any leaks)
- Monitor GPU memory trends, alert on gradual increase

### Risk 3: CPU Bottleneck

**Mitigation**:

- Multiprocessing with separate worker processes
- Profile CPU usage and tune core allocation
- Consider upgrading CPU if bottleneck persists

### Trade-off: Latency vs Throughput

- **Design choice**: Optimize for throughput (batch processing)
- **Impact**: Individual PDF latency 30-90s (acceptable for biomedical use case)
- **Alternative**: Real-time processing would require different architecture (streaming, dedicated workers)

## Migration Strategy

### Phase 1: Parallel Implementation (Week 1-2)

- Implement new MinerU service alongside stub
- Route 10% of traffic to new service (feature flag)
- Compare quality and performance metrics

### Phase 2: Gradual Rollout (Week 3-4)

- Increase to 50% traffic if quality metrics positive
- Monitor error rates, latency, GPU utilization
- A/B test retrieval quality (downstream impact)

### Phase 3: Full Cutover (Week 5)

- Route 100% traffic to new MinerU service
- Keep stub implementation for emergency fallback
- Backfill historical PDFs (low priority queue)

### Rollback Plan

```yaml
mineru:
  enabled: false  # Reverts to stub implementation
```

Docker images tagged with versions for quick rollback:

- `mineru-service:stub` (current)
- `mineru-service:2.5.4` (new)

## Open Questions

1. **Model caching**: Should we pre-download all MinerU models in Docker image or lazy-load?
   - **Resolution**: Pre-download in Docker build to ensure availability

2. **Batch size tuning**: What's optimal batch size per worker?
   - **Resolution**: Start with batch_size=8, tune based on GPU memory usage

3. **Output format**: JSON vs Markdown for MinerU output?
   - **Resolution**: JSON for structured parsing, Markdown as fallback

4. **Worker restart policy**: How often should workers restart to clear memory leaks?
   - **Resolution**: Restart after 100 successful jobs or 4 hours, whichever comes first

## References

- [MinerU GitHub](https://github.com/opendatalab/MinerU)
- [MinerU Documentation](https://mineru.readthedocs.io/)
- CUDA 12.8 Documentation
- PyTorch Multiprocessing Best Practices
- Kafka Consumer Group Management
