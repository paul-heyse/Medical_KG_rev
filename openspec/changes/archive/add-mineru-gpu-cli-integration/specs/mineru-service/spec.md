# Specification: MinerU GPU Service

## ADDED Requirements

### Requirement: MinerU CLI Integration

The system SHALL integrate with the official MinerU library (version ≥2.5.4) using its built-in CLI command for PDF processing.

#### Scenario: Built-in CLI invocation

- **WHEN** a PDF processing request is received
- **THEN** the system SHALL invoke the `mineru` CLI command (not a custom-built CLI)
- **AND** pass appropriate arguments for output format, VRAM limits, and extraction options
- **AND** capture stdout/stderr for logging and error handling

#### Scenario: Version validation

- **WHEN** the service starts
- **THEN** the system SHALL verify MinerU version is ≥2.5.4
- **AND** fail-fast if version requirement not met
- **AND** log the installed MinerU version

### Requirement: GPU Resource Management

The system SHALL manage GPU resources with specific VRAM allocation per worker to prevent out-of-memory errors.

#### Scenario: Worker VRAM allocation

- **WHEN** a worker process is initialized
- **THEN** the system SHALL allocate a maximum of 7GB VRAM per worker
- **AND** configure the MinerU CLI with `--vram-limit 7G` argument
- **AND** assign GPU device ID via `CUDA_VISIBLE_DEVICES` environment variable

#### Scenario: Multi-worker configuration

- **WHEN** the worker pool is initialized
- **THEN** the system SHALL create 4 parallel workers by default
- **AND** assign each worker to the same GPU (RTX 5090 with 32GB VRAM)
- **AND** ensure total VRAM allocation does not exceed 28GB (leaving 4GB overhead)

#### Scenario: CUDA version validation

- **WHEN** the service starts
- **THEN** the system SHALL verify CUDA version is 12.8
- **AND** fail-fast if CUDA version mismatch detected
- **AND** log CUDA version and GPU device information

### Requirement: CPU Optimization

The system SHALL optimize CPU utilization through multiprocessing to prevent CPU bottleneck during PDF processing.

#### Scenario: Multiprocessing configuration

- **WHEN** the worker pool is initialized
- **THEN** the system SHALL use Python's `multiprocessing.Pool` with 4 worker processes
- **AND** configure environment variables for numpy/scipy optimization (OMP_NUM_THREADS, MKL_NUM_THREADS)
- **AND** allow OS scheduler to manage CPU core assignment (no manual affinity)

#### Scenario: CPU bottleneck prevention

- **WHEN** workers are processing PDFs
- **THEN** the system SHALL utilize multiple CPU cores for parallel post-processing
- **AND** monitor CPU utilization metrics
- **AND** emit alerts if CPU utilization consistently below 60% (indicating bottleneck)

### Requirement: Subprocess Isolation

The system SHALL execute MinerU CLI in isolated subprocesses with resource limits and timeout enforcement.

#### Scenario: Subprocess execution

- **WHEN** a worker invokes MinerU CLI
- **THEN** the system SHALL run `mineru` command in separate subprocess
- **AND** set subprocess timeout to 300 seconds (configurable)
- **AND** capture stdout and stderr for logging
- **AND** terminate subprocess on timeout or parent process exit

#### Scenario: Environment isolation

- **WHEN** a subprocess is created for MinerU CLI
- **THEN** the system SHALL set `CUDA_VISIBLE_DEVICES` to assigned GPU ID
- **AND** inherit other environment variables from parent process
- **AND** set working directory to isolated temp location

#### Scenario: Error handling

- **WHEN** MinerU CLI subprocess fails
- **THEN** the system SHALL capture exit code and error output
- **AND** retry with exponential backoff for transient errors (3 attempts)
- **AND** move PDF to dead letter queue after max retries
- **AND** emit failure metrics by error type

### Requirement: Structured Output Parsing

The system SHALL parse MinerU CLI output (JSON format) into structured IR Block, Table, Figure, and Equation objects.

#### Scenario: JSON output parsing

- **WHEN** MinerU CLI completes successfully
- **THEN** the system SHALL read JSON output file from specified directory
- **AND** validate JSON structure against expected schema
- **AND** parse text blocks with bounding boxes and confidence scores
- **AND** parse table structures with cells, headers, and captions

#### Scenario: Table extraction

- **WHEN** a table is detected in MinerU output
- **THEN** the system SHALL create Table object with cell coordinates
- **AND** preserve header rows and column relationships
- **AND** extract table caption if present
- **AND** assign unique table ID with doc_id prefix

#### Scenario: Figure extraction

- **WHEN** a figure is detected in MinerU output
- **THEN** the system SHALL create Figure object with image path and caption
- **AND** upload figure image to object storage (MinIO/S3)
- **AND** generate signed URL for image access
- **AND** extract figure type (plot, diagram, molecular structure)

#### Scenario: Equation extraction

- **WHEN** an equation is detected in MinerU output
- **THEN** the system SHALL create Equation object with LaTeX representation
- **AND** extract MathML if available
- **AND** classify as display or inline equation
- **AND** preserve equation bounding box coordinates

### Requirement: Parallel Worker Pool

The system SHALL implement a parallel worker pool with Kafka integration for distributed PDF processing.

#### Scenario: Worker pool initialization

- **WHEN** the service starts
- **THEN** the system SHALL create 4 worker processes (configurable)
- **AND** assign each worker a unique worker_id
- **AND** validate GPU availability for each worker
- **AND** subscribe to `pdf.parse.requests.v1` Kafka topic

#### Scenario: Job distribution

- **WHEN** a PDF processing request arrives on Kafka topic
- **THEN** the system SHALL assign job to next available worker
- **AND** track worker status (idle, processing, error)
- **AND** emit queue depth metrics per worker

#### Scenario: Worker failure recovery

- **WHEN** a worker process crashes
- **THEN** the system SHALL restart worker process automatically
- **AND** reassign failed job to another worker
- **AND** emit worker failure metrics
- **AND** log crash details for debugging

### Requirement: Provenance Tracking

The system SHALL track complete provenance metadata for all PDF processing operations.

#### Scenario: Processing metadata capture

- **WHEN** a PDF is processed successfully
- **THEN** the system SHALL record MinerU version used
- **AND** record model names (layout, table, figure detection models)
- **AND** record GPU device ID and worker ID
- **AND** record processing start and end timestamps
- **AND** record total processing duration

#### Scenario: Provenance in output

- **WHEN** processing results are published
- **THEN** the system SHALL include provenance metadata in output
- **AND** attach provenance to each extracted Block, Table, Figure
- **AND** enable tracing from output back to original PDF and processing job

### Requirement: Performance Monitoring

The system SHALL emit comprehensive metrics for GPU utilization, processing latency, and throughput.

#### Scenario: GPU metrics

- **WHEN** workers are processing PDFs
- **THEN** the system SHALL emit `mineru_gpu_memory_usage_bytes` gauge per GPU
- **AND** emit `mineru_processing_duration_seconds` histogram per worker
- **AND** track GPU utilization percentage per worker

#### Scenario: Throughput metrics

- **WHEN** processing operations complete
- **THEN** the system SHALL increment `mineru_pdf_pages_processed_total` counter
- **AND** emit `mineru_table_extraction_count` histogram
- **AND** emit `mineru_figure_extraction_count` histogram

#### Scenario: Error metrics

- **WHEN** processing errors occur
- **THEN** the system SHALL increment `mineru_cli_failures_total` counter by error type
- **AND** distinguish transient vs permanent failures
- **AND** emit metrics for timeout, OOM, and CLI crash errors

### Requirement: Configuration Management

The system SHALL support comprehensive configuration for worker count, VRAM allocation, CUDA version, and CPU optimization.

#### Scenario: Default configuration

- **WHEN** the service starts without custom configuration
- **THEN** the system SHALL use 4 workers by default
- **AND** allocate 7GB VRAM per worker
- **AND** require CUDA version 12.8
- **AND** enable multiprocessing for CPU optimization

#### Scenario: Configuration validation

- **WHEN** configuration is loaded
- **THEN** the system SHALL validate worker_count × vram_per_worker_gb ≤ total_gpu_vram
- **AND** validate CUDA version matches requirement
- **AND** validate MinerU version ≥2.5.4
- **AND** fail-fast if validation errors detected

#### Scenario: Runtime reconfiguration

- **WHEN** configuration is updated via feature flag
- **THEN** the system SHALL support graceful worker pool resize
- **AND** drain existing jobs before applying new configuration
- **AND** restart workers with new settings
- **AND** emit metrics for configuration changes

## Dependencies

- `mineru[gpu]>=2.5.4` - Official MinerU library with GPU support
- `CUDA 12.8` - GPU compute platform
- `kafka-python` - Kafka consumer/producer integration
- `multiprocessing` (stdlib) - Parallel worker processes
- `subprocess` (stdlib) - CLI invocation
- `psutil` - Process and GPU monitoring
