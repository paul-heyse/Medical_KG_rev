# Specification: PDF Processing Pipeline Integration

## ADDED Requirements

### Requirement: Downstream Pipeline Integration

The system SHALL integrate MinerU processing outputs into the existing ingestion pipeline for seamless chunking, embedding, and retrieval.

#### Scenario: Pipeline stage updates

- **WHEN** a PDF parsing job completes
- **THEN** the system SHALL update job ledger state from `pdf_parsing` to `pdf_parsed`
- **AND** trigger post-processing stage for output transformation
- **AND** publish to `pdf.parse.results.v1` Kafka topic with structured results

#### Scenario: IR Block conversion

- **WHEN** MinerU output is parsed
- **THEN** the system SHALL convert text blocks to IR Block objects
- **AND** preserve bounding box coordinates in Block metadata
- **AND** maintain reading order for multi-column layouts
- **AND** attach provenance (MinerU version, model names, confidence scores)

#### Scenario: Multi-path processing

- **WHEN** post-processing stage receives parsed PDF
- **THEN** the system SHALL route text blocks to chunking service
- **AND** route tables to table-specific processing
- **AND** route figures to object storage upload
- **AND** route equations to inline rendering or linking

### Requirement: Table-Aware Chunking

The system SHALL leverage extracted table structures for improved table-aware chunking strategies.

#### Scenario: Table structure preservation

- **WHEN** a table is processed for chunking
- **THEN** the system SHALL keep table rows together in single chunk
- **AND** include table caption in chunk metadata
- **AND** preserve cell coordinates for span grounding
- **AND** tag chunk with `is_table=true` metadata

#### Scenario: Table-heavy documents

- **WHEN** a document contains multiple tables (e.g., dosing schedules, trial results)
- **THEN** the system SHALL create separate chunks per table
- **AND** include table headers in each chunk
- **AND** link related tables via metadata (e.g., arm_id, endpoint_id)

#### Scenario: Mixed content chunking

- **WHEN** text and tables are interspersed in document
- **THEN** the system SHALL respect table boundaries during chunking
- **AND** never split tables across chunks
- **AND** create separate chunks for table-adjacent paragraphs

### Requirement: Figure Metadata Integration

The system SHALL integrate extracted figure metadata into document blocks for cross-referencing and citation.

#### Scenario: Figure storage

- **WHEN** a figure is extracted from PDF
- **THEN** the system SHALL upload figure image to object storage (MinIO/S3)
- **AND** store in tenant-isolated path: `{tenant_id}/figures/{doc_id}/{figure_id}.{ext}`
- **AND** generate signed URL with 7-day expiration
- **AND** store figure metadata in separate Figure object

#### Scenario: Figure linking

- **WHEN** text blocks reference figures (e.g., "see Figure 2")
- **THEN** the system SHALL create cross-reference links in Block metadata
- **AND** include figure_id and signed URL
- **AND** enable retrieval systems to fetch figure along with text

#### Scenario: Figure caption indexing

- **WHEN** a figure with caption is extracted
- **THEN** the system SHALL create separate Block for caption text
- **AND** link caption Block to figure via figure_id
- **AND** enable semantic search over figure captions

### Requirement: Equation Rendering

The system SHALL render extracted equations inline or link to external representation based on complexity.

#### Scenario: Inline equation rendering

- **WHEN** a simple equation is extracted (LaTeX length < 100 chars)
- **THEN** the system SHALL render equation inline in Block text
- **AND** include LaTeX source in Block metadata
- **AND** preserve equation bounding box for span grounding

#### Scenario: Complex equation linking

- **WHEN** a complex equation is extracted (LaTeX length ≥ 100 chars)
- **THEN** the system SHALL store equation in separate Equation object
- **AND** insert placeholder in Block text with equation_id
- **AND** enable retrieval systems to fetch equation on demand

#### Scenario: Equation metadata

- **WHEN** an equation is processed
- **THEN** the system SHALL classify as display or inline equation
- **AND** extract surrounding context (±2 sentences)
- **AND** store MathML representation if available

### Requirement: Post-Processing Transformation

The system SHALL transform MinerU outputs into chunking-ready IR blocks with layout signals and structure preservation.

#### Scenario: Reading order preservation

- **WHEN** multi-column PDF is processed
- **THEN** the system SHALL order blocks by reading sequence (not spatial order)
- **AND** use MinerU's layout analysis results
- **AND** validate reading order with heuristics (left-to-right, top-to-bottom)

#### Scenario: Layout signal extraction

- **WHEN** blocks are converted to IR format
- **THEN** the system SHALL extract layout signals (font size, bold, italic, position)
- **AND** use signals to improve heading detection
- **AND** use signals to identify section boundaries
- **AND** pass layout signals to semantic chunking strategies

#### Scenario: Section hierarchy reconstruction

- **WHEN** document headings are detected
- **THEN** the system SHALL reconstruct section hierarchy
- **AND** assign title_path to each Block (e.g., ["Methods", "Statistical Analysis"])
- **AND** enable section-aware chunking strategies
- **AND** preserve IMRaD structure for scientific papers

### Requirement: Quality Validation

The system SHALL validate extraction quality and completeness before proceeding to downstream stages.

#### Scenario: Extraction completeness check

- **WHEN** MinerU processing completes
- **THEN** the system SHALL verify output contains at least one block
- **AND** verify page count matches input PDF
- **AND** verify total extracted text length > 100 characters
- **AND** emit warning if no tables/figures extracted from multi-page PDF

#### Scenario: Structure validation

- **WHEN** tables are extracted
- **THEN** the system SHALL validate table structure (rows > 0, columns > 0)
- **AND** validate cell coordinates within table bounds
- **AND** validate header detection (at least one header row)
- **AND** reject malformed tables with error logging

#### Scenario: Quality metrics

- **WHEN** extraction completes successfully
- **THEN** the system SHALL compute quality metrics (blocks per page, tables per page, figures per page)
- **AND** emit metrics to Prometheus
- **AND** flag low-quality extractions (e.g., < 50 chars per page)

### Requirement: Error Recovery

The system SHALL implement robust error recovery for partial failures and degraded outputs.

#### Scenario: Partial extraction success

- **WHEN** MinerU extracts text but fails on tables/figures
- **THEN** the system SHALL proceed with text-only processing
- **AND** emit warning metrics for failed extraction types
- **AND** enable manual retry for table/figure extraction

#### Scenario: Corrupted PDF handling

- **WHEN** MinerU CLI reports corrupted or unreadable PDF
- **THEN** the system SHALL mark job as permanently failed
- **AND** move PDF to dead letter queue for manual inspection
- **AND** emit error metrics with failure reason
- **AND** notify user via audit log

#### Scenario: Timeout recovery

- **WHEN** MinerU CLI times out (> 300 seconds)
- **THEN** the system SHALL terminate subprocess gracefully
- **AND** retry with reduced batch size if batch job
- **AND** move to DLQ after 3 timeout attempts
- **AND** emit timeout breach metrics

### Requirement: Batch Processing Optimization

The system SHALL optimize throughput through intelligent batching and resource allocation.

#### Scenario: Batch size adaptation

- **WHEN** worker processes PDFs
- **THEN** the system SHALL adjust batch size based on PDF complexity
- **AND** reduce batch size if GPU memory usage > 90%
- **AND** increase batch size if GPU memory usage < 50%
- **AND** maintain batch size between 1-16 PDFs

#### Scenario: Priority-based scheduling

- **WHEN** multiple PDF jobs are queued
- **THEN** the system SHALL prioritize jobs by priority field (0-10)
- **AND** process high-priority jobs (priority ≥ 7) first
- **AND** balance priorities to prevent starvation
- **AND** emit queue age metrics by priority level

#### Scenario: Load balancing

- **WHEN** workers have unequal queue depths
- **THEN** the system SHALL distribute new jobs to least-loaded worker
- **AND** rebalance jobs if worker becomes idle
- **AND** consider worker processing speed in load calculations

### Requirement: Integration Testing

The system SHALL provide comprehensive integration tests for end-to-end PDF processing pipeline.

#### Scenario: Sample PDF processing

- **WHEN** integration tests run
- **THEN** the system SHALL process sample PDFs from each type (clinical trial, drug label, research paper)
- **AND** validate block count, table count, figure count
- **AND** validate reading order correctness
- **AND** validate chunking output quality

#### Scenario: Quality comparison

- **WHEN** integration tests run
- **THEN** the system SHALL compare extraction quality: stub vs MinerU
- **AND** measure table extraction accuracy (precision, recall)
- **AND** measure figure extraction completeness
- **AND** validate improvements in downstream retrieval quality

#### Scenario: Performance benchmarking

- **WHEN** performance tests run
- **THEN** the system SHALL measure end-to-end processing time per PDF
- **AND** measure GPU memory usage patterns
- **AND** measure CPU utilization during processing
- **AND** validate throughput meets target (50-100 PDFs/hour per GPU)

## Dependencies

- `mineru-service` specification (core PDF processing)
- `chunking-system` specification (table-aware, figure-aware strategies)
- `ingestion-orchestration` specification (pipeline stages, job ledger)
- `object-storage` (figure/image upload)
- `kafka` (pipeline event publishing)
