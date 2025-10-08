# Parsing Capability: Spec Delta

## ADDED Requirements

### Requirement: MinerU Two-Phase Gate with Explicit Resume

The system SHALL enforce a manual two-phase gate for PDF processing where MinerU produces IR on GPU, the pipeline HALTS at ledger state `pdf_ir_ready`, and processing resumes only upon explicit `postpdf-start` trigger.

**Rationale**: Enables quality inspection after OCR, enforces GPU-only policy, prevents silent CPU fallbacks, maintains audit trail for compliance.

#### Scenario: PDF processing halts after MinerU success

- **GIVEN** a PDF job submitted via `/v1/ingest/pmc` with `include_pdf=true`
- **WHEN** MinerU successfully processes the PDF and produces structured IR with page/bbox maps
- **THEN** the ledger is updated to `pdf_ir_ready=true`
- **AND** the pipeline HALTS (does not proceed to chunking/embedding)
- **AND** a CloudEvent `mineru.gate.waiting` is emitted

#### Scenario: Explicit postpdf-start resumes chunking

- **GIVEN** a job in ledger state `pdf_ir_ready=true`
- **WHEN** an operator calls `POST /v1/jobs/{job_id}/postpdf-start`
- **THEN** the ledger is updated to `postpdf_start_triggered=true`
- **AND** the pipeline resumes at the chunking stage
- **AND** a CloudEvent `postpdf.start.triggered` is emitted with `trigger_source="manual"`

#### Scenario: MinerU failure on GPU unavailability

- **GIVEN** a PDF job submitted when GPU is unavailable (torch.cuda.is_available()=False)
- **WHEN** MinerU service attempts to process the PDF
- **THEN** the service raises `GpuNotAvailableError("MinerU requires GPU")`
- **AND** the ledger is updated to `status="mineru_failed", error="GPU unavailable"`
- **AND** the pipeline aborts (no CPU fallback attempted)

#### Scenario: Dagster sensor auto-triggers postpdf-start

- **GIVEN** a job in ledger state `pdf_ir_ready=true` for 5 minutes
- **WHEN** the Dagster `pdf_ir_ready_sensor` polls the ledger
- **THEN** the sensor triggers `postpdf-start` automatically
- **AND** the ledger records `trigger_source="auto_sensor"`
- **AND** a CloudEvent `postpdf.start.triggered` includes `triggered_by="dagster-sensor"`

---

### Requirement: Docling Scope Limitation (Non-PDF Only)

The system SHALL restrict Docling usage to HTML, XML, and text parsing, explicitly prohibiting Docling from the PDF OCR path to enforce GPU-only policy.

**Rationale**: MinerU is the sole production PDF path; Docling lacks GPU-only semantics and would enable accidental CPU fallbacks.

#### Scenario: Docling validation guard rejects PDFs

- **GIVEN** a parser instance initialized with Docling
- **WHEN** the parser is called with `format="pdf"` and content bytes
- **THEN** the parser raises `ValueError("Docling cannot be used for PDF parsing in production. Use MinerU for PDF OCR (GPU-only policy).")`
- **AND** the error message includes remediation steps

#### Scenario: Docling successfully parses HTML guidelines

- **GIVEN** an HTML guideline document with complex markup
- **WHEN** the parser is called with `format="html"` and content bytes
- **THEN** Docling parses the HTML using `docling.partition.html`
- **AND** the result is mapped to IR Document with blocks and sections
- **AND** no validation error is raised

#### Scenario: Docling successfully parses XML

- **GIVEN** a JATS XML article from PMC
- **WHEN** the parser is called with `format="xml"` and content bytes
- **THEN** Docling parses the XML using `docling.partition.xml`
- **AND** the result is mapped to IR Document with IMRaD sections
- **AND** no validation error is raised

---

### Requirement: Unstructured Wrapper for XML/HTML Parsing

The system SHALL provide an `unstructured` library wrapper for XML and HTML parsing as the default non-PDF parser, replacing custom XML parsers.

**Rationale**: `unstructured` is a proven library with support for JATS XML, SPL XML, and HTML; eliminates custom parsing code.

#### Scenario: Unstructured parses JATS XML

- **GIVEN** a PMC article in JATS XML format
- **WHEN** the `UnstructuredParser` is called with the XML content
- **THEN** the parser uses `unstructured.partition_xml(text=content)`
- **AND** JATS elements (front, body, back) are mapped to IR sections
- **AND** figure captions and table XML are preserved in block metadata

#### Scenario: Unstructured parses SPL XML with LOINC sections

- **GIVEN** an SPL drug label with LOINC-coded sections
- **WHEN** the `UnstructuredParser` is called with the SPL XML content
- **THEN** the parser extracts LOINC codes from section elements
- **AND** each section becomes an IR Section with `section_label="LOINC:34089-3 Indications"`
- **AND** nested subsections (Dosage, Warnings) are preserved

#### Scenario: Unstructured parses HTML guidelines

- **GIVEN** an HTML guideline document with recommendation tables
- **WHEN** the `UnstructuredParser` is called with the HTML content
- **THEN** the parser extracts text content and table HTML
- **AND** heading hierarchy is preserved in IR sections
- **AND** table HTML is attached to blocks for downstream chunking

---

### Requirement: MinerU Output Format (Structured IR with Page/Bbox Maps)

The system SHALL require MinerU to produce structured IR with page numbers, bounding boxes, table HTML, and char offset maps, enabling span-grounded chunking and retrieval.

**Rationale**: Downstream chunking needs precise offsets; retrieval needs page/bbox for citation; table HTML preserves structure when rectangularization fails.

#### Scenario: MinerU produces IR with page/bbox for all blocks

- **GIVEN** a 10-page PDF processed by MinerU
- **WHEN** MinerU completes successfully
- **THEN** the IR Document includes 10+ blocks with `page_bbox` metadata
- **AND** each block has `{"page": N, "bbox": [x1, y1, x2, y2]}`
- **AND** char offsets map to the MinerU-produced Markdown/JSON

#### Scenario: MinerU preserves table HTML when confidence low

- **GIVEN** a PDF with a complex table (merged cells, nested headers)
- **WHEN** MinerU processes the table and calculates confidence=0.65
- **THEN** the IR Block includes `table_html` field with original HTML
- **AND** the block is tagged with `rectangularization_confidence=0.65`
- **AND** downstream chunking preserves the HTML (does not attempt rectangularization)

#### Scenario: MinerU char offset accuracy for span grounding

- **GIVEN** a PDF with text "Significant reduction in HbA1c (p<0.001)" on page 5
- **WHEN** MinerU produces IR
- **THEN** the corresponding block has `char_offsets=(14502, 14550)`
- **AND** extracting text[14502:14550] returns exactly "Significant reduction in HbA1c (p<0.001)"
- **AND** the block includes `page_bbox={"page": 5, "bbox": [120, 450, 480, 480]}`

---

## MODIFIED Requirements

### Requirement: PDF Parsing API (Modified)

The PDF parsing API SHALL enforce MinerU-only path with explicit GPU checks and fail-fast semantics.

**Previous Behavior**: PDF parsing had ambiguous paths (MinerU, Docling, or bespoke logic) with silent CPU fallbacks.

**New Behavior**: MinerU SHALL be the sole PDF path, GPU availability MUST be checked on startup, and failures SHALL halt the pipeline.

#### Scenario: GPU availability check on service startup

- **GIVEN** the MinerU service is starting up
- **WHEN** the service initializes
- **THEN** the service checks `torch.cuda.is_available()`
- **AND** if GPU is unavailable, raises `GpuNotAvailableError` and refuses to start
- **AND** no CPU fallback is attempted

#### Scenario: PDF parsing always routes to MinerU

- **GIVEN** a PDF ingestion job
- **WHEN** the orchestrator determines the document is a PDF
- **THEN** the job is routed to the MinerU service (GPU-bound)
- **AND** no other parser (Docling, bespoke) is considered
- **AND** the ledger records `parser_used="mineru"`

---

## REMOVED Requirements

### Requirement: Custom XML Parsers (Removed)

**Removed**: The requirement for custom XML parsers (`XMLParser`, `JATSParser`, `SPLParser`) is **REMOVED** in favor of the `unstructured` library wrapper.

**Reason**: Custom XML parsers duplicated proven library functionality, lacked community support, and created maintenance burden.

**Migration**: All custom XML parsing logic has been deleted. New XML ingestion uses `unstructured.partition_xml()` via the `UnstructuredParser` wrapper.

---

### Requirement: Bespoke PDF Parsing Logic (Removed)

**Removed**: The requirement for bespoke PDF parsing logic (`pdf_parser.py`) is **REMOVED** in favor of MinerU as the sole PDF path.

**Reason**: Bespoke PDF logic lacked OCR capability, had inconsistent quality, and violated GPU-only policy.

**Migration**: All bespoke PDF parsing has been deleted. All PDF ingestion routes to MinerU with explicit GPU checks and fail-fast semantics.
