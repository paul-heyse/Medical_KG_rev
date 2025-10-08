## Why

The current system lacks actual PDF acquisition and MinerU processing capabilities. While the OpenAlex adapter provides metadata, it doesn't extract or use PDF URLs for downloading, and no stage actually triggers MinerU processing or updates the ledger with `pdf_ir_ready` status. This prevents the PDF two-phase pipeline from functioning as intended.

## What Changes

- **⬇️ PDF Download Implementation**: Extract PDF URLs from document metadata and download files
- **🤖 MinerU Integration**: Trigger MinerU processing on downloaded PDFs and update ledger state
- **📊 Progress Tracking**: Monitor download and processing progress with proper state management
- **🔄 Error Recovery**: Handle download failures and MinerU processing errors gracefully

## Impact

- **Affected specs**: `specs/orchestration/spec.md`, `specs/adapters/spec.md` (PDF processing capabilities)
- **Affected code**:
  - `src/Medical_KG_rev/adapters/biomedical.py` - Extract PDF URLs from OpenAlex metadata
  - `src/Medical_KG_rev/orchestration/dagster/stages.py` - PDF download and MinerU stages
  - `src/Medical_KG_rev/orchestration/ledger.py` - PDF state tracking
  - `src/Medical_KG_rev/services/mineru/` - MinerU service integration
- **Breaking changes**: None - adds missing functionality to incomplete PDF pipeline
- **Migration path**: Existing metadata-only processing continues while PDF processing is added

## Success Criteria

- ✅ PDF URLs are extracted from document metadata and downloaded
- ✅ Downloaded PDFs are processed through MinerU service
- ✅ Ledger is updated with `pdf_downloaded` and `pdf_ir_ready` states
- ✅ Download and processing errors are handled appropriately
- ✅ Progress tracking and metrics are implemented
