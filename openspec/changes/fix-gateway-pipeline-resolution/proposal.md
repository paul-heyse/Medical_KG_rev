## Why

The gateway currently lacks proper pipeline resolution logic for ingestion requests. When clients submit ingestion batches, the gateway should intelligently resolve the appropriate pipeline topology based on dataset metadata and document characteristics, but instead relies on hardcoded or missing logic that doesn't properly handle PDF documents and fallback scenarios.

## What Changes

- **🔧 Intelligent Pipeline Resolution**: Implement smart pipeline selection based on dataset metadata and document properties
- **📋 PDF Detection Logic**: Add detection for `document_type="pdf"` to trigger `pdf-two-phase` pipeline
- **🔗 Orchestrator Integration**: Properly connect gateway pipeline resolution to Dagster orchestrator execution
- **⚙️ Configuration-Driven Logic**: Move pipeline selection logic from hardcoded values to configurable rules

## Impact

- **Affected specs**: `specs/gateway/spec.md` (pipeline resolution capabilities)
- **Affected code**:
  - `src/Medical_KG_rev/gateway/services.py` - Add pipeline resolution logic
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py` - Update job submission interface
  - `config/gateway/pipeline-resolution.yaml` - New configuration for resolution rules
- **Breaking changes**: None - enhances existing ingestion flow without changing API contracts
- **Migration path**: Existing hardcoded pipeline selection will be replaced by configuration-driven logic

## Success Criteria

- ✅ Gateway correctly resolves `pdf-two-phase` pipeline for PDF documents
- ✅ Fallback pipeline selection works for non-PDF documents
- ✅ Pipeline resolution is configurable via YAML rules
- ✅ Integration with Dagster orchestrator maintains existing API compatibility
- ✅ Error handling provides clear feedback for invalid pipeline configurations
