## Why

The `pdf-two-phase` pipeline topology is incomplete and lacks proper stage configuration. The ingest stage has no adapter binding, gates are defined but ignored by the execution engine, and the pipeline doesn't properly handle PDF acquisition and MinerU processing. This prevents the PDF pipeline from functioning as intended for document processing workflows.

## What Changes

- **📋 Complete Pipeline Configuration**: Add proper adapter binding and stage configuration for PDF pipeline
- **🚪 Gate Stage Integration**: Implement gate handling in pipeline topology and execution
- **⬇️ Download Stage Definition**: Define download stage for PDF acquisition
- **🔗 Stage Dependencies**: Properly configure stage dependencies and execution order

## Impact

- **Affected specs**: `specs/orchestration/spec.md` (pipeline topology capabilities)
- **Affected code**:
  - `config/orchestration/pipelines/pdf-two-phase.yaml` - Complete pipeline configuration
  - `src/Medical_KG_rev/orchestration/dagster/runtime.py` - Update topology loading
  - `src/Medical_KG_rev/orchestration/dagster/stages.py` - Add missing stage builders
- **Breaking changes**: None - enhances existing incomplete pipeline
- **Migration path**: Existing partial PDF pipeline will be completed with backward compatibility

## Success Criteria

- ✅ PDF pipeline includes all required stages with proper configuration
- ✅ Gate definitions are properly integrated into pipeline execution
- ✅ Download stage is defined and can be executed
- ✅ Stage dependencies correctly reflect two-phase execution model
- ✅ Pipeline validation catches configuration errors
