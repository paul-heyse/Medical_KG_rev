# Legacy Orchestration Decommission Checklist

## Files Scheduled for Removal

### Orchestration Service
- ~~`src/Medical_KG_rev/orchestration/orchestrator.py`~~ (removed)
- ~~`src/Medical_KG_rev/orchestration/worker.py`~~ (removed)
- ~~src/Medical_KG_rev/orchestration/pipeline.py~~
- ~~src/Medical_KG_rev/orchestration/profiles.py~~
- ~~src/Medical_KG_rev/orchestration/ingestion_pipeline.py~~
- ~~src/Medical_KG_rev/orchestration/config_manager.py~~
- `src/Medical_KG_rev/orchestration/kafka.py`
- ~~src/Medical_KG_rev/orchestration/query_builder.py~~
- ~~src/Medical_KG_rev/orchestration/retrieval_pipeline.py~~

### Services Layer Adapters
- `src/Medical_KG_rev/services/retrieval/indexing_service.py`
- `src/Medical_KG_rev/services/embedding/service.py`
- `src/Medical_KG_rev/services/retrieval/chunking.py`

### Legacy Tests
- ~~`tests/orchestration/test_orchestrator.py`~~ (removed)
- ~~`tests/orchestration/test_workers.py`~~ (removed)
- ~~`tests/orchestration/test_integration.py`~~ (removed)
- ~~`tests/orchestration/test_adapter_plugin_integration.py`~~ (removed)
- ~~tests/orchestration/test_retrieval_pipeline.py~~

## Imports to Update
- Replace `from Medical_KG_rev.orchestration import Orchestrator` with `from Medical_KG_rev.orchestration.dagster.runtime import DagsterOrchestrator`
- Replace `from Medical_KG_rev.orchestration.worker import IngestWorker, MappingWorker` with Dagster job submission helpers
- Replace `from Medical_KG_rev.services.retrieval.chunking import ...` with `from Medical_KG_rev.orchestration.haystack.components import HaystackChunker`
- Replace `from Medical_KG_rev.services.embedding.service import ...` with `from Medical_KG_rev.orchestration.haystack.components import HaystackEmbedder`
- Replace `from Medical_KG_rev.services.retrieval.indexing_service import ...` with `HaystackIndexWriter`

## Documentation to Update
- `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`
- `docs/guides/orchestration-pipelines.md`
- `README.md`
- `docs/asyncapi.yaml`
- `openspec/changes/add-dag-orchestration-pipeline/HARD_CUTOVER_STRATEGY.md` (confirm completed steps)

## Follow-Up Actions
- Update gateway service wiring to instantiate `DagsterOrchestrator`
- Remove legacy pipeline registration logic and profile selection
- Ensure CI lint/test jobs reference new Dagster entrypoints
- Regenerate API and AsyncAPI docs after removal
