# Legacy Orchestration Dependencies

## Import References
- `src/Medical_KG_rev/orchestration/__init__.py`
  - Imports `OrchestrationError`, `Orchestrator`, `IngestWorker`, `MappingWorker`
- `src/Medical_KG_rev/orchestration/worker.py`
  - Imports `Orchestrator` for worker coordination
- `src/Medical_KG_rev/gateway/services.py`
  - Imports `Orchestrator`, `IngestWorker`, and `MappingWorker` for REST orchestration

## Call Sites
- `src/Medical_KG_rev/orchestration/worker.py`
  - Calls `Orchestrator.execute_pipeline` when processing Kafka jobs
- `src/Medical_KG_rev/gateway/services.py`
  - Instantiates `Orchestrator` for request submission
- `src/Medical_KG_rev/orchestration/query_builder.py`
  - Constructs retrieval stage orchestrators (`RetrievalOrchestrator`, `FusionOrchestrator`, `RerankOrchestrator`, `FinalSelectorOrchestrator`)

## Legacy Pipeline Classes
- `src/Medical_KG_rev/orchestration/retrieval_pipeline.py`
  - Defines retrieval orchestrators that must be replaced by Dagster stage execution

## Replacement Plan
- Introduce `DagsterOrchestrator` entrypoint that submits jobs to Dagster definitions
- Replace worker-based pipeline execution with Dagster daemon triggers
- Replace retrieval orchestrators with Haystack retriever wrappers
- Update gateway service to call `DagsterOrchestrator.submit` instead of creating legacy workers
