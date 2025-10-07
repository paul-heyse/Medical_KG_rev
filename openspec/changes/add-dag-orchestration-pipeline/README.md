# Add DAG-Based Orchestration Pipeline

## Overview

This change refactors the orchestration system from hardcoded Python pipelines to **declarative, DAG-based execution** using Dagster, Haystack 2, and portable resilience/observability libraries.

**Key Outcome**: Pipeline topology defined in YAML, stages as typed contracts, GPU fail-fast preserved, PDF two-phase gate explicit, and full lineage tracking via CloudEvents + OpenLineage.

## Problem

Current orchestration hardwires stage sequences in Python code (`Orchestrator._register_default_pipelines()`), making it difficult to:

1. Visualize and modify pipeline flow without code changes
2. Implement complex gates (PDF two-phase: halt after download, resume after MinerU)
3. Swap underlying implementations (chunkers, embedders) without cascading changes
4. Configure resilience (retries, backoff, circuit breakers) per-stage
5. Track provenance and lineage in a portable format

## Solution

### 1. **Declarative Pipeline Topology**

- **Files**: `config/orchestration/pipelines/auto.yaml`, `pdf-two-phase.yaml`
- **Content**: Stages in order with dependencies, gate conditions, resilience policy refs
- **Benefit**: Engineers edit YAML to add/reorder stages, no Python deployment needed

### 2. **Typed Stage Contracts** (Python Protocols)

- **Ports**: `IngestStage`, `ParseStage`, `ChunkStage`, `EmbedStage`, `IndexStage`, `ExtractStage`, `KGStage`
- **Adapters**: Haystack wrappers, custom implementations, legacy service wrappers
- **Benefit**: Swap chunkers/embedders without touching orchestration, static type safety

### 3. **Dagster Workflow Engine**

- **Library**: Dagster 1.5.0+ (<https://dagster.io/>)
- **Local-first**: Runs as simple process, no scheduler required, UI at `localhost:3000`
- **Sensors**: Poll Job Ledger for `pdf_ir_ready=true`, trigger post-PDF stages
- **Benefit**: Visualize runs, retry failed stages, debug with logs/lineage

### 4. **Haystack 2 for Text Operations**

- **Library**: haystack-ai 2.0+ (<https://haystack.deepset.ai/>)
- **Components**: `DocumentSplitter` (chunk), `OpenAIDocumentEmbedder` (embed via vLLM), `OpenSearchDocumentWriter` (index)
- **Benefit**: Vendor-agnostic, composable, preserves hybrid retrieval (BM25 + SPLADE + dense)

### 5. **Resilience as Configuration**

- **Libraries**:
  - **tenacity 8.2.0+** (<https://tenacity.readthedocs.io/>) - Retries, backoff
  - **pybreaker 1.0.0+** (<https://github.com/danielfm/pybreaker>) - Circuit breakers
  - **aiolimiter 1.1.0+** (<https://github.com/mjpieters/aiolimiter>) - Rate limiting

- **Policies**: `default`, `gpu-bound`, `polite-api` defined in `config/orchestration/resilience.yaml`
- **Benefit**: Tune retries/backoff per stage without code changes

### 6. **CloudEvents + OpenLineage Observability**

- **CloudEvents 1.0** (<https://cloudevents.io/>) - Portable event format for stage lifecycle
- **OpenLineage** (<https://openlineage.io/>) - Optional lineage for Marquez visualization
- **Topic**: `orchestration.events.v1` on Kafka
- **Benefit**: Complete audit trail, integrates with existing Prometheus/Grafana

### 7. **AsyncAPI Documentation**

- **Spec**: `docs/asyncapi.yaml` describes Kafka topics (orchestration.events, mineru.queue)
- **Validation**: CI/CD runs `asyncapi validate` to catch schema errors
- **Benefit**: Clear queue contracts for future contributors

### 8. **Testing with respx**

- **Library**: respx 0.20.0+ (<https://lundberg.github.io/respx/>) - Mock HTTP calls
- **Usage**: Validate resilience (retries, circuit breakers), adapter calls, model endpoints
- **Benefit**: Comprehensive stage-level tests without live external dependencies

## PDF Two-Phase Gate in Practice

**Non-PDF Pipeline** (auto): Runs end-to-end automatically: ingest → parse → chunk → embed → index → extract → kg

**PDF Pipeline** (pdf-two-phase):

1. **Pre-PDF**: ingest → download
2. **Gate**: Job Ledger updated with `pdf_downloaded=true`, pipeline pauses
3. **MinerU**: Separate GPU worker processes PDF, sets `pdf_ir_ready=true`
4. **Resume**: `pdf_ir_ready_sensor` detects condition, starts post-PDF: chunk → embed → index → extract → kg

All failures (MinerU, embed, extract) set job to failed state immediately with no downstream work (fail-fast preserved).

## Key Components

| Component | Purpose | Library |
|-----------|---------|---------|
| **Dagster** | DAG execution, sensors, UI | dagster 1.5.0+ |
| **Haystack 2** | Text ops (chunk, embed, index) | haystack-ai 2.0+ |
| **tenacity** | Retry decorators | tenacity 8.2.0+ |
| **pybreaker** | Circuit breakers | pybreaker 1.0.0+ |
| **aiolimiter** | Rate limiting | aiolimiter 1.1.0+ |
| **CloudEvents** | Event envelope | cloudevents 1.9.0+ |
| **OpenLineage** | Lineage records (optional) | openlineage-python 1.0+ |
| **respx** | HTTP mocking for tests | respx 0.20.0+ |

## Example: Auto Pipeline YAML

```yaml
# config/orchestration/pipelines/auto.yaml
version: "1.0"
name: "auto"
description: "Non-PDF sources: ClinicalTrials, OpenAlex, OpenFDA"
applicable_sources: [clinicaltrials, openalex, openfda-drug-label]

stages:
  - name: ingest
    type: IngestStage
    policy: default

  - name: parse
    type: ParseStage
    policy: default
    depends_on: [ingest]

  - name: chunk
    type: ChunkStage
    policy: default
    depends_on: [parse]

  - name: embed
    type: EmbedStage
    policy: gpu-bound
    depends_on: [chunk]

  - name: index
    type: IndexStage
    policy: default
    depends_on: [embed]

  - name: extract
    type: ExtractStage
    policy: gpu-bound
    depends_on: [index]

  - name: kg
    type: KGStage
    policy: default
    depends_on: [extract]
```

## Example: PDF Two-Phase YAML with Gate

```yaml
# config/orchestration/pipelines/pdf-two-phase.yaml
version: "1.0"
name: "pdf-two-phase"
description: "PDF sources with MinerU gate"
applicable_sources: [pmc-fulltext, unpaywall, core]

stages:
  - name: ingest
    type: IngestStage
    policy: default

  - name: download_pdf
    type: DownloadStage
    policy: polite-api
    depends_on: [ingest]

  # GATE: Wait for MinerU to set pdf_ir_ready=true
  - name: pdf_gate
    type: Gate
    condition:
      ledger_field: pdf_ir_ready
      equals: true
    resume_stage: chunk

  - name: chunk
    type: ChunkStage
    policy: gpu-bound
    depends_on: [pdf_gate]

  - name: embed
    type: EmbedStage
    policy: gpu-bound
    depends_on: [chunk]

  - name: index
    type: IndexStage
    policy: default
    depends_on: [embed]

  - name: extract
    type: ExtractStage
    policy: gpu-bound
    depends_on: [index]

  - name: kg
    type: KGStage
    policy: default
    depends_on: [extract]
```

## Example: Resilience Policy

```yaml
# config/orchestration/resilience.yaml
policies:
  default:
    max_attempts: 3
    backoff_strategy: exponential
    backoff_initial_seconds: 1.0
    backoff_max_seconds: 10.0
    backoff_jitter_seconds: 1.0
    timeout_seconds: 30

  gpu-bound:
    max_attempts: 1  # Fail-fast for GPU
    backoff_strategy: none
    timeout_seconds: 60
    circuit_breaker:
      failure_threshold: 5
      reset_timeout_seconds: 60

  polite-api:
    max_attempts: 10
    backoff_strategy: linear
    backoff_initial_seconds: 1.0
    backoff_max_seconds: 10.0
    timeout_seconds: 10
    rate_limit_per_second: 5.0
```

## Example: Stage Contract Implementation

```python
# src/Medical_KG_rev/orchestration/stages/contracts.py
from typing import Protocol
from dataclasses import dataclass

@dataclass
class StageContext:
    tenant_id: str
    doc_id: str
    correlation_id: str
    metadata: dict[str, object]

class ChunkStage(Protocol):
    def execute(self, ctx: StageContext, document: Document) -> list[Chunk]:
        ...

# Implementation with Haystack
class HaystackChunker(ChunkStage):
    def __init__(self):
        self.splitter = DocumentSplitter(split_by="sentence", split_length=5)

    def execute(self, ctx: StageContext, document: Document) -> list[Chunk]:
        # Convert IR → Haystack
        haystack_docs = [Document(content=s.text) for s in document.sections]

        # Split
        result = self.splitter.run(documents=haystack_docs)

        # Convert back to IR
        chunks = [
            Chunk(chunk_id=f"{ctx.doc_id}:chunk:{i}", text=d.content)
            for i, d in enumerate(result["documents"])
        ]
        return chunks
```

## Migration Path

| Phase | Duration | Actions |
|-------|----------|---------|
| **Phase 1** | Week 1-2 | Add Dagster, implement stage contracts, create YAML topologies, deploy to dev with `MK_USE_DAGSTER=false` |
| **Phase 2** | Week 3-4 | Enable for non-PDF sources (ClinicalTrials, OpenAlex), validate outputs match legacy, performance comparison |
| **Phase 3** | Week 5-6 | Implement PDF sensor, enable for PMC/Unpaywall, test gate halt/resume, monitor for stalled jobs |
| **Phase 4** | Week 7-8 | Enable for 100% production traffic, deprecate legacy orchestration, schedule removal in v0.3.0 |

## Benefits

| Benefit | Description |
|---------|-------------|
| **Visibility** | Dagster UI shows real-time pipeline execution, stage timings, failure points |
| **Flexibility** | Add/remove/reorder stages by editing YAML, no code deployment |
| **Reliability** | Centralized resilience policies, circuit breakers prevent cascading failures |
| **Debuggability** | CloudEvents + OpenLineage provide complete audit trail |
| **Testability** | respx mocks enable comprehensive stage-level testing |
| **Extensibility** | Haystack components can be swapped (LlamaIndex, HuggingFace) without touching orchestration |

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| **Learning curve** | Comprehensive docs, video walkthroughs, pair programming |
| **Haystack 2 maturity** | Stage contracts insulate rest of system, can swap components |
| **Migration complexity** | Feature flag for gradual rollout, comprehensive integration tests |
| **Operational overhead** | Local-first design, health checks, runbook for Dagster ops |

## Files Changed

**New Files**:

- `config/orchestration/pipelines/auto.yaml`
- `config/orchestration/pipelines/pdf-two-phase.yaml`
- `config/orchestration/resilience.yaml`
- `src/Medical_KG_rev/orchestration/stages/contracts.py`
- `src/Medical_KG_rev/orchestration/dagster/jobs.py`
- `src/Medical_KG_rev/orchestration/dagster/sensors.py`
- `src/Medical_KG_rev/services/haystack/chunker.py`
- `src/Medical_KG_rev/services/haystack/embedder.py`
- `src/Medical_KG_rev/services/haystack/index_writer.py`
- `src/Medical_KG_rev/orchestration/observability/cloudevents.py`
- `docs/asyncapi.yaml` (updated with orchestration topics)

**Modified Files**:

- `src/Medical_KG_rev/orchestration/orchestrator.py` (add Dagster routing)
- `src/Medical_KG_rev/orchestration/ledger.py` (new fields: `pdf_downloaded`, `pdf_ir_ready`, `current_stage`)
- `requirements.txt` (add dagster, haystack-ai, tenacity, pybreaker, aiolimiter, cloudevents, openlineage-python)
- `requirements-dev.txt` (add respx)
- `ops/docker-compose.yml` (add dagster-webserver, dagster-daemon)
- `ops/k8s/base/` (add Dagster deployments)

## Next Steps

1. Review proposal with team (architecture, backend, ML engineers)
2. Approve and prioritize for sprint planning
3. Implement Phase 1: Foundation & Dagster setup (Week 1-2)
4. Validate with integration tests (auto pipeline, PDF two-phase)
5. Gradual rollout to production (Phases 2-4)
6. Deprecate legacy orchestration in v0.3.0

## Related Documents

- **Proposal**: `proposal.md` - Why, What, Impact
- **Tasks**: `tasks.md` - 176 implementation tasks across 16 work streams
- **Design**: `design.md` - Technical decisions, alternatives, risks
- **Spec Deltas**:
  - `specs/orchestration/spec.md` - DAG execution, resilience, CloudEvents
  - `specs/ingestion/spec.md` - Adapter stages, PDF download
  - `specs/retrieval/spec.md` - Haystack chunking, embedding, indexing
  - `specs/observability/spec.md` - CloudEvents, OpenLineage, Prometheus metrics
