# Change Proposal: Ingestion & Orchestration System

## Why

Build the orchestration engine that coordinates the multi-step ingestion pipeline using Apache Kafka for async messaging, implements two-phase processing for PDFs (metadata → GPU parsing), manages job state via ledger, and handles multi-source enrichment workflows (OpenAlex → Unpaywall → PDF → MinerU).

## What Changes

- Apache Kafka setup with topics (ingest.requests.v1, ingest.results.v1, mapping.events.v1)
- Job orchestrator service with state machine (queued → processing → completed/failed)
- Ledger system for tracking job state and document processing stages
- Two-phase pipeline: auto-pipeline (fast) vs manual pipeline (GPU-bound)
- Multi-adapter chaining for literature enrichment
- Idempotency keys and deduplication
- Job queue priority and retry mechanisms
- Worker processes for consuming Kafka messages
- Dead letter queue for failed jobs
- Job status API and SSE streaming

## Impact

- **Affected specs**: NEW capability `ingestion-orchestration`
- **Affected code**:
  - `src/Medical_KG_rev/orchestration/` - Orchestrator service
  - `src/Medical_KG_rev/orchestration/ledger.py` - State tracking
  - `src/Medical_KG_rev/orchestration/kafka_client.py` - Kafka producer/consumer
  - `src/Medical_KG_rev/orchestration/pipeline.py` - Pipeline definitions
  - `src/Medical_KG_rev/orchestration/workers.py` - Background workers
  - `docker-compose.yml` - Add Kafka, Zookeeper services
  - `tests/orchestration/` - Orchestration integration tests
