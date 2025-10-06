from __future__ import annotations

from Medical_KG_rev.gateway.sse.manager import EventStreamManager
from Medical_KG_rev.orchestration import IngestWorker, KafkaClient, MappingWorker, Orchestrator
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.orchestrator import (
    DEAD_LETTER_TOPIC,
    INGEST_REQUESTS_TOPIC,
    INGEST_RESULTS_TOPIC,
    MAPPING_EVENTS_TOPIC,
)


def setup_components():
    kafka = KafkaClient()
    kafka.create_topics(
        [INGEST_REQUESTS_TOPIC, INGEST_RESULTS_TOPIC, MAPPING_EVENTS_TOPIC, DEAD_LETTER_TOPIC]
    )
    ledger = JobLedger()
    events = EventStreamManager()
    orchestrator = Orchestrator(kafka, ledger, events)
    ingest_worker = IngestWorker(orchestrator, kafka, ledger, events)
    mapping_worker = MappingWorker(kafka, ledger, events)
    return orchestrator, ingest_worker, mapping_worker


def test_ingest_worker_processes_pipeline() -> None:
    orchestrator, ingest_worker, mapping_worker = setup_components()
    entry = orchestrator.submit_job(
        tenant_id="tenant",
        dataset="papers",
        item={"id": "worker-test"},
    )
    ingest_worker.run_once()
    mapping_worker.run_once()
    list(orchestrator.kafka.consume(INGEST_RESULTS_TOPIC))

    status = orchestrator.ledger.get(entry.job_id)
    assert status is not None
    assert status.status == "completed"
    assert orchestrator.kafka.pending(INGEST_RESULTS_TOPIC) == 0


def test_mapping_worker_updates_metadata() -> None:
    orchestrator, ingest_worker, mapping_worker = setup_components()
    entry = orchestrator.submit_job(
        tenant_id="tenant",
        dataset="papers",
        item={"id": "map-test", "document_type": "pdf"},
    )

    message = next(orchestrator.kafka.consume(INGEST_REQUESTS_TOPIC))
    orchestrator.execute_pipeline(entry.job_id, message.value)

    mapping_worker.run_once()
    status = orchestrator.ledger.get(entry.job_id)
    assert status is not None
    assert status.metadata.get("mapping") is True
