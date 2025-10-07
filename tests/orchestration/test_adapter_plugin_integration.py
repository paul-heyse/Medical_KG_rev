import pytest

from Medical_KG_rev.adapters import (
    AdapterDomain,
    AdapterMetadata,
    AdapterPluginManager,
    AdapterResponse,
    ValidationOutcome,
)
from Medical_KG_rev.adapters.plugins.base import BaseAdapterPlugin
from Medical_KG_rev.orchestration.kafka import KafkaClient
from Medical_KG_rev.orchestration.ledger import JobLedger
from Medical_KG_rev.orchestration.orchestrator import Orchestrator
from Medical_KG_rev.gateway.sse.manager import EventStreamManager


class _StubPlugin(BaseAdapterPlugin):
    metadata = AdapterMetadata(
        name="dummy",
        version="1.0.0",
        domain=AdapterDomain.BIOMEDICAL,
        summary="Stub adapter",
    )

    def fetch(self, request):
        return AdapterResponse(items=[["doc"]])

    def parse(self, response, request):
        response.items = response.items[0]
        return response

    def validate(self, response, request):
        return ValidationOutcome(valid=True)


@pytest.fixture
def stub_manager(monkeypatch):
    manager = AdapterPluginManager()
    manager.register(_StubPlugin())
    monkeypatch.setattr(
        "Medical_KG_rev.orchestration.orchestrator.get_plugin_manager",
        lambda: manager,
    )
    return manager


def test_orchestrator_adapter_chain_executes_plugins(stub_manager):
    kafka = KafkaClient()
    ledger = JobLedger()
    events = EventStreamManager()
    orchestrator = Orchestrator(kafka, ledger, events)
    entry = ledger.create(
        job_id="job-1",
        doc_key="doc-1",
        tenant_id="tenant",
        metadata={"domains": [AdapterDomain.BIOMEDICAL.value]},
    )
    context = orchestrator._handle_adapter_chain(entry, {})
    assert context["adapter_chain"] == ["dummy"]
    adapter_response = context["adapter_responses"][0]
    assert adapter_response["items"] == ["doc"]
    assert adapter_response["metadata"]["adapter"] == "dummy"
    assert adapter_response["telemetry"]["stages"][0]["status"] == "success"
    assert adapter_response["telemetry"]["pipeline"] == "default"
    stored = ledger.get("job-1")
    assert stored is not None
    assert stored.metadata.get("adapter_versions", {}).get("dummy") == "1.0.0"
