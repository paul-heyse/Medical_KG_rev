"""Example adapter plugin used for documentation and testing."""

from __future__ import annotations

from Medical_KG_rev.adapters.plugins.base import ReadOnlyAdapterPlugin
from Medical_KG_rev.adapters.plugins.models import (
    AdapterDomain,
    AdapterMetadata,
    AdapterRequest,
    AdapterResponse,
    BiomedicalPayload,
    AdapterResponseEnvelope,
)


class ExampleAdapterPlugin(ReadOnlyAdapterPlugin):
    """Trivial adapter returning static payload for testing the framework."""

    metadata = AdapterMetadata(
        name="example",
        version="1.0.0",
        domain=AdapterDomain.BIOMEDICAL,
        summary="Example adapter demonstrating Pluggy integration",
        capabilities=["search", "metadata"],
        maintainer="Medical KG Team",
    )

    async_mode: bool = False

    def fetch(self, request: AdapterRequest) -> AdapterResponse:
        envelope = AdapterResponseEnvelope(
            items=[{"message": "hello world", "tenant": request.tenant_id}],
            metadata={"correlation_id": request.correlation_id},
        )
        envelope.attach_payload(
            AdapterDomain.BIOMEDICAL,
            BiomedicalPayload(mesh_terms=["pluggy", "example"], trial_phase="N/A"),
        )
        return envelope
