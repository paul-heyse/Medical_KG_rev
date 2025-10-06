from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from Medical_KG_rev.gateway.app import create_app


def test_wsdl_available() -> None:
    client = TestClient(create_app())
    response = client.get("/soap/wsdl")
    assert response.status_code == 200
    assert "GatewayService" in response.text


def test_ingest_operation() -> None:
    client = TestClient(create_app())
    envelope = """
    <Envelope>
      <Body>
        <Ingest dataset=\"clinicaltrials\" tenantId=\"tenant\">
          <item><id>doc-1</id></item>
        </Ingest>
      </Body>
    </Envelope>
    """
    response = client.post("/soap", data=envelope, headers={"Content-Type": "text/xml"})
    assert response.status_code == 200
    assert "IngestResponse" in response.text
