"""Minimal SOAP adapter bridging to the gateway service."""

from __future__ import annotations

import textwrap
import xml.etree.ElementTree as ET
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response, status

from ..models import IngestionRequest, RetrieveRequest
from ..services import GatewayService, get_gateway_service

router = APIRouter(prefix="/soap", tags=["soap"])

WSDL_TEMPLATE = textwrap.dedent(
    """
    <?xml version="1.0" encoding="UTF-8"?>
    <definitions xmlns="http://schemas.xmlsoap.org/wsdl/"
                 xmlns:tns="http://medical-kg/gateway"
                 xmlns:soap="http://schemas.xmlsoap.org/wsdl/soap/"
                 xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                 targetNamespace="http://medical-kg/gateway">
      <types/>
      <message name="IngestRequest"/>
      <message name="RetrieveRequest"/>
      <portType name="GatewayPortType">
        <operation name="Ingest"/>
        <operation name="Retrieve"/>
      </portType>
      <binding name="GatewayBinding" type="tns:GatewayPortType">
        <soap:binding style="document" transport="http://schemas.xmlsoap.org/soap/http"/>
      </binding>
      <service name="GatewayService">
        <documentation>Minimal SOAP wrapper for the Medical KG gateway.</documentation>
        <port name="GatewayPort" binding="tns:GatewayBinding">
          <soap:address location="http://localhost:8000/soap"/>
        </port>
      </service>
    </definitions>
    """
).strip()


@router.get("/wsdl", include_in_schema=False)
async def serve_wsdl() -> Response:
    return Response(content=WSDL_TEMPLATE, media_type="application/xml")


def _parse_items(element: ET.Element) -> List[dict]:
    items: List[dict] = []
    for node in element.findall("item"):
        item_payload = {child.tag: child.text or "" for child in node}
        items.append(item_payload)
    return items


@router.post("", include_in_schema=False)
async def soap_entrypoint(
    body: str,
    service: GatewayService = Depends(get_gateway_service),
) -> Response:
    try:
        envelope = ET.fromstring(body)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    action_node = envelope.find(".//{*}Ingest")
    if action_node is not None:
        dataset = action_node.get("dataset", "clinicaltrials")
        tenant_id = action_node.get("tenantId", "tenant-default")
        items = _parse_items(action_node)
        request = IngestionRequest(tenant_id=tenant_id, items=items)
        result = service.ingest(dataset, request)
        payload = "".join(
            f"<operation id=\"{status.job_id}\" message=\"{status.message or ''}\"/>"
            for status in result.operations
        )
        return Response(
            content=f"<Envelope><Body><IngestResponse total=\"{result.total}\">{payload}</IngestResponse></Body></Envelope>",
            media_type="application/xml",
        )

    retrieve_node = envelope.find(".//{*}Retrieve")
    if retrieve_node is not None:
        tenant_id = retrieve_node.get("tenantId", "tenant-default")
        query = retrieve_node.get("query", "")
        request = RetrieveRequest(tenant_id=tenant_id, query=query)
        result = service.retrieve(request)
        documents = "".join(
            f"<document id=\"{doc.id}\" title=\"{doc.title}\" score=\"{doc.score}\"/>"
            for doc in result.documents
        )
        return Response(
            content=f"<Envelope><Body><RetrieveResponse total=\"{result.total}\">{documents}</RetrieveResponse></Body></Envelope>",
            media_type="application/xml",
        )

    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported SOAP action")
