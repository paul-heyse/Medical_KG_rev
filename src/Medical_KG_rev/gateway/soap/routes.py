"""Minimal SOAP adapter bridging to the gateway service."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from fastapi import APIRouter, Body, Depends, HTTPException, Response, status
import textwrap

from ...auth import Scopes, SecurityContext
from ...auth.audit import get_audit_trail
from ...auth.dependencies import get_security_context
from ...auth.rate_limit import RateLimitExceeded, build_rate_limiter
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


def _parse_items(element: ET.Element) -> list[dict]:
    items: list[dict] = []
    for node in element.findall("item"):
        item_payload = {child.tag: child.text or "" for child in node}
        items.append(item_payload)
    return items


@router.post("", include_in_schema=False)
async def soap_entrypoint(
    body: str = Body(..., media_type="text/xml"),
    security: SecurityContext = Depends(get_security_context),
    service: GatewayService = Depends(get_gateway_service),
) -> Response:
    rate_limiter = build_rate_limiter()
    try:
        envelope = ET.fromstring(body)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    action_node = envelope.find(".//{*}Ingest")
    if action_node is not None:
        try:
            rate_limiter.check(security.identity, "SOAP:Ingest")
        except RateLimitExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(int(exc.retry_after))},
            )
        if not security.has_scope(Scopes.INGEST_WRITE):
            raise HTTPException(status_code=403, detail="Missing required scopes: ingest:write")
        dataset = action_node.get("dataset", "clinicaltrials")
        tenant_id = action_node.get("tenantId", "tenant-default")
        if tenant_id != security.tenant_id:
            raise HTTPException(status_code=403, detail="Tenant mismatch")
        items = _parse_items(action_node)
        request = IngestionRequest(tenant_id=tenant_id, items=items)
        result = service.ingest(dataset, request)
        payload = "".join(
            f"<operation id=\"{status.job_id}\" message=\"{status.message or ''}\"/>"
            for status in result.operations
        )
        get_audit_trail().record(
            context=security,
            action="soap_ingest",
            resource=f"dataset:{dataset}",
            metadata={"items": len(items)},
        )
        return Response(
            content=f'<Envelope><Body><IngestResponse total="{result.total}">{payload}</IngestResponse></Body></Envelope>',
            media_type="application/xml",
        )

    retrieve_node = envelope.find(".//{*}Retrieve")
    if retrieve_node is not None:
        try:
            rate_limiter.check(security.identity, "SOAP:Retrieve")
        except RateLimitExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(int(exc.retry_after))},
            )
        if not security.has_scope(Scopes.RETRIEVE_READ):
            raise HTTPException(status_code=403, detail="Missing required scopes: kg:read")
        tenant_id = retrieve_node.get("tenantId", "tenant-default")
        if tenant_id != security.tenant_id:
            raise HTTPException(status_code=403, detail="Tenant mismatch")
        query = retrieve_node.get("query", "")
        request = RetrieveRequest(tenant_id=tenant_id, query=query)
        result = service.retrieve(request)
        documents = "".join(
            f'<document id="{doc.id}" title="{doc.title}" score="{doc.score}"/>'
            for doc in result.documents
        )
        get_audit_trail().record(
            context=security,
            action="soap_retrieve",
            resource="retrieve",
            metadata={"total": result.total},
        )
        return Response(
            content=f'<Envelope><Body><RetrieveResponse total="{result.total}">{documents}</RetrieveResponse></Body></Envelope>',
            media_type="application/xml",
        )

    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported SOAP action")
