"""gRPC server stubs wired to the gateway service layer."""

from __future__ import annotations

import asyncio
import importlib.util
from datetime import datetime, timezone

import grpc

_grpc_health_spec = None
try:
    _grpc_health_spec = importlib.util.find_spec("grpc_health")
except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
    _grpc_health_spec = None

if _grpc_health_spec is not None:
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc
else:  # pragma: no cover - fallback when grpc-health not installed

    class _StubHealth:
        class HealthServicer:  # type: ignore[too-many-ancestors]
            def __init__(self) -> None:
                self._statuses = {}

            async def Watch(self, request, context):  # pragma: no cover - stub
                return None

            def set(self, service: str, status: str) -> None:
                self._statuses[service] = status

    class _StubHealthPb2:
        class HealthCheckResponse:  # pragma: no cover - stub
            SERVING = "SERVING"

    class _StubHealthGrpc:
        @staticmethod
        def add_HealthServicer_to_server(*args, **kwargs):  # pragma: no cover - stub
            return None

    health = _StubHealth()
    health_pb2 = _StubHealthPb2()
    health_pb2_grpc = _StubHealthGrpc()

from ..models import (
    ChunkRequest,
    EmbedRequest,
    EmbeddingOptions,
    ExtractionRequest,
    IngestionRequest,
)
from ..services import GatewayService, get_gateway_service
from ..auth.scopes import Scopes

try:  # pragma: no cover - generated modules may be missing in CI
    from Medical_KG_rev.proto.gen import (
        embedding_pb2,
        embedding_pb2_grpc,
        extraction_pb2,
        extraction_pb2_grpc,
        ingestion_pb2,
        ingestion_pb2_grpc,
        mineru_pb2,
        mineru_pb2_grpc,
    )
except ImportError:  # pragma: no cover - generation happens in CI
    embedding_pb2 = embedding_pb2_grpc = None
    extraction_pb2 = extraction_pb2_grpc = None
    ingestion_pb2 = ingestion_pb2_grpc = None
    mineru_pb2 = mineru_pb2_grpc = None


class MineruService(mineru_pb2_grpc.MineruServiceServicer if mineru_pb2_grpc else object):
    def __init__(self, service: GatewayService) -> None:
        self.service = service

    async def ProcessPdf(self, request, context):  # type: ignore[override]
        chunk_request = ChunkRequest(tenant_id=request.tenant_id, document_id=request.document_id)
        chunks = self.service.chunk_document(chunk_request)
        if mineru_pb2 is None:
            return None
        started_at = datetime.now(timezone.utc)
        response = mineru_pb2.ProcessPdfResponse()
        document = response.document
        document.document_id = request.document_id
        document.tenant_id = request.tenant_id
        for chunk in chunks:
            block = document.blocks.add()
            block.id = f"{chunk.document_id}-chunk-{chunk.chunk_index}"
            block.page = getattr(chunk, "page", 0)
            block.kind = "chunk"
            block.text = chunk.content or ""
            block.confidence = 1.0
            block.reading_order = int(chunk.chunk_index)
        metadata = response.metadata
        metadata.document_id = request.document_id
        metadata.worker_id = "gateway"
        metadata.started_at = started_at.isoformat()
        metadata.completed_at = datetime.now(timezone.utc).isoformat()
        metadata.duration_seconds = 0.0
        return response

    async def BatchProcessPdf(self, request, context):  # type: ignore[override]
        if mineru_pb2 is None:
            return None
        reply = mineru_pb2.BatchProcessPdfResponse()
        for item in request.requests:
            single = await self.ProcessPdf(item, context)
            if single is None:
                continue
            document = reply.documents.add()
            document.CopyFrom(single.document)
            metadata = reply.metadata.add()
            metadata.CopyFrom(single.metadata)
        return reply


class EmbeddingService(
    embedding_pb2_grpc.EmbeddingServiceServicer if embedding_pb2_grpc else object
):
    def __init__(self, service: GatewayService) -> None:
        self.service = service

    async def Embed(self, request, context):  # type: ignore[override]
        embed_request = EmbedRequest(
            tenant_id=request.tenant_id,
            texts=list(request.inputs),
            namespace=request.namespace,
            options=EmbeddingOptions(normalize=request.normalize),
        )
        result = self.service.embed(embed_request)
        if embedding_pb2 is None:
            return None
        response = embedding_pb2.EmbedResponse()
        response.namespace = result.namespace
        if result.metadata:
            response.metadata.provider = result.metadata.provider
            if result.metadata.dimension is not None:
                response.metadata.dimension = int(result.metadata.dimension)
            response.metadata.duration_ms = float(result.metadata.duration_ms)
            response.metadata.model = result.metadata.model
        for vector in result.embeddings:
            message = response.embeddings.add()
            message.id = vector.id
            message.model = vector.model
            message.namespace = vector.namespace
            message.kind = vector.kind
            message.dimension = int(vector.dimension)
            if vector.vector:
                message.values.extend(vector.vector)
            if vector.terms:
                message.terms.update(vector.terms)
            for key, value in vector.metadata.items():
                message.metadata[key] = str(value)
        return response

    async def ListNamespaces(self, request, context):  # type: ignore[override]
        if embedding_pb2 is None:
            return None
        namespaces = self.service.list_namespaces(
            tenant_id=request.tenant_id,
            scope=Scopes.EMBED_READ,
        )
        response = embedding_pb2.ListNamespacesResponse()
        for item in namespaces:
            proto = response.namespaces.add()
            proto.id = item.id
            proto.provider = item.provider
            proto.kind = item.kind
            if item.dimension is not None:
                proto.dimension = int(item.dimension)
            if item.max_tokens is not None:
                proto.max_tokens = int(item.max_tokens)
            proto.enabled = bool(item.enabled)
            proto.allowed_tenants.extend(item.allowed_tenants)
            proto.allowed_scopes.extend(item.allowed_scopes)
        return response

    async def ValidateTexts(self, request, context):  # type: ignore[override]
        if embedding_pb2 is None:
            return None
        result = self.service.validate_namespace_texts(
            tenant_id=request.tenant_id,
            namespace=request.namespace,
            texts=list(request.texts),
        )
        response = embedding_pb2.ValidateTextsResponse()
        response.namespace = result.namespace
        response.valid = result.valid
        for entry in result.results:
            proto = response.results.add()
            proto.text_index = entry.text_index
            proto.token_count = entry.token_count
            proto.exceeds_budget = entry.exceeds_budget
            if entry.warning:
                proto.warning = entry.warning
        return response


class ExtractionService(
    extraction_pb2_grpc.ExtractionServiceServicer if extraction_pb2_grpc else object
):
    def __init__(self, service: GatewayService) -> None:
        self.service = service

    async def Extract(self, request, context):  # type: ignore[override]
        extraction_request = ExtractionRequest(
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            options={},
        )
        result = self.service.extract(request.kind, extraction_request)
        if extraction_pb2 is None:
            return None
        response = extraction_pb2.ExtractionResponse()
        for item in result.results:
            response.results.add(
                kind=result.kind, document_id=result.document_id, value=item.get("value", "")
            )
        return response


class IngestionService(
    ingestion_pb2_grpc.IngestionServiceServicer if ingestion_pb2_grpc else object
):
    def __init__(self, service: GatewayService) -> None:
        self.service = service

    async def Submit(self, request, context):  # type: ignore[override]
        ingestion_request = IngestionRequest(
            tenant_id=request.tenant_id,
            items=[{"id": item_id} for item_id in request.item_ids],
            metadata={"dataset": request.dataset},
        )
        result = self.service.ingest(request.dataset, ingestion_request)
        if ingestion_pb2 is None:
            return None
        response = ingestion_pb2.IngestionJobResponse()
        for status in result.operations:
            response.operations.add(
                job_id=status.job_id, status=status.status, message=status.message or ""
            )
        return response


class GatewayGrpcServer:
    """Wrapper that registers all gateway services with a grpc.aio server."""

    def __init__(self, service: GatewayService | None = None) -> None:
        self.service = service or get_gateway_service()
        self._server: grpc.aio.Server | None = None

    async def start(self, host: str = "0.0.0.0", port: int = 50051) -> None:
        self._server = grpc.aio.server()
        if mineru_pb2_grpc:
            mineru_pb2_grpc.add_MineruServiceServicer_to_server(
                MineruService(self.service), self._server
            )
        if embedding_pb2_grpc:
            embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
                EmbeddingService(self.service), self._server
            )
        if extraction_pb2_grpc:
            extraction_pb2_grpc.add_ExtractionServiceServicer_to_server(
                ExtractionService(self.service), self._server
            )
        if ingestion_pb2_grpc:
            ingestion_pb2_grpc.add_IngestionServiceServicer_to_server(
                IngestionService(self.service), self._server
            )

        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self._server)
        health_servicer.set("GatewayService", health_pb2.HealthCheckResponse.SERVING)

        self._server.add_insecure_port(f"{host}:{port}")
        await self._server.start()

    async def wait_for_termination(self) -> None:
        if self._server is None:
            return
        await self._server.wait_for_termination()


async def serve() -> None:
    server = GatewayGrpcServer()
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    asyncio.run(serve())
