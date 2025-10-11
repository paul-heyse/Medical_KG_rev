"""gRPC server stubs wired to the gateway service layer."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from Medical_KG_rev.auth.scopes import Scopes
from Medical_KG_rev.gateway.models import (
    EmbeddingOptions,
    EmbedRequest,
    ExtractionRequest,
    IngestionRequest,
)
from Medical_KG_rev.gateway.services import GatewayService, get_gateway_service

# Proto imports - handle missing generated files gracefully
try:
    from Medical_KG_rev.proto.gen import (
        embedding_pb2,
        embedding_pb2_grpc,
        extraction_pb2,
        extraction_pb2_grpc,
        ingestion_pb2,
        ingestion_pb2_grpc,
    )
except ImportError:  # pragma: no cover - generation happens in CI
    embedding_pb2 = embedding_pb2_grpc = None
    extraction_pb2 = extraction_pb2_grpc = None
    ingestion_pb2 = ingestion_pb2_grpc = None


class EmbeddingService(
    embedding_pb2_grpc.EmbeddingServiceServicer if embedding_pb2_grpc else object
):
    """gRPC service for embedding operations."""

    def __init__(self, service: GatewayService) -> None:
        """Initialize the embedding service."""
        self.service = service

    async def Embed(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Any:
        """Handle embedding requests."""
        if not embedding_pb2:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Embedding service not available")
            return None

        try:
            # Convert gRPC request to gateway model
            embed_request = EmbedRequest(
                tenant_id=request.tenant_id,
                namespace=request.namespace,
                texts=list(request.texts),
                model=request.model,
                options=EmbeddingOptions(
                    normalize=request.options.normalize if request.options else False,
                    return_metadata=request.options.return_metadata if request.options else False,
                ),
            )

            # Process embedding
            response = await self.service.embed_text(embed_request)

            # Convert response to gRPC format
            return embedding_pb2.EmbedResponse(
                vectors=[
                    embedding_pb2.EmbeddingVector(
                        id=vector.id,
                        model=vector.model,
                        values=vector.values,
                        metadata=vector.metadata,
                    )
                    for vector in response.vectors
                ],
                processing_time=response.processing_time,
                model_used=response.model_used,
                namespace_used=response.namespace_used,
            )

        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Embedding failed: {exc}")
            return None


class ExtractionService(
    extraction_pb2_grpc.ExtractionServiceServicer if extraction_pb2_grpc else object
):
    """gRPC service for extraction operations."""

    def __init__(self, service: GatewayService) -> None:
        """Initialize the extraction service."""
        self.service = service

    async def Extract(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Any:
        """Handle extraction requests."""
        if not extraction_pb2:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Extraction service not available")
            return None

        try:
            # Convert gRPC request to gateway model
            extraction_request = ExtractionRequest(
                tenant_id=request.tenant_id,
                document_id=request.document_id,
                content=request.content,
                extraction_type=request.extraction_type,
                options=request.options,
            )

            # Process extraction
            response = await self.service.extract_entities(extraction_request)

            # Convert response to gRPC format
            return extraction_pb2.ExtractionResponse(
                entities=[
                    extraction_pb2.Entity(
                        id=entity.id,
                        type=entity.type,
                        text=entity.text,
                        confidence=entity.confidence,
                        metadata=entity.metadata,
                    )
                    for entity in response.entities
                ],
                processing_time=response.processing_time,
                extraction_type=response.extraction_type,
            )

        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Extraction failed: {exc}")
            return None


class IngestionService(
    ingestion_pb2_grpc.IngestionServiceServicer if ingestion_pb2_grpc else object
):
    """gRPC service for ingestion operations."""

    def __init__(self, service: GatewayService) -> None:
        """Initialize the ingestion service."""
        self.service = service

    async def Ingest(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Any:
        """Handle ingestion requests."""
        if not ingestion_pb2:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Ingestion service not available")
            return None

        try:
            # Convert gRPC request to gateway model
            ingestion_request = IngestionRequest(
                tenant_id=request.tenant_id,
                document_id=request.document_id,
                content=request.content,
                content_type=request.content_type,
                metadata=request.metadata,
            )

            # Process ingestion
            response = await self.service.ingest_document(ingestion_request)

            # Convert response to gRPC format
            return ingestion_pb2.IngestionResponse(
                document_id=response.document_id,
                status=response.status,
                processing_time=response.processing_time,
                metadata=response.metadata,
            )

        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Ingestion failed: {exc}")
            return None


class HealthService(health_pb2_grpc.HealthServicer):
    """gRPC health check service."""

    def __init__(self) -> None:
        """Initialize the health service."""
        self._status = health_pb2.HealthCheckResponse.SERVING

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        """Check service health."""
        return health_pb2.HealthCheckResponse(status=self._status)

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> None:
        """Watch service health."""
        # Simple implementation - just send current status
        yield health_pb2.HealthCheckResponse(status=self._status)


class GRPCServer:
    """gRPC server for gateway services."""

    def __init__(self, port: int = 50051) -> None:
        """Initialize the gRPC server."""
        self.port = port
        self.server = grpc.aio.server()
        self.gateway_service = get_gateway_service()

    def add_services(self) -> None:
        """Add services to the gRPC server."""
        # Add embedding service
        if embedding_pb2_grpc:
            embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
                EmbeddingService(self.gateway_service), self.server
            )

        # Add extraction service
        if extraction_pb2_grpc:
            extraction_pb2_grpc.add_ExtractionServiceServicer_to_server(
                ExtractionService(self.gateway_service), self.server
            )

        # Add ingestion service
        if ingestion_pb2_grpc:
            ingestion_pb2_grpc.add_IngestionServiceServicer_to_server(
                IngestionService(self.gateway_service), self.server
            )

        # Add health service
        health_pb2_grpc.add_HealthServicer_to_server(HealthService(), self.server)

    async def start(self) -> None:
        """Start the gRPC server."""
        self.add_services()

        listen_addr = f"[::]:{self.port}"
        self.server.add_insecure_port(listen_addr)

        await self.server.start()
        print(f"gRPC server started on {listen_addr}")

    async def stop(self) -> None:
        """Stop the gRPC server."""
        await self.server.stop(grace=5.0)

    async def serve(self) -> None:
        """Serve the gRPC server."""
        await self.start()
        try:
            await self.server.wait_for_termination()
        except KeyboardInterrupt:
            await self.stop()


async def main() -> None:
    """Main function to run the gRPC server."""
    server = GRPCServer()
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
