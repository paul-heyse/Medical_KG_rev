"""GraphQL DataLoader utilities for the gateway."""

from __future__ import annotations

from collections.abc import Iterable

from strawberry.dataloader import DataLoader

from ..models import DocumentSummary
from ..services import GatewayService


class GraphQLLoaders:
    """Collection of lazily evaluated loaders shared via context."""

    def __init__(self, service: GatewayService) -> None:
        self.service = service
        self.document_loader: DataLoader[str, DocumentSummary] = DataLoader(self._load_documents)
        self.organization_loader: DataLoader[str, dict] = DataLoader(self._load_organizations)

    async def _load_documents(self, identifiers: Iterable[str]) -> list[DocumentSummary]:
        documents: list[DocumentSummary] = []
        for identifier in identifiers:
            documents.append(
                DocumentSummary(
                    id=identifier,
                    title=f"Document {identifier}",
                    score=0.9,
                    summary="Loaded via DataLoader",
                    source="loader",
                    metadata={},
                )
            )
        return documents

    async def _load_organizations(self, identifiers: Iterable[str]) -> list[dict]:
        return [
            {
                "id": identifier,
                "name": f"Organization {identifier}",
                "country": "US",
            }
            for identifier in identifiers
        ]
