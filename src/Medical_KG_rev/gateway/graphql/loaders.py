"""GraphQL DataLoader utilities for the gateway.

Key Responsibilities:
    - Provide reusable Strawberry ``DataLoader`` instances for common gateway
      lookups
    - Cache in-flight lookups to minimise duplicated downstream calls
    - Expose loader factories that can be shared through the GraphQL context

Collaborators:
    - Upstream: GraphQL resolvers fetch data through these loaders
    - Downstream: Gateway service abstractions perform the actual data access

Side Effects:
    - Issues RPC/HTTP calls via the injected ``GatewayService``

Thread Safety:
    - Loaders are intended to be scoped per-request and are not thread-safe

Performance Characteristics:
    - Batching reduces the number of downstream calls for identical keys
"""

from __future__ import annotations

from collections.abc import Iterable

from strawberry.dataloader import DataLoader

from ..models import DocumentSummary
from ..services import GatewayService

# ==============================================================================
# LOADER DEFINITIONS
# ==============================================================================


class GraphQLLoaders:
    """Collection of lazily evaluated loaders shared via GraphQL context.

    Attributes:
        service: Gateway service responsible for fulfilling data requests.
        document_loader: Loader for fetching :class:`DocumentSummary` objects.
        organization_loader: Loader for fetching organisation metadata.
    """

    def __init__(self, service: GatewayService) -> None:
        """Initialise loader collection.

        Args:
            service: Gateway service used for downstream operations.
        """
        self.service = service
        self.document_loader: DataLoader[str, DocumentSummary] = DataLoader(self._load_documents)
        self.organization_loader: DataLoader[str, dict] = DataLoader(self._load_organizations)

    async def _load_documents(self, identifiers: Iterable[str]) -> list[DocumentSummary]:
        """Load document summaries for the supplied identifiers.

        Args:
            identifiers: Iterable of document identifiers to resolve.

        Returns:
            Ordered list of document summaries matching the identifiers.

        Raises:
            NotImplementedError: Always raised until a concrete implementation
                is provided.
        """
        raise NotImplementedError(
            "Document loading not implemented. "
            "This GraphQL loader requires a real document service implementation. "
            "Please implement or configure a proper document service."
        )

    async def _load_organizations(self, identifiers: Iterable[str]) -> list[dict]:
        """Load organisation metadata for the supplied identifiers.

        Args:
            identifiers: Iterable of organisation identifiers to resolve.

        Returns:
            Ordered list of organisation payloads matching the identifiers.

        Raises:
            NotImplementedError: Always raised until a concrete implementation
                is provided.
        """
        raise NotImplementedError(
            "Organization loading not implemented. "
            "This GraphQL loader requires a real organization service implementation. "
            "Please implement or configure a proper organization service."
        )
