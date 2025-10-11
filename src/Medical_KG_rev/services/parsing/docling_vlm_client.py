"""Minimal asynchronous client stub for the Docling VLM service.

The original implementation depended on a rich gRPC stack that is not present
in the torch-free fork. To keep downstream imports functioning we provide a
small shim that exposes the same high level methods but simply raises
``DoclingVLMError`` when invoked.
"""

from __future__ import annotations

import logging
from typing import Any

from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMResult
from Medical_KG_rev.services.parsing.exceptions import DoclingVLMError

logger = logging.getLogger(__name__)


class DoclingVLMClient:
    """Lightweight placeholder client."""

    def __init__(self, service_endpoint: str = "docling-vlm:9000") -> None:
        self.service_endpoint = service_endpoint
        logger.warning("Docling VLM gRPC client is not available in torch-free mode")

    async def process_pdf(
        self,
        pdf_path: str,
        config: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> DoclingVLMResult:
        """Process a PDF via the VLM service (not implemented)."""
        raise DoclingVLMError("Docling VLM client not available in this build")

    async def close(self) -> None:
        """Close any underlying resources (no-op)."""
        return None
