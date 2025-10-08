from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class PdfMetadata:
    """Metadata describing a remotely hosted PDF resource."""

    url: str
    content_type: str | None
    size: int | None
    last_modified: datetime | None
    accessible: bool
    headers: dict[str, Any]

    @property
    def normalized_content_type(self) -> str | None:
        if not self.content_type:
            return None
        return self.content_type.lower().strip()


class PdfUrlValidator:
    """Validate that a PDF URL is reachable and resembles a PDF resource."""

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        max_attempts: int = 3,
        user_agent: str = "Medical-KG-PDF-Validator/1.0",
    ) -> None:
        self._timeout = timeout
        self._retry = Retrying(
            stop=stop_after_attempt(max(1, max_attempts)),
            wait=wait_exponential(multiplier=0.75, min=0.5, max=5.0),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        self._headers = {"User-Agent": user_agent}

    def validate(self, url: str) -> PdfMetadata:
        """Perform a HEAD request to validate the target URL."""

        def _head_request() -> httpx.Response:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                response = client.request("HEAD", url, headers=self._headers)
                if response.status_code == 405:
                    response = client.request("GET", url, headers=self._headers, stream=True)
                    response.read(1024)
                response.raise_for_status()
                return response

        try:
            response = self._retry(_head_request)
        except RetryError as exc:
            error = exc.last_attempt.exception()
            logger.warning(
                "pdf.validator.unreachable",
                url=url,
                error=str(error),
            )
            raise
        except httpx.HTTPError as exc:
            logger.warning("pdf.validator.head_failed", url=url, error=str(exc))
            raise

        content_type = response.headers.get("Content-Type")
        size_header = response.headers.get("Content-Length")
        last_modified_header = response.headers.get("Last-Modified")

        content_length: int | None = None
        if size_header is not None:
            try:
                content_length = int(size_header)
            except ValueError:
                logger.debug(
                    "pdf.validator.invalid_length",
                    url=url,
                    content_length=size_header,
                )
                content_length = None

        last_modified: datetime | None = None
        if last_modified_header:
            try:
                last_modified = datetime.strptime(last_modified_header, "%a, %d %b %Y %H:%M:%S %Z")
                if last_modified.tzinfo is None:
                    last_modified = last_modified.replace(tzinfo=timezone.utc)
            except Exception:
                last_modified = None

        accessible = True
        if content_type and "pdf" not in content_type.lower():
            logger.debug("pdf.validator.suspect_mime", url=url, content_type=content_type)

        metadata = PdfMetadata(
            url=url,
            content_type=content_type,
            size=content_length,
            last_modified=last_modified,
            accessible=accessible,
            headers=dict(response.headers),
        )
        return metadata


__all__ = ["PdfMetadata", "PdfUrlValidator"]
