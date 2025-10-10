"""HTTP wrapper mixin for common HTTP operations."""

from __future__ import annotations

from typing import Any

from Medical_KG_rev.utils.http_client import HttpClient


class HTTPWrapperMixin:
    """Mixin providing common HTTP operations for adapters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._http_client: HttpClient | None = None

    @property
    def http_client(self) -> HttpClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = HttpClient(
                base_url=getattr(self, "base_url", ""),
                retry=getattr(self, "retry", None),
                rate_limit=getattr(self, "rate_limit", None),
                circuit_breaker=getattr(self, "circuit_breaker", None),
            )
        return self._http_client

    def _get_json(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Make GET request and return JSON response."""
        response = self.http_client.request("GET", path, **kwargs)
        json_data = response.json()
        return json_data if isinstance(json_data, dict) else {}

    def _get_text(self, path: str, **kwargs: Any) -> str:
        """Make GET request and return text response."""
        response = self.http_client.request("GET", path, **kwargs)
        return response.text

    def _post_json(
        self, path: str, data: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Make POST request with JSON data and return JSON response."""
        response = self.http_client.request("POST", path, json=data, **kwargs)
        json_data = response.json()
        return json_data if isinstance(json_data, dict) else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Make HTTP request."""
        return self.http_client.request(method, path, **kwargs)
