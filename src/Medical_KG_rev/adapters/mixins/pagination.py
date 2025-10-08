"""Pagination mixin for handling paginated API responses."""

from __future__ import annotations

from typing import Any, Callable, Generator, Iterable


class PaginationMixin:
    """Mixin providing pagination utilities for adapters."""

    def paginate_results(
        self,
        fetch_func: Callable[..., dict[str, Any]],
        *args: Any,
        page_size: int = 100,
        max_pages: int | None = None,
        **kwargs: Any
    ) -> Generator[dict[str, Any], None, None]:
        """Paginate through API results."""
        page = 1
        pages_fetched = 0

        while True:
            if max_pages and pages_fetched >= max_pages:
                break

            # Fetch current page
            page_data = fetch_func(*args, page=page, per_page=page_size, **kwargs)

            if not page_data:
                break

            # Extract items from page data
            items = self._extract_items_from_page(page_data)
            if not items:
                break

            # Yield each item
            for item in items:
                yield item

            # Check if there are more pages
            if not self._has_next_page(page_data):
                break

            page += 1
            pages_fetched += 1

    def _extract_items_from_page(self, page_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract items from paginated response."""
        # Common field names for items
        item_fields = ["items", "results", "data", "works", "studies", "records"]

        for field in item_fields:
            if field in page_data and isinstance(page_data[field], list):
                items = page_data[field]
                if items and isinstance(items[0], dict):
                    return [item for item in items if isinstance(item, dict)]

        # If no standard field found, return empty list
        return []

    def _has_next_page(self, page_data: dict[str, Any]) -> bool:
        """Check if there are more pages available."""
        # Common pagination indicators
        pagination_fields = [
            "has_next",
            "next_page",
            "next",
            "more_results",
            "has_more",
        ]

        for field in pagination_fields:
            if field in page_data:
                value = page_data[field]
                return bool(value) if isinstance(value, (bool, int, str)) else False

        # Check for page number indicators
        if "page" in page_data and "total_pages" in page_data:
            page_num = page_data["page"]
            total_pages = page_data["total_pages"]
            if isinstance(page_num, int) and isinstance(total_pages, int):
                return page_num < total_pages

        if "current_page" in page_data and "total_pages" in page_data:
            current_page = page_data["current_page"]
            total_pages = page_data["total_pages"]
            if isinstance(current_page, int) and isinstance(total_pages, int):
                return current_page < total_pages

        # Default to False if no pagination info found
        return False

    def get_total_count(self, page_data: dict[str, Any]) -> int | None:
        """Get total count from paginated response."""
        count_fields = ["total", "total_count", "total_results", "count"]

        for field in count_fields:
            if field in page_data:
                return int(page_data[field])

        return None
