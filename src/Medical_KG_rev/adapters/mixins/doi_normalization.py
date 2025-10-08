"""DOI normalization mixin for consistent DOI handling."""

from __future__ import annotations

import re
from typing import Any

from Medical_KG_rev.utils.validation import validate_doi


class DOINormalizationMixin:
    """Mixin providing DOI normalization utilities."""

    # DOI regex pattern
    DOI_PATTERN = re.compile(r"^10\.\d{4,}/.*", re.IGNORECASE)

    def normalize_doi(self, doi: str) -> str:
        """Normalize DOI format."""
        if not doi:
            return ""

        # Remove any whitespace
        doi = doi.strip()

        # Remove common prefixes
        prefixes = ["doi:", "DOI:", "https://doi.org/", "http://dx.doi.org/"]
        for prefix in prefixes:
            if doi.lower().startswith(prefix.lower()):
                doi = doi[len(prefix):]
                break

        # Validate DOI format
        try:
            validate_doi(doi)
            return doi
        except ValueError:
            # If validation fails, return original
            return doi

    def extract_doi_from_url(self, url: str) -> str | None:
        """Extract DOI from URL."""
        if not url:
            return None

        # Common DOI URL patterns
        patterns = [
            r"https?://doi\.org/(10\.\d{4,}/.*)",
            r"https?://dx\.doi\.org/(10\.\d{4,}/.*)",
            r"https?://.*\.doi\.org/(10\.\d{4,}/.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return self.normalize_doi(match.group(1))

        return None

    def is_valid_doi(self, doi: str) -> bool:
        """Check if DOI is valid."""
        try:
            validate_doi(doi)
            return True
        except ValueError:
            return False

    def extract_dois_from_text(self, text: str) -> list[str]:
        """Extract all DOIs from text."""
        if not text:
            return []

        # Find all DOI patterns in text
        matches = self.DOI_PATTERN.findall(text)
        return [self.normalize_doi(match) for match in matches if self.is_valid_doi(match)]
