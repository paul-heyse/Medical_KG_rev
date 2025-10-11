"""Open access metadata mixin for shaping OA-related data."""

from __future__ import annotations

from typing import Any



class OpenAccessMetadataMixin:
    """Mixin providing open access metadata utilities."""

    def shape_open_access_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Shape open access metadata into standard format."""
        metadata = {
            "is_open_access": False,
            "oa_status": "closed",
            "oa_url": None,
            "license": None,
            "repository": None,
        }

        # Check for open access indicators
        oa_indicators = [
            "is_oa",
            "is_open_access",
            "open_access",
            "oa_status",
            "access_rights",
        ]

        for indicator in oa_indicators:
            if indicator in data:
                value = data[indicator]
                if isinstance(value, bool):
                    metadata["is_open_access"] = value
                    metadata["oa_status"] = "open" if value else "closed"
                elif isinstance(value, str):
                    metadata["oa_status"] = value.lower()
                    metadata["is_open_access"] = value.lower() in ["open", "true", "yes"]
                break

        # Extract open access URL
        url_fields = ["oa_url", "open_access_url", "pdf_url", "download_url", "url"]
        for field in url_fields:
            if data.get(field):
                metadata["oa_url"] = data[field]
                break

        # Extract license information
        license_fields = ["license", "licence", "rights", "copyright"]
        for field in license_fields:
            if data.get(field):
                metadata["license"] = data[field]
                break

        # Extract repository information
        repo_fields = ["repository", "repo", "source", "provider"]
        for field in repo_fields:
            if data.get(field):
                metadata["repository"] = data[field]
                break

        return metadata

    def detect_open_access_type(self, data: dict[str, Any]) -> str:
        """Detect the type of open access."""
        # Check for specific OA types
        oa_types = {
            "gold": ["gold", "journal", "publisher"],
            "green": ["green", "repository", "archive"],
            "hybrid": ["hybrid", "mixed"],
            "bronze": ["bronze", "delayed"],
            "diamond": ["diamond", "platinum"],
        }

        # Check various fields for OA type indicators
        check_fields = [
            "oa_type",
            "access_type",
            "open_access_type",
            "publisher_type",
            "source_type",
        ]

        for field in check_fields:
            if field in data:
                value = str(data[field]).lower()
                for oa_type, keywords in oa_types.items():
                    if any(keyword in value for keyword in keywords):
                        return oa_type

        # Default based on URL patterns
        url = data.get("oa_url", "") or data.get("url", "")
        if url:
            if "doi.org" in url or "publisher" in url.lower():
                return "gold"
            elif any(repo in url.lower() for repo in ["arxiv", "pmc", "biorxiv", "zenodo"]):
                return "green"

        return "unknown"

    def extract_license_info(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract license information."""
        license_info = {
            "license": None,
            "license_url": None,
            "license_type": None,
        }

        # Extract license text
        license_fields = ["license", "licence", "rights", "copyright"]
        for field in license_fields:
            if data.get(field):
                license_info["license"] = data[field]
                break

        # Extract license URL
        url_fields = ["license_url", "licence_url", "rights_url", "copyright_url"]
        for field in url_fields:
            if data.get(field):
                license_info["license_url"] = data[field]
                break

        # Determine license type
        if license_info["license"]:
            license_text = license_info["license"].lower()
            if "cc-by-sa" in license_text:
                license_info["license_type"] = "cc-by-sa"
            elif "cc-by-nc-sa" in license_text:
                license_info["license_type"] = "cc-by-nc-sa"
            elif "cc-by-nc" in license_text:
                license_info["license_type"] = "cc-by-nc"
            elif "cc-by" in license_text:
                license_info["license_type"] = "cc-by"
            elif "cc0" in license_text:
                license_info["license_type"] = "cc0"
            elif "mit" in license_text:
                license_info["license_type"] = "mit"
            elif "apache" in license_text:
                license_info["license_type"] = "apache"
            elif "gpl" in license_text:
                license_info["license_type"] = "gpl"
            else:
                license_info["license_type"] = "other"

        return license_info
