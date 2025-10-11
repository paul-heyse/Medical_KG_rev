"""Shared mixins for common adapter behaviors."""

from .doi_normalization import DOINormalizationMixin
from .http_wrapper import HTTPWrapperMixin
from .open_access_metadata import OpenAccessMetadataMixin
from .pagination import PaginationMixin
from .pdf_manifest import PdfManifestMixin
from .storage_helpers import StorageHelperMixin


__all__ = [
    "DOINormalizationMixin",
    "HTTPWrapperMixin",
    "OpenAccessMetadataMixin",
    "PaginationMixin",
    "PdfManifestMixin",
    "StorageHelperMixin",
]
