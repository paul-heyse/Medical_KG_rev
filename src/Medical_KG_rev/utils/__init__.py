"""Utility modules for the foundation layer."""
from .errors import FoundationError, ProblemDetail
from .http_client import AsyncHttpClient, HttpClient, RateLimiter, RetryConfig
from .identifiers import build_document_id, hash_content, normalize_identifier
from .logging import (
    bind_correlation_id,
    configure_logging,
    configure_tracing,
    get_correlation_id,
    get_logger,
    reset_correlation_id,
)
from .metadata import flatten_metadata
from .spans import merge_overlapping, spans_within
from .time import ensure_utc, utc_now
from .validation import validate_doi, validate_nct_id, validate_pmcid, validate_pmid
from .versioning import Version

__all__ = [
    "AsyncHttpClient",
    "FoundationError",
    "HttpClient",
    "ProblemDetail",
    "RateLimiter",
    "RetryConfig",
    "bind_correlation_id",
    "build_document_id",
    "configure_logging",
    "configure_tracing",
    "ensure_utc",
    "flatten_metadata",
    "get_correlation_id",
    "get_logger",
    "hash_content",
    "merge_overlapping",
    "normalize_identifier",
    "reset_correlation_id",
    "spans_within",
    "utc_now",
    "validate_doi",
    "validate_nct_id",
    "validate_pmcid",
    "validate_pmid",
    "Version",
]
