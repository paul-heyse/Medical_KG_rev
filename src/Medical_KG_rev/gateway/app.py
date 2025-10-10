"""FastAPI application wiring all protocol handlers.

This module provides the main FastAPI application that integrates all protocol
handlers (REST, GraphQL, gRPC, SOAP, SSE) into a unified gateway. It handles
middleware configuration, error handling, and application lifecycle management.

Key Responsibilities:
    - Application initialization and configuration
    - Middleware setup (CORS, security headers, caching, tenant validation)
    - Protocol handler integration (REST, GraphQL, gRPC, SOAP, SSE)
    - Error handling and exception translation
    - Health check endpoints and monitoring integration
    - Static file serving and documentation endpoints

Collaborators:
    - Upstream: ASGI server (Uvicorn, Gunicorn)
    - Downstream: All protocol handlers, middleware components, services

Side Effects:
    - Starts observability instrumentation
    - Configures middleware pipeline
    - Mounts static files and documentation
    - Registers all protocol routes

Thread Safety:
    - Thread-safe: FastAPI application is designed for concurrent requests
    - Middleware components handle concurrent access appropriately

Performance Characteristics:
    - O(1) request routing overhead
    - Middleware pipeline adds minimal latency
    - Static file serving optimized for production

Example:
    >>> from Medical_KG_rev.gateway.app import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn Medical_KG_rev.gateway.app:app

"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from Medical_KG_rev.chunking.exceptions import (
    ChunkingFailedError,
    MineruGpuUnavailableError,
    MineruOutOfMemoryError,
    ProfileNotFoundError,
    TokenizerMismatchError,
)

from ..config.settings import PdfProcessingBackend, get_settings
from ..observability import setup_observability
from ..services.health import CheckResult, HealthService, success
from ..services.parsing.docling_vlm_service import DoclingVLMService
from ..utils.logging import get_correlation_id, get_logger
from .graphql.schema import graphql_router
from .middleware import CachePolicy, CachingMiddleware, TenantValidationMiddleware
from .models import ProblemDetail
from .presentation.lifecycle import RequestLifecycleMiddleware
from .rest.router import JSONAPI_CONTENT_TYPE, health_router
from .rest.router import router as rest_router
from .services import GatewayError, get_gateway_service
from .soap.routes import router as soap_router
from .sse.routes import router as sse_router

# ==============================================================================
# MIDDLEWARE IMPLEMENTATION
# ==============================================================================


class JSONAPIResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        if response.media_type == "application/json":
            response.media_type = JSONAPI_CONTENT_TYPE
        return response


# ==============================================================================
# ERROR HANDLING
# ==============================================================================

logger = get_logger(__name__)


def create_problem_response(detail: ProblemDetail) -> JSONResponse:
    """Create a JSON response for problem details.

    Args:
        detail: Problem detail object containing error information.

    Returns:
        JSON response with appropriate status code and headers.

    """
    payload: dict[str, Any] = detail.model_dump(mode="json")
    status = payload.get("status", 500)
    headers: dict[str, str] | None = None
    retry_after = (
        detail.extensions.get("retry_after") if isinstance(detail.extensions, dict) else None
    )
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        headers = {"Retry-After": str(int(retry_after))}
    return JSONResponse(
        payload,
        status_code=status,
        media_type="application/problem+json",
        headers=headers,
    )


# ==============================================================================
# MIDDLEWARE IMPLEMENTATION
# ==============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, *, headers_config) -> None:  # type: ignore[override]
        super().__init__(app)
        self._cfg = headers_config

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.url.scheme != "https" and request.app.state.settings.security.enforce_https:
            forwarded_proto = request.headers.get("x-forwarded-proto")
            if forwarded_proto != "https":
                raise HTTPException(status_code=400, detail="HTTPS is required")
        response = await call_next(request)
        response.headers.setdefault(
            "Strict-Transport-Security", f"max-age={self._cfg.hsts_max_age}; includeSubDomains"
        )
        response.headers.setdefault("Content-Security-Policy", self._cfg.content_security_policy)
        response.headers.setdefault("X-Frame-Options", self._cfg.frame_options)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-XSS-Protection", "1; mode=block")
        return response


# ==============================================================================
# APPLICATION FACTORY
# ==============================================================================


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Medical KG Multi-Protocol Gateway", version="0.1.0")
    app.state.settings = settings
    app.state.jwt_cache = {}

    setup_observability(app, settings)

    cache_settings = settings.caching

    def to_policy(policy) -> CachePolicy:
        return CachePolicy(
            ttl=policy.ttl,
            scope=policy.scope,
            vary=tuple(policy.vary),
            etag=policy.etag,
            last_modified=policy.last_modified,
        )

    app.add_middleware(
        CachingMiddleware,
        policies={path: to_policy(policy) for path, policy in cache_settings.endpoints.items()},
        default_policy=to_policy(cache_settings.default),
    )

    app.add_middleware(JSONAPIResponseMiddleware)
    app.add_middleware(
        RequestLifecycleMiddleware,
        correlation_header=settings.observability.logging.correlation_id_header,
    )
    app.add_middleware(TenantValidationMiddleware)
    app.add_middleware(SecurityHeadersMiddleware, headers_config=settings.security.headers)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.security.cors.allow_origins),
        allow_methods=list(settings.security.cors.allow_methods),
        allow_headers=list(settings.security.cors.allow_headers),
    )

    app.include_router(health_router)
    app.include_router(rest_router)
    app.include_router(sse_router)
    app.include_router(graphql_router, prefix="/graphql")
    app.include_router(soap_router)
    app.mount("/static", StaticFiles(directory="docs"), name="static")

    gateway_service = get_gateway_service()
    docling_service = DoclingVLMService(settings.docling_vlm)

    def kafka_check() -> CheckResult:
        health = gateway_service.orchestrator.kafka.health()
        if all(health.values()):
            return success("brokers reachable")
        detail = ", ".join(f"{key}={value}" for key, value in health.items())
        return CheckResult(status="error", detail=detail)

    def docling_check() -> CheckResult:
        if settings.pdf_processing_backend != PdfProcessingBackend.DOCLING_VLM:
            return success("Docling backend disabled")
        return docling_service.health()

    app.state.health = HealthService(
        checks={
            "neo4j": lambda: success("neo4j stub"),
            "opensearch": lambda: success("opensearch stub"),
            "kafka": kafka_check,
            "redis": lambda: success("redis stub"),
            "docling_vlm": docling_check,
        },
        version=app.version,
    )

    @app.get("/docs/openapi", include_in_schema=False)
    async def openapi_docs() -> HTMLResponse:
        return get_swagger_ui_html(openapi_url="/openapi.json", title="REST API documentation")

    @app.get("/docs/graphql", include_in_schema=False)
    async def graphql_docs() -> HTMLResponse:
        return HTMLResponse(
            """
            <html>
            <body>
            <iframe src="/graphql" style="width:100%;height:100vh;border:0;"></iframe>
            </body>
            </html>
            """,
            media_type="text/html",
        )

    @app.get("/docs/asyncapi", include_in_schema=False)
    async def asyncapi_docs() -> HTMLResponse:
        return HTMLResponse(
            """
            <html>
            <body>
            <h1>AsyncAPI Streams</h1>
            <p>Download the specification at <a href="/static/asyncapi.yaml">/static/asyncapi.yaml</a>.</p>
            </body>
            </html>
            """,
            media_type="text/html",
        )

    def _log_problem(event: str, detail: ProblemDetail) -> None:
        logger.error(
            event,
            extra={
                "correlation_id": get_correlation_id(),
                "problem": detail.model_dump(mode="json"),
            },
        )

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        _log_problem("gateway.error", exc.detail)
        return create_problem_response(exc.detail)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
        detail = ProblemDetail(
            title=str(exc.detail),
            status=exc.status_code,
            type="https://httpstatuses.com/" + str(exc.status_code),
        )
        _log_problem("gateway.http_error", detail)
        return create_problem_response(detail)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = ProblemDetail(
            title="Request validation failed",
            status=422,
            type="https://httpstatuses.com/422",
            detail="One or more parameters are invalid.",
            extensions={"errors": exc.errors()},
        )
        _log_problem("gateway.validation_error", detail)
        return create_problem_response(detail)

    @app.exception_handler(ProfileNotFoundError)
    async def handle_profile_error(_: Request, exc: ProfileNotFoundError) -> JSONResponse:
        detail = ProblemDetail(
            title="Chunking profile not found",
            status=400,
            type="https://medical-kg/errors/chunking-profile-not-found",
            detail=str(exc),
            extensions={"available_profiles": list(getattr(exc, "available", []))},
        )
        _log_problem("gateway.chunking.profile_not_found", detail)
        return create_problem_response(detail)

    @app.exception_handler(TokenizerMismatchError)
    async def handle_tokenizer_error(_: Request, exc: TokenizerMismatchError) -> JSONResponse:
        detail = ProblemDetail(
            title="Tokenizer mismatch",
            status=500,
            type="https://medical-kg/errors/tokenizer-mismatch",
            detail=str(exc),
        )
        _log_problem("gateway.chunking.tokenizer_mismatch", detail)
        return create_problem_response(detail)

    @app.exception_handler(ChunkingFailedError)
    async def handle_chunking_failed(_: Request, exc: ChunkingFailedError) -> JSONResponse:
        message = exc.detail or str(exc) or "Chunking process failed"
        detail = ProblemDetail(
            title="Chunking failed",
            status=500,
            type="https://medical-kg/errors/chunking-failed",
            detail=message,
        )
        _log_problem("gateway.chunking.failed", detail)
        return create_problem_response(detail)

    @app.exception_handler(MineruOutOfMemoryError)
    async def handle_mineru_oom(_: Request, exc: MineruOutOfMemoryError) -> JSONResponse:
        detail = ProblemDetail(
            title="MinerU out of memory",
            status=503,
            type="https://medical-kg/errors/mineru-oom",
            detail=str(exc),
            extensions={"reason": "gpu_out_of_memory"},
        )
        _log_problem("gateway.mineru.out_of_memory", detail)
        return create_problem_response(detail)

    @app.exception_handler(MineruGpuUnavailableError)
    async def handle_mineru_unavailable(_: Request, exc: MineruGpuUnavailableError) -> JSONResponse:
        detail = ProblemDetail(
            title="MinerU GPU unavailable",
            status=503,
            type="https://medical-kg/errors/mineru-gpu-unavailable",
            detail=str(exc),
            extensions={"reason": "gpu_unavailable"},
        )
        _log_problem("gateway.mineru.gpu_unavailable", detail)
        return create_problem_response(detail)

    return app


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "JSONAPIResponseMiddleware",
    "SecurityHeadersMiddleware",
    "create_app",
    "create_problem_response",
]
