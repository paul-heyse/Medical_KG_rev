"""FastAPI application wiring all protocol handlers."""

from __future__ import annotations

import logging
import uuid
from time import perf_counter
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .graphql.schema import graphql_router
from .models import ProblemDetail
from .middleware import CachePolicy, CachingMiddleware
from .rest.router import JSONAPI_CONTENT_TYPE, health_router, router as rest_router
from .services import GatewayError
from .sse.routes import router as sse_router
from .soap.routes import router as soap_router
from ..config.settings import get_settings
from ..observability import setup_observability
from ..services.health import CheckResult, HealthService, success
from ..utils.logging import (
    bind_correlation_id,
    get_correlation_id,
    get_logger,
    reset_correlation_id,
)
from .services import get_gateway_service


class JSONAPIResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        if response.media_type == "application/json":
            response.media_type = JSONAPI_CONTENT_TYPE
        return response


logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, *, correlation_header: str) -> None:  # type: ignore[override]
        super().__init__(app)
        self._correlation_header = correlation_header

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        provided = request.headers.get(self._correlation_header) if self._correlation_header else None
        existing = getattr(request.state, "correlation_id", None) or get_correlation_id()
        correlation_id = provided or existing or str(uuid.uuid4())
        setattr(request.state, "correlation_id", correlation_id)
        token = bind_correlation_id(correlation_id)
        started = perf_counter()

        logger.info(
            "gateway.request",
            extra={"method": request.method, "path": request.url.path, "correlation_id": correlation_id},
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started) * 1000
            logger.exception(
                "gateway.request.error",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                    "correlation_id": correlation_id,
                },
            )
            reset_correlation_id(token)
            raise

        duration_ms = (perf_counter() - started) * 1000
        logger.info(
            "gateway.response",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "correlation_id": correlation_id,
            },
        )

        if self._correlation_header:
            response.headers.setdefault(self._correlation_header, correlation_id)

        reset_correlation_id(token)
        return response


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
        response.headers.setdefault("Strict-Transport-Security", f"max-age={self._cfg.hsts_max_age}; includeSubDomains")
        response.headers.setdefault("Content-Security-Policy", self._cfg.content_security_policy)
        response.headers.setdefault("X-Frame-Options", self._cfg.frame_options)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-XSS-Protection", "1; mode=block")
        return response


def create_problem_response(detail: ProblemDetail) -> JSONResponse:
    payload: Dict[str, Any] = detail.model_dump(mode="json")
    status = payload.get("status", 500)
    return JSONResponse(payload, status_code=status, media_type="application/problem+json")


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
        RequestLoggingMiddleware,
        correlation_header=settings.observability.logging.correlation_id_header,
    )
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

    def kafka_check() -> CheckResult:
        health = gateway_service.orchestrator.kafka.health()
        if all(health.values()):
            return success("brokers reachable")
        detail = ", ".join(f"{key}={value}" for key, value in health.items())
        return CheckResult(status="error", detail=detail)

    app.state.health = HealthService(
        checks={
            "neo4j": lambda: success("neo4j stub"),
            "opensearch": lambda: success("opensearch stub"),
            "kafka": kafka_check,
            "redis": lambda: success("redis stub"),
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

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return create_problem_response(exc.detail)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
        detail = ProblemDetail(
            title=str(exc.detail),
            status=exc.status_code,
            type="https://httpstatuses.com/" + str(exc.status_code),
        )
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
        return create_problem_response(detail)

    return app
