"""FastAPI application wiring all protocol handlers."""

from __future__ import annotations

import logging
from typing import Any, Dict

try:
    import structlog
except ImportError:  # pragma: no cover - structlog optional in lightweight envs
    structlog = None

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .graphql.schema import graphql_router
from .models import ProblemDetail
from .rest.router import JSONAPI_CONTENT_TYPE, router as rest_router
from .services import GatewayError
from .sse.routes import router as sse_router
from .soap.routes import router as soap_router
from ..config.settings import get_settings


class JSONAPIResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        if response.media_type == "application/json":
            response.media_type = JSONAPI_CONTENT_TYPE
        return response


logger = structlog.get_logger(__name__) if structlog else logging.getLogger(__name__)

if not structlog:
    logging.basicConfig(level=logging.INFO)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        logger.info("gateway.request", method=request.method, path=request.url.path)
        response = await call_next(request)
        logger.info(
            "gateway.response",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )
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

    app.add_middleware(JSONAPIResponseMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware, headers_config=settings.security.headers)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.security.cors.allow_origins),
        allow_methods=list(settings.security.cors.allow_methods),
        allow_headers=list(settings.security.cors.allow_headers),
    )

    app.include_router(rest_router)
    app.include_router(sse_router)
    app.include_router(graphql_router, prefix="/graphql")
    app.include_router(soap_router)
    app.mount("/static", StaticFiles(directory="docs"), name="static")

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
