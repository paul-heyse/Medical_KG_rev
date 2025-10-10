from __future__ import annotations

import gzip
import json
from urllib.parse import urlencode

import pytest
from pydantic import BaseModel
from starlette.requests import Request

from Medical_KG_rev.auth.context import SecurityContext
from Medical_KG_rev.gateway.presentation.errors import ErrorDetail
from Medical_KG_rev.gateway.presentation.jsonapi import JSONAPIPresenter
from Medical_KG_rev.gateway.presentation.lifecycle import (
    RequestLifecycle,
    pop_lifecycle,
    push_lifecycle,
)
from Medical_KG_rev.gateway.presentation.odata import ODataParams
from Medical_KG_rev.gateway.presentation.requests import apply_tenant_context
from Medical_KG_rev.observability.metrics import REQUEST_COUNTER


def _fake_request(query: dict[str, Any]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": urlencode(query, doseq=True).encode(),
    }
    return Request(scope)


class SampleModel(BaseModel):
    tenant_id: str | None = None
    value: str


def test_jsonapi_presenter_formats_error() -> None:
    presenter = JSONAPIPresenter()
    payload = presenter.success({"id": 1})
    assert payload.media_type == "application/vnd.api+json"
    assert "X-Response-Time-Ms" in payload.headers
    error = presenter.error(
        ErrorDetail(status=404, code="missing", title="Missing", detail="Not found"),
        status_code=404,
    )
    assert error.status_code == 404
    assert error.media_type == "application/vnd.api+json"
    assert "X-Response-Time-Ms" in error.headers


def test_jsonapi_presenter_records_metrics_and_headers() -> None:
    presenter = JSONAPIPresenter()
    lifecycle = RequestLifecycle(method="GET", path="/items", correlation_id="corr-1")
    token = push_lifecycle(lifecycle)
    try:
        counter = REQUEST_COUNTER.labels("GET", "/items", "201")
        before = counter._value.get()  # type: ignore[attr-defined]
        response = presenter.success(
            {"id": 42},
            status_code=201,
            cache_control="public, max-age=60",
        )
        after = counter._value.get()  # type: ignore[attr-defined]
        assert after == before + 1
        assert response.headers["X-Correlation-ID"] == "corr-1"
        assert response.headers["Cache-Control"] == "public, max-age=60"
    finally:
        pop_lifecycle(token)


def test_jsonapi_presenter_supports_compression() -> None:
    presenter = JSONAPIPresenter()
    response = presenter.success({"message": "hi"}, compress=True)
    assert response.headers["Content-Encoding"] == "gzip"
    decompressed = gzip.decompress(response.body).decode("utf-8")  # type: ignore[arg-type]
    payload: dict[str, Any] = json.loads(decompressed)
    assert payload["data"]["message"] == "hi"


def test_apply_tenant_context_sets_tenant_id() -> None:
    security = SecurityContext(subject="user", tenant_id="tenant", scopes=set())
    model = SampleModel(value="ok")
    request = _fake_request({})
    updated = apply_tenant_context(model, security, request)
    assert updated.tenant_id == "tenant"
    assert request.state.requested_tenant_id == "tenant"


def test_apply_tenant_context_rejects_mismatched_tenant() -> None:
    security = SecurityContext(subject="user", tenant_id="tenant", scopes=set())
    model = SampleModel(value="ok", tenant_id="other")
    with pytest.raises(PermissionError):
        apply_tenant_context(model, security)


def test_odata_params_parses_standard_arguments() -> None:
    request = _fake_request({"$select": "id,name", "$top": "5"})
    params = ODataParams.from_request(request)
    assert params.select == ["id", "name"]
    assert params.top == 5
