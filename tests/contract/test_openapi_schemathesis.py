import pytest

schemathesis = pytest.importorskip("schemathesis")

from Medical_KG_rev.gateway.app import create_app

_app = create_app()
schema = schemathesis.openapi.from_asgi("/openapi.json", _app)


def test_openapi_contract_validates() -> None:
    schema.validate()
