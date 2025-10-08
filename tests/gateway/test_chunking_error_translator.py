import sys
from types import SimpleNamespace

import pytest


class _YamlStub(SimpleNamespace):
    YAMLError = Exception

    @staticmethod
    def safe_load(*_args, **_kwargs):
        return {}


sys.modules.setdefault("yaml", _YamlStub())

from Medical_KG_rev.chunking.exceptions import (
    ChunkerConfigurationError,
    ChunkingUnavailableError,
    InvalidDocumentError,
)
from Medical_KG_rev.gateway.chunking_errors import ChunkingErrorTranslator
from Medical_KG_rev.services.retrieval.chunking import ChunkCommand


@pytest.fixture
def command() -> ChunkCommand:
    return ChunkCommand(
        tenant_id="tenant-a",
        document_id="doc-1",
        text="hello world",
        strategy="section",
        options={"profile": "fast"},
    )


def test_translator_maps_invalid_document(command: ChunkCommand) -> None:
    translator = ChunkingErrorTranslator(strategies=["section", "semantic"])
    report = translator.translate(InvalidDocumentError("bad text"), command=command)

    assert report is not None
    assert report.problem.status == 400
    assert report.metric == "InvalidDocumentError"


def test_translator_maps_configuration_error(command: ChunkCommand) -> None:
    translator = ChunkingErrorTranslator(strategies=["section", "semantic"])
    report = translator.translate(
        ChunkerConfigurationError("unsupported"),
        command=command,
    )

    assert report is not None
    assert report.problem.status == 422
    assert report.problem.extensions["valid_strategies"] == ["section", "semantic"]


def test_translator_maps_unavailable_with_retry(command: ChunkCommand) -> None:
    translator = ChunkingErrorTranslator(strategies=["section"])
    report = translator.translate(ChunkingUnavailableError(retry_after=12), command=command)

    assert report is not None
    assert report.problem.status == 503
    assert report.problem.extensions["retry_after"] == 12
