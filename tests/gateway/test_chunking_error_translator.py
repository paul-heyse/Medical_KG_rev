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


from __future__ import annotations

from Medical_KG_rev.chunking.exceptions import (
    ChunkingUnavailableError,
    InvalidDocumentError,
    ProfileNotFoundError,
)
from Medical_KG_rev.gateway.chunking import ChunkingErrorTranslator
from Medical_KG_rev.services.retrieval.chunking_command import ChunkCommand


def _profile_not_found() -> ProfileNotFoundError:
    return ProfileNotFoundError("semantic", ("semantic", "section"))


def test_translator_maps_profile_not_found() -> None:
    calls: list[tuple[str | None, str]] = []
    translator = ChunkingErrorTranslator(
        available_strategies=lambda: ["semantic"],
        failure_recorder=lambda profile, error: calls.append((profile, error)),
    )
    command = ChunkCommand(
        tenant_id="tenant-x",
        document_id="doc-42",
        text="content",
        metadata={"profile": "semantic"},
    )
    result = translator.translate(_profile_not_found(), command=command)
    assert result.detail.status == 400
    assert result.detail.extensions["available_profiles"]
    assert result.failure_type == "ProfileNotFoundError"
    assert calls == [("semantic", "ProfileNotFoundError")]


def test_translator_handles_invalid_document_without_command() -> None:
    translator = ChunkingErrorTranslator(available_strategies=lambda: [])
    exc = InvalidDocumentError("bad input")
    result = translator.translate(exc, command=None)
    assert result.detail.status == 400
    assert result.detail.instance is None


def test_translator_reraises_unknown_errors() -> None:
    translator = ChunkingErrorTranslator(available_strategies=lambda: [])
    command = ChunkCommand(tenant_id="tenant-x", document_id="doc-42", text="body")

    class UnexpectedError(RuntimeError):
        pass

    try:
        translator.translate(UnexpectedError("boom"), command=command)
    except UnexpectedError:
        pass
    else:  # pragma: no cover - ensure branch fails test if not raised
        raise AssertionError("Translator should re-raise unexpected exceptions")


def test_translator_includes_retry_after_for_unavailable() -> None:
    translator = ChunkingErrorTranslator(available_strategies=lambda: [])
    command = ChunkCommand(tenant_id="tenant-x", document_id="doc-42", text="body")
    exc = ChunkingUnavailableError(5.2)
    result = translator.translate(exc, command=command)
    assert result.detail.status == 503
    assert result.detail.extensions["retry_after"] == 5
