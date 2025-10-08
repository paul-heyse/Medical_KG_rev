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
