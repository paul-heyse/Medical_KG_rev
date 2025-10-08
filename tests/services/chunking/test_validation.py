import pytest

from Medical_KG_rev.services.chunking.port import Chunk
from Medical_KG_rev.services.chunking.validation import (
    ChunkValidationError,
    ensure_valid_chunks,
    validate_chunk,
)


def _chunk(**overrides) -> Chunk:
    data = {
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
        "text": "content",
        "char_offsets": (0, 7),
        "section_label": "Intro",
        "intent_hint": "narrative",
        "metadata": {
            "chunking_profile": "default",
            "source_system": "unit-test",
            "chunker_version": "simple",
            "created_at": "2024-01-01T00:00:00Z",
        },
    }
    data.update(overrides)
    return Chunk(**data)


def test_validate_chunk_success():
    result = validate_chunk(_chunk())
    assert result.valid is True


def test_validate_chunk_missing_metadata_raises():
    chunk = _chunk(metadata={"chunking_profile": "default"})
    with pytest.raises(ChunkValidationError):
        ensure_valid_chunks([chunk])


def test_validate_chunk_invalid_offsets():
    chunk = _chunk(char_offsets=(5, 2))
    result = validate_chunk(chunk)
    assert result.valid is False
    assert "ordered" in (result.reason or "")


def test_validate_chunk_schema_errors():
    bad_chunk = _chunk(text="", metadata={})
    result = validate_chunk(bad_chunk)
    assert result.valid is False
    assert "at least 1 character" in (result.reason or "")
