"""Tests for identifier utilities."""

from Medical_KG_rev.utils.identifiers import build_document_id, hash_content, normalize_identifier


def test_hash_content_deterministic() -> None:
    """`hash_content` should return a stable 12 character digest."""
    value = hash_content("clinical-trial-123")
    assert value == hash_content("clinical-trial-123")
    assert len(value) == 12
    assert value.isalnum()


def test_build_document_id_uses_content_hash() -> None:
    """When content is provided the identifier includes the derived suffix."""
    doc_id = build_document_id("ctgov", "NCT00000000", content="payload")
    assert doc_id.startswith("ctgov:NCT00000000#v1:")
    suffix = doc_id.split(":")[-1]
    assert suffix == hash_content("payload")


def test_build_document_id_random_suffix_when_no_content(monkeypatch) -> None:
    """When content is absent the suffix should be random but deterministic for the test."""
    monkeypatch.setattr("Medical_KG_rev.utils.identifiers.secrets.token_hex", lambda size: "a1b2c3d4e5f6")
    doc_id = build_document_id("mesh", "D012345")
    assert doc_id == "mesh:D012345#v1:a1b2c3d4e5f6"


def test_normalize_identifier_strips_and_lowercases() -> None:
    """`normalize_identifier` should trim whitespace and lowercase."""
    assert normalize_identifier("  RXNORM:12345  ") == "rxnorm:12345"
