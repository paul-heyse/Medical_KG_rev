from Medical_KG_rev.utils.identifiers import build_document_id, hash_content, normalize_identifier


def test_hash_content_length():
    digest = hash_content("hello")
    assert len(digest) == 12


def test_build_document_id_deterministic():
    doc_id = build_document_id("source", "123", content="hello")
    assert doc_id.startswith("source:123#v1:")


def test_normalize_identifier():
    assert normalize_identifier(" ABC ") == "abc"
