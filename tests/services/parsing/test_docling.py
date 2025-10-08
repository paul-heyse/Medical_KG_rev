from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from Medical_KG_rev.services.parsing.docling import DoclingParser


class _FakePartition:
    def __call__(self, *, content: bytes, format: str):
        return [SimpleNamespace(text=content.decode("utf-8"), metadata={})]


@pytest.fixture(autouse=True)
def _inject_fake_docling(monkeypatch):
    fake_module = ModuleType("docling")
    fake_module.partition = _FakePartition()
    monkeypatch.setitem(sys.modules, "docling", fake_module)
    yield
    sys.modules.pop("docling", None)


def test_docling_rejects_pdf():
    parser = DoclingParser()
    with pytest.raises(ValueError) as exc:
        parser.parse(content=b"pdf", fmt="pdf", doc_id="doc")
    assert "Docling cannot be used for PDF parsing" in str(exc.value)


def test_docling_parses_html():
    parser = DoclingParser()
    document = parser.parse(content=b"<p>Hello</p>", fmt="html", doc_id="doc")
    assert document.source == "docling-html"
    assert document.sections[0].blocks[0].text == "<p>Hello</p>"
