from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from Medical_KG_rev.services.parsing.unstructured_parser import UnstructuredParser


class _FakeElement:
    def __init__(self, text: str, section: str | None = None) -> None:
        self.text = text
        self.metadata = SimpleNamespace(section=section, to_dict=lambda: {"section": section})


@pytest.fixture(autouse=True)
def _inject_fake_unstructured(monkeypatch):
    xml_module = ModuleType("unstructured.partition.xml")
    xml_module.partition_xml = lambda *, text: [_FakeElement(text, "Section")]
    html_module = ModuleType("unstructured.partition.html")
    html_module.partition_html = lambda *, text: [_FakeElement(text, None)]
    partition_pkg = ModuleType("unstructured.partition")
    partition_pkg.xml = xml_module
    partition_pkg.html = html_module
    unstructured_pkg = ModuleType("unstructured")
    unstructured_pkg.partition = partition_pkg
    monkeypatch.setitem(sys.modules, "unstructured", unstructured_pkg)
    monkeypatch.setitem(sys.modules, "unstructured.partition", partition_pkg)
    monkeypatch.setitem(sys.modules, "unstructured.partition.xml", xml_module)
    monkeypatch.setitem(sys.modules, "unstructured.partition.html", html_module)
    yield
    sys.modules.pop("unstructured", None)
    sys.modules.pop("unstructured.partition", None)
    sys.modules.pop("unstructured.partition.xml", None)
    sys.modules.pop("unstructured.partition.html", None)


def test_unstructured_xml():
    parser = UnstructuredParser()
    document = parser.parse(content="<xml>data</xml>", fmt="xml", doc_id="doc")
    assert document.sections[0].title == "Section"
    assert document.sections[0].blocks[0].text == "<xml>data</xml>"


def test_unstructured_html():
    parser = UnstructuredParser()
    document = parser.parse(content="<p>hello</p>", fmt="html", doc_id="doc")
    assert document.sections[0].blocks[0].text == "<p>hello</p>"


def test_unstructured_invalid_format():
    parser = UnstructuredParser()
    with pytest.raises(ValueError):
        parser.parse(content="text", fmt="txt", doc_id="doc")
