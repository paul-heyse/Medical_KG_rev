"""Example adapter implementation used for tests and documentation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from Medical_KG_rev.models import Block, Document, Section

from .base import AdapterContext, BaseAdapter


class ExampleAdapter(BaseAdapter):
    """Adapter that produces a single in-memory document."""

    def __init__(self) -> None:
        super().__init__(name="example")

    def fetch(self, context: AdapterContext) -> Iterable[dict]:
        return [{"id": "example-1", "title": "Example"}]

    def parse(self, payloads: Iterable[dict], context: AdapterContext) -> Sequence[Document]:
        documents = []
        for payload in payloads:
            block = Block(id="b1", text="Example block", spans=[])
            section = Section(id="s1", title="Intro", blocks=[block])
            documents.append(
                Document(
                    id=payload["id"], source="example", title=payload["title"], sections=[section]
                )
            )
        return documents

    def write(self, documents: Sequence[Document], context: AdapterContext) -> None:
        # No-op write for example adapter
        return None
