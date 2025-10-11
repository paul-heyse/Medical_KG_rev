"""Wrapper around the unstructured library."""

from __future__ import annotations

try:
    from unstructured.partition.html import partition_html
    from unstructured.partition.xml import partition_xml
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("unstructured[local-inference]>=0.12.0 is required") from exc

from Medical_KG_rev.models.ir import Block, BlockType, Document, Section


class UnstructuredParser:
    """Parse HTML/XML documents using the unstructured library."""

    def __init__(self) -> None:
        self._partition_xml = partition_xml
        self._partition_html = partition_html

    def parse(self, *, content: str, fmt: str, doc_id: str) -> Document:
        fmt_normalized = fmt.lower()
        if fmt_normalized == "xml":
            elements = self._partition_xml(text=content)
        elif fmt_normalized == "html":
            elements = self._partition_html(text=content)
        else:
            raise ValueError("Unstructured parser only supports 'xml' and 'html'")
        blocks = []
        sections = []
        current_section_blocks: list[Block] = []
        current_section_title: str | None = None
        section_index = 0
        for element in elements:
            metadata = getattr(element, "metadata", {})
            if hasattr(metadata, "to_dict"):
                metadata_dict = metadata.to_dict()  # type: ignore[call-arg]
            elif isinstance(metadata, dict):
                metadata_dict = dict(metadata)
            else:
                metadata_dict = dict(getattr(metadata, "__dict__", {}))
            title = metadata_dict.get("section") or getattr(metadata, "section", None)
            if title != current_section_title:
                if current_section_blocks:
                    sections.append(
                        Section(
                            id=f"{doc_id}-section-{section_index}",
                            title=current_section_title,
                            blocks=current_section_blocks,
                        )
                    )
                    section_index += 1
                current_section_blocks = []
                current_section_title = title
            text = getattr(element, "text", "") or ""
            block = Block(
                id=f"{doc_id}-block-{len(blocks)}",
                type=BlockType.PARAGRAPH,
                text=text,
                metadata=metadata_dict,
            )
            blocks.append(block)
            current_section_blocks.append(block)
        if current_section_blocks:
            sections.append(
                Section(
                    id=f"{doc_id}-section-{section_index}",
                    title=current_section_title,
                    blocks=current_section_blocks,
                )
            )
        return Document(id=doc_id, source=f"unstructured-{fmt_normalized}", sections=sections)
