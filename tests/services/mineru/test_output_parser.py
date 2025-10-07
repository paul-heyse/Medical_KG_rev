from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from Medical_KG_rev.services.mineru.output_parser import MineruOutputParser


def test_output_parser_parses_tables_and_blocks():
    payload = {
        "document_id": "doc-123",
        "blocks": [
            {
                "id": "blk-1",
                "page": 1,
                "type": "table",
                "text": "A|B",
                "bbox": [0.0, 0.0, 0.5, 0.5],
                "confidence": 0.9,
                "reading_order": 1,
                "table_id": "tbl-1",
            }
        ],
        "tables": [
            {
                "id": "tbl-1",
                "page": 1,
                "headers": ["A", "B"],
                "cells": [
                    {"row": 0, "column": 0, "content": "1"},
                    {"row": 0, "column": 1, "content": "2"},
                ],
            }
        ],
        "figures": [],
        "equations": [],
    }
    parser = MineruOutputParser()
    parsed = parser.parse_dict(payload)
    assert parsed.document_id == "doc-123"
    assert parsed.blocks[0].table_id == "tbl-1"
    assert parsed.tables[0].headers == ("A", "B")
