from __future__ import annotations

import pytest

from Medical_KG_rev.services.retrieval.query_dsl import QueryDSL, QueryValidationError


def test_query_dsl_validates_filters_and_facets():
    dsl = QueryDSL(allowed_filters={"source": {"trial", "clinical"}, "status": set()})
    payload = {"filters": {"source": "trial"}, "facets": ["source"]}
    parsed = dsl.parse(payload)
    assert parsed["filters"] == {"source": "trial"}
    assert parsed["facets"] == ["source"]


def test_query_dsl_rejects_unknown_filter():
    dsl = QueryDSL(allowed_filters={})
    with pytest.raises(QueryValidationError):
        dsl.parse({"filters": {"unknown": "value"}})
