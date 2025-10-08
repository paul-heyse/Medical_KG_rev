from pathlib import Path

import pytest

from Medical_KG_rev.services.evaluation.test_sets import (
    QueryType,
    TestSetManager,
    build_test_set,
    cohens_kappa,
)


def test_loads_yaml_test_set() -> None:
    manager = TestSetManager(root=Path("eval/test_sets"))
    test_set = manager.load("test_set_v1")
    assert test_set.version == "v1"
    assert len(test_set.queries) == 3
    assert {query.query_type for query in test_set.queries} == {
        QueryType.COMPLEX_CLINICAL,
        QueryType.EXACT_TERM,
        QueryType.PARAPHRASE,
    }


def test_build_test_set_validates_schema() -> None:
    queries = [
        {
            "query_id": "QX",
            "query_text": "example",
            "query_type": "exact_term",
            "relevant_docs": [{"doc_id": "A", "grade": 4}],
        }
    ]
    with pytest.raises(ValueError):
        build_test_set("invalid", queries, version="v0")


def test_split_preserves_counts(tmp_path: Path) -> None:
    manager = TestSetManager(root=tmp_path)
    payload = [
        {
            "query_id": f"Q{i}",
            "query_text": f"text-{i}",
            "query_type": "exact_term" if i % 2 == 0 else "paraphrase",
            "relevant_docs": [{"doc_id": f"D{i}", "grade": 1}],
        }
        for i in range(10)
    ]
    test_set = build_test_set("custom", payload, version="v1")
    eval_set, holdout = test_set.split(holdout_ratio=0.2, seed=1)
    assert len(eval_set.queries) + len(holdout.queries) == len(test_set.queries)
    assert eval_set.version == holdout.version == "v1"


def test_refresh_writes_version(tmp_path: Path) -> None:
    manager = TestSetManager(root=tmp_path)
    payload = [
        {
            "query_id": "Q1",
            "query_text": "abc",
            "query_type": "exact_term",
            "relevant_docs": [{"doc_id": "D1", "grade": 1}],
        }
    ]
    refreshed = manager.refresh("dataset", new_queries=payload, version="v2")
    assert refreshed.version == "v2"
    loaded = manager.load("dataset")
    assert loaded.version == "v2"


def test_cohens_kappa_handles_agreement() -> None:
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)


