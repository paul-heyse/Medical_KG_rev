from Medical_KG_rev.utils.metadata import flatten_metadata


def test_flatten_metadata():
    records = [{"a": 1}, {"b": 2}]
    flattened = flatten_metadata(records, prefix="meta.")
    assert flattened["meta.0.a"] == 1
    assert flattened["meta.1.b"] == 2
