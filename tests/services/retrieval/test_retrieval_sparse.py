from Medical_KG_rev.services.retrieval.sparse import (
    BM25FRetriever,
    BM25Retriever,
    NeuralSparseRetriever,
    SPLADEDocWriter,
    SPLADEQueryEncoder,
    SparseDocument,
    SparseQueryBuilder,
)


def _documents() -> list[SparseDocument]:
    return [
        SparseDocument(
            doc_id="1",
            text="hypertension treatment reduces systolic blood pressure",
            fields={"arm": "treatment"},
        ),
        SparseDocument(
            doc_id="2", text="placebo arm recorded minimal improvement", fields={"arm": "control"}
        ),
        SparseDocument(
            doc_id="3", text="adverse events were rare in cohort", fields={"arm": "treatment"}
        ),
    ]


def test_bm25_filters_and_scores() -> None:
    retriever = BM25Retriever()
    retriever.index_documents(_documents())
    results = retriever.search("blood pressure", filters={"arm": "treatment"}, top_k=2)
    assert results
    ids = {doc.doc_id for _id, _score, doc in results}
    assert "1" in ids


def test_bm25f_applies_field_boosts() -> None:
    retriever = BM25FRetriever(boosts={"arm": 2.0})
    retriever.index_documents(_documents())
    results = retriever.search("placebo arm", top_k=1)
    assert results[0][0] == "2"


def test_splade_writer_and_neural_retriever() -> None:
    writer = SPLADEDocWriter()
    writer.write("1", {"hypertension": 1.0, "blood": 0.8})
    writer.write("2", {"placebo": 0.6})
    retriever = NeuralSparseRetriever(writer)
    results = retriever.search(query="blood pressure", top_k=2)
    assert results[0][0] == "1"


def test_sparse_query_builder() -> None:
    builder = SparseQueryBuilder()
    query = (
        builder.add_term("text", "blood")
        .add_rank_feature("rank_features.splade", 1.5)
        .add_filter("arm", "treatment")
        .build()
    )
    assert "bool" in query
    assert any(item["term"] == {"arm": "treatment"} for item in query["bool"]["filter"])


def test_splade_encoder_normalises_terms() -> None:
    encoder = SPLADEQueryEncoder()
    weights = encoder.encode("blood blood pressure")
    assert weights["blood"] > weights["pressure"]
