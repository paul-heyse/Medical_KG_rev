from __future__ import annotations

from Medical_KG_rev.services.retrieval.opensearch_client import (
    DocumentIndexTemplate,
    OpenSearchClient,
)


def test_bulk_index_and_search_with_filters():
    client = OpenSearchClient()
    client.put_index_template(
        DocumentIndexTemplate(
            name="documents", settings={}, mappings={"properties": {"text": {"type": "text"}}}
        )
    )

    documents = [
        {"id": "1", "text": "The patient experienced headaches", "source": "clinical"},
        {"id": "2", "text": "Headache and nausea reported", "source": "trial"},
    ]
    client.bulk_index("documents", documents, id_field="id")

    results = client.search("documents", "headache", filters={"source": "trial"})

    assert len(results) == 1
    assert results[0]["_id"] == "2"
    assert any(span["term"] == "headache" for span in results[0]["highlight"])


def test_splade_strategy_scores_unique_terms():
    client = OpenSearchClient()
    client.index("documents", "1", {"text": "unique term"})
    client.index("documents", "2", {"text": "repeated term term"})

    results = client.search("documents", "term unique", strategy="splade")

    assert results[0]["_id"] == "1"


def test_index_propagates_chunking_profile_for_filters():
    client = OpenSearchClient()
    client.index(
        "chunks",
        "chunk-1",
        {
            "text": "Immunotherapy improved outcomes",
            "metadata": {"chunking_profile": "pmc-imrad"},
        },
    )
    client.index(
        "chunks",
        "chunk-2",
        {
            "text": "Registry adverse events recorded",
            "metadata": {"chunking_profile": "ctgov-registry"},
        },
    )

    filtered = client.search(
        "chunks",
        "events",
        filters={"chunking_profile": "ctgov-registry"},
    )

    assert len(filtered) == 1
    assert filtered[0]["_id"] == "chunk-2"
    assert filtered[0]["_source"]["chunking_profile"] == "ctgov-registry"
