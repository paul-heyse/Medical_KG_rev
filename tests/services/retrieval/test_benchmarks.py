from Medical_KG_rev.services.retrieval.benchmarks import benchmark_reranking_latency
from Medical_KG_rev.services.retrieval.reranker import CrossEncoderReranker


def _documents(count: int = 100) -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    for index in range(count):
        docs.append({
            "id": f"doc-{index}",
            "text": f"Document {index} about headaches and treatments",
            "score": 0.5,
        })
    return docs


def test_benchmark_reranking_latency_meets_target() -> None:
    reranker = CrossEncoderReranker()
    stats = benchmark_reranking_latency(reranker, "headache", _documents(), runs=5, top_k=50)

    assert stats["p95_ms"] < 150.0
    assert stats["runs"] == 5.0
