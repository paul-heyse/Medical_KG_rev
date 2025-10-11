import pytest
from fastapi.testclient import TestClient

from Medical_KG_rev.gateway.app import create_app
from Medical_KG_rev.gateway.coordinators import EmbeddingCoordinator
from Medical_KG_rev.services.evaluation import EvaluationConfig, EvaluationResult, MetricSummary


@pytest.fixture
def client(monkeypatch) -> TestClient:
    app = create_app()
    result = EvaluationResult(
        dataset="demo",
        test_set_version="v1",
        metrics={
            "recall@10": MetricSummary(mean=0.82, median=0.82, std=0.0, ci_low=0.8, ci_high=0.84),
            "ndcg@10": MetricSummary(mean=0.78, median=0.78, std=0.0, ci_low=0.76, ci_high=0.8),
            "mrr": MetricSummary(mean=0.9, median=0.9, std=0.0),
        },
        latency=MetricSummary(mean=12.0, median=12.0, std=0.5),
        per_query={"Q1": {"recall@10": 1.0}},
        per_query_type={"exact_term": {"recall@10": 0.9}},
        cache_key="abc123",
        cache_hit=False,
        config=EvaluationConfig(top_k=10),
    )

    # Mock evaluation service - evaluation is handled by coordinators now
    monkeypatch.setattr("Medical_KG_rev.gateway.app.evaluation_service", lambda req: result)
    return TestClient(app)


def test_evaluation_endpoint_returns_metrics(client: TestClient) -> None:
    response = client.post(
        "/v1/evaluate",
        json={"tenant_id": "tenant", "test_set_name": "test_set_v1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["dataset"] == "demo"
    assert payload["data"]["metrics"]["recall@10"]["mean"] == pytest.approx(0.82)
    assert payload["meta"]["cache"]["hit"] is False
