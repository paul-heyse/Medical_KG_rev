from Medical_KG_rev.services.vector_store.gpu import (
    GPUFallbackStrategy,
    GPUResourceManager,
    plan_batches,
    summarise_stats,
)


def test_plan_batches_handles_cpu_fallback() -> None:
    manager = GPUResourceManager(require_gpu=False, preferred_batch_size=2)
    batches = plan_batches(5, manager=manager)
    lengths = [batch.stop - batch.start for batch in batches]
    assert lengths == [2, 2, 1]


def test_fallback_strategy_logs_when_missing() -> None:
    messages: list[str] = []
    strategy = GPUFallbackStrategy(logger=messages.append)
    available = strategy.guard(operation="test", require_gpu=False)
    assert available is False or available is True
    assert messages


def test_summarise_stats_empty() -> None:
    summary = summarise_stats([])
    assert summary["devices"] == 0
