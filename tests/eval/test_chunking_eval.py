import importlib.util
import sys
from pathlib import Path


def _load_runner():
    module_path = Path(__file__).resolve().parents[2] / "eval" / "chunking_eval.py"
    spec = importlib.util.spec_from_file_location("chunking_eval", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module.ChunkingEvaluationRunner


def test_chunking_evaluation_runner() -> None:
    ChunkingEvaluationRunner = _load_runner()
    runner = ChunkingEvaluationRunner(
        [
            "section_aware",
            "semantic_splitter",
        ]
    )
    summaries = runner.run()
    assert "section_aware" in summaries
    summary = summaries["section_aware"]
    assert 0.0 <= summary.boundary_f1 <= 1.0
    assert summary.chunk_count > 0
    assert summary.latency_ms >= 0.0
