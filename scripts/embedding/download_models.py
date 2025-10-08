"""Utility script to download embedding models required by the vLLM/Pyserini stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QWEN_REPO = "Qwen/Qwen2.5-Coder-1.5B"
QWEN_TARGET = Path("models/qwen3-embedding-8b")
SPLADE_MODEL = "naver/splade-v3"
SPLADE_TARGET = Path("models/splade-v3")


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_qwen(local_dir: Path = QWEN_TARGET) -> Path:
    _ensure_directory(local_dir)
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "huggingface_hub is required to download the Qwen3 embedding model"
        ) from exc
    snapshot_download(
        repo_id=QWEN_REPO,
        local_dir=local_dir,
        allow_patterns=["*.json", "*.bin", "*.model", "tokenizer*"],
    )
    return local_dir


def download_splade(cache_dir: Path = SPLADE_TARGET) -> Path:
    _ensure_directory(cache_dir)
    try:
        from pyserini.encode import SpladeQueryEncoder
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyserini is required to materialize the SPLADE encoder") from exc
    encoder = SpladeQueryEncoder(SPLADE_MODEL, cache_dir=str(cache_dir))
    # Trigger materialization by encoding a trivial document.
    encoder.encode("initialization test", top_k=1)
    return cache_dir


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format", choices={"text", "json"}, default="text", help="Output format"
    )
    args = parser.parse_args()

    results: dict[str, str] = {}
    try:
        qwen_path = download_qwen()
        results["qwen3"] = str(qwen_path.resolve())
    except Exception as exc:  # pragma: no cover - best effort logging
        results["qwen3_error"] = str(exc)
    try:
        splade_path = download_splade()
        results["splade_v3"] = str(splade_path.resolve())
    except Exception as exc:  # pragma: no cover - best effort logging
        results["splade_v3_error"] = str(exc)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        for key, value in results.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    cli()
