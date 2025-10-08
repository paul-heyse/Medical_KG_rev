"""Check GPU availability and dependency versions before starting vLLM."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

REQUIRED_BINARIES = ["nvidia-smi"]
REQUIRED_MODULES = ["torch", "vllm", "pyserini", "faiss"]


def _check_gpu() -> dict[str, str | bool]:
    nvidia_smi = shutil.which("nvidia-smi")
    available = nvidia_smi is not None
    info: dict[str, str | bool] = {"available": available}
    if available and nvidia_smi:
        info["path"] = nvidia_smi
    return info


def _module_versions() -> dict[str, str | None]:
    results: dict[str, str | None] = {}
    for module in REQUIRED_MODULES:
        try:
            pkg = importlib.import_module(module)
            version = getattr(pkg, "__version__", "unknown")
        except ModuleNotFoundError:
            version = None
        results[module] = version
    return results


def check_environment() -> dict[str, object]:
    return {
        "gpu": _check_gpu(),
        "modules": _module_versions(),
        "models": {
            "qwen3": Path("models/qwen3-embedding-8b").exists(),
            "splade_v3": Path("models/splade-v3").exists(),
        },
    }


def main() -> None:
    print(json.dumps(check_environment(), indent=2))


if __name__ == "__main__":
    main()
