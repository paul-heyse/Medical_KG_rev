"""Check GPU availability and dependency versions before starting vLLM."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from pathlib import Path

REQUIRED_MODULES = ["torch", "pyserini", "faiss"]
DOCKER_COMPOSE_SERVICE = "vllm-embedding"
DOCKER_COMPOSE_FILE = Path("docker-compose.yml")
VLLM_DOCKERFILE = Path("ops/Dockerfile.vllm")
VLLM_IMAGE = "ghcr.io/example/vllm-embedding:latest"


def _check_gpu() -> dict[str, str | bool]:
    nvidia_smi = shutil.which("nvidia-smi")
    available = nvidia_smi is not None
    info: dict[str, str | bool] = {"available": available}
    if available and nvidia_smi:
        info["path"] = nvidia_smi
    return info


def _check_docker() -> dict[str, object]:
    docker_path = shutil.which("docker")
    info: dict[str, object] = {"available": docker_path is not None}
    if not docker_path:
        return info

    info["path"] = docker_path
    info["dockerfile"] = str(VLLM_DOCKERFILE.resolve()) if VLLM_DOCKERFILE.exists() else None
    if DOCKER_COMPOSE_FILE.exists():
        info["compose_file"] = str(DOCKER_COMPOSE_FILE.resolve())
        try:
            result = subprocess.run(
                [
                    docker_path,
                    "compose",
                    "-f",
                    str(DOCKER_COMPOSE_FILE),
                    "config",
                    "--services",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                services = {
                    line.strip() for line in result.stdout.splitlines() if line.strip()
                }
                info["compose_service"] = DOCKER_COMPOSE_SERVICE in services
            else:
                info["compose_service"] = False
        except FileNotFoundError:
            info["compose_service"] = False
        except Exception:  # pragma: no cover - best effort detection
            info["compose_service"] = False
    else:
        info["compose_file"] = None

    try:
        result = subprocess.run(
            [docker_path, "image", "inspect", VLLM_IMAGE],
            check=False,
            capture_output=True,
        )
        info["image_present"] = result.returncode == 0
    except FileNotFoundError:
        info["image_present"] = False
    except Exception:  # pragma: no cover - best effort detection
        info["image_present"] = False

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
        "docker": _check_docker(),
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
