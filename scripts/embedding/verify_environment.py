"""Check GPU availability and dependency versions before starting vLLM."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from pathlib import Path

REQUIRED_MODULES = ["torch", "pyserini", "faiss"]
DOCKER_COMPOSE_SERVICE = "vllm-qwen3"
DOCKER_COMPOSE_FILE = Path("docker-compose.yml")
VLLM_CATALOG = Path("ops/vllm/catalog.json")


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
    if VLLM_CATALOG.exists():
        info["catalog_path"] = str(VLLM_CATALOG.resolve())
        try:
            catalog_data = json.loads(VLLM_CATALOG.read_text())
        except json.JSONDecodeError:
            catalog_data = {"models": []}
            info["catalog_error"] = "invalid JSON in catalog"
    else:
        catalog_data = {"models": []}
        info["catalog_path"] = None

    dockerfiles: dict[str, str | None] = {}
    images: dict[str, bool] = {}
    catalog_entries: list[dict[str, object]] = []
    for model in catalog_data.get("models", []):
        name = model.get("name", "unknown")
        dockerfile_name = model.get("dockerfile")
        image_tag = model.get("image")
        dockerfile_path = (
            VLLM_CATALOG.parent / dockerfile_name
            if dockerfile_name
            else None
        )
        exists = dockerfile_path.exists() if dockerfile_path else False
        dockerfiles[name] = str(dockerfile_path.resolve()) if exists else None

        image_present = False
        if image_tag and docker_path:
            try:
                result = subprocess.run(
                    [docker_path, "image", "inspect", image_tag],
                    check=False,
                    capture_output=True,
                )
                image_present = result.returncode == 0
            except FileNotFoundError:
                image_present = False
            except Exception:
                image_present = False
        images[image_tag or name] = image_present
        catalog_entries.append(
            {
                "name": name,
                "dockerfile": dockerfiles[name],
                "dockerfile_exists": exists,
                "image": image_tag,
                "image_present": image_present,
            }
        )

    info["dockerfiles"] = dockerfiles
    info["images"] = images
    info["catalog_models"] = catalog_entries
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
