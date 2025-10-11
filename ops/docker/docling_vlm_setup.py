"""Helper script for preparing the Docling VLM runtime image."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import structlog
from Medical_KG_rev.config.docling_config import DoclingVLMConfig
from Medical_KG_rev.services import GpuNotAvailableError
from Medical_KG_rev.services.parsing.docling_vlm_service import DoclingVLMService

try:  # pragma: no cover - optional dependency
    from huggingface_hub import login, snapshot_download
except Exception:  # pragma: no cover - huggingface hub is optional outside docker builds
    login = None  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


def ensure_cache_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_model(config: DoclingVLMConfig, *, token: str | None = None) -> Path:
    if snapshot_download is None:
        logger.warning("huggingface_hub.missing", message="Skipping model download")
        return config.model_path

    if token and login is not None:
        login(token=token, add_to_git_credential=True)

    logger.info("docling_vlm.download", model=config.model_name, target=str(config.model_path))
    snapshot_download(
        repo_id=config.model_name,
        local_dir=str(config.model_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return config.model_path


def validate_model_assets(path: Path) -> dict[str, Any]:
    manifest = {
        "path": str(path),
        "exists": path.exists(),
        "files": [],
    }
    if not path.exists():
        logger.warning("docling_vlm.cache_missing", path=str(path))
        return manifest

    files = [p.name for p in path.glob("**/*") if p.is_file()]
    manifest["files"] = files
    if not files:
        logger.warning("docling_vlm.cache_empty", path=str(path))
    return manifest


def warmup_model(config: DoclingVLMConfig) -> None:
    logger.info("docling_vlm.warmup.start")
    try:
        service = DoclingVLMService(config=config, eager=True)
    except GpuNotAvailableError as exc:  # pragma: no cover - GPU optional during builds
        logger.warning("docling_vlm.warmup.skipped", reason=str(exc))
        return
    health = service.health()
    logger.info("docling_vlm.warmup.complete", health=json.dumps(health))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensure-cache", action="store_true", help="Create cache directory")
    parser.add_argument("--download", action="store_true", help="Download the Gemma3 model")
    parser.add_argument("--warmup", action="store_true", help="Warm up the Docling pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to Docling YAML config",
    )
    parser.add_argument("--token", type=str, default=None, help="HuggingFace access token")
    args = parser.parse_args(argv)

    config = (
        DoclingVLMConfig.from_yaml(args.config) if args.config else DoclingVLMConfig.from_dict(None)
    )
    ensure_cache_directory(config.model_path)

    if args.ensure_cache:
        logger.info("docling_vlm.cache_ready", path=str(config.model_path))

    if args.download:
        token = args.token or os.environ.get("HUGGINGFACE_TOKEN")
        download_model(config, token=token)

    manifest = validate_model_assets(config.model_path)

    if args.warmup:
        warmup_model(config)

    logger.info("docling_vlm.manifest", manifest=json.dumps(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover - invoked during Docker build
    raise SystemExit(main(sys.argv[1:]))
