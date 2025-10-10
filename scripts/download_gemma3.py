#!/usr/bin/env python3
#!/usr/bin/env python3
"""Utility script to download and validate the Gemma3 12B Docling checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

DEFAULT_REPO = "google/gemma-3-12b-it"
DEFAULT_TARGET = Path("/models/docling-vlm")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Hugging Face repository containing the Gemma3 weights",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Directory where the model snapshot should be stored",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face access token (falls back to env configuration)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision or branch to download",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the target directory without downloading",
    )
    return parser.parse_args()


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download(repo: str, target: Path, token: str | None, revision: str) -> Path:
    snapshot_path = snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )
    return Path(snapshot_path)


def _validate_snapshot(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Gemma3 configuration not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required_keys = {"model_type", "architectures"}
    if not required_keys.issubset(config):
        missing = sorted(required_keys - set(config))
        raise ValueError(f"Gemma3 config missing keys: {missing}")
    return config


def main() -> int:
    args = _parse_args()
    target = args.target.expanduser().resolve()
    _ensure_directory(target)

    if not args.verify_only:
        try:
            _download(args.repo, target, args.token, args.revision)
        except Exception as exc:  # pragma: no cover - network/hub errors
            print(f"Failed to download Gemma3 snapshot: {exc}", file=sys.stderr)
            return 1

    try:
        config = _validate_snapshot(target)
    except Exception as exc:  # pragma: no cover - validation errors
        print(f"Gemma3 snapshot validation failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "path": str(target),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
