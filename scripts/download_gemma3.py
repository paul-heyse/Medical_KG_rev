#!/usr/bin/env python3
"""Download and validate the Gemma3 12B Docling checkpoint.

This helper script wraps :func:`huggingface_hub.snapshot_download` with
Docling-specific defaults. It ensures the checkpoint is stored in the
configured model directory and performs a light-weight validation step so CI
pipelines can fail fast when the download is incomplete.

Example
-------
Run the script with an access token (interactive login is also supported)::

    python scripts/download_gemma3.py --token $HF_TOKEN --revision main \
        --target /models/gemma3-12b

The command is idempotent. If the model already exists locally the validation
step still runs, verifying that the critical assets are present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfFolder, snapshot_download

REPO_ID = "docling/gemma3-12b-docling"
EXPECTED_FILES = {
    "config.json",
    "docling-metadata.json",
    "model.safetensors",
    "tokenizer.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Gemma3 12B checkpoint")
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("/models/gemma3-12b"),
        help="Destination directory for the checkpoint",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision or commit hash to download",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face access token (falls back to cached login)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation of expected artifacts",
    )
    return parser.parse_args()


def ensure_login(token: str | None) -> None:
    if token:
        HfFolder.save_token(token)
    elif not HfFolder.get_token():
        raise SystemExit(
            "No Hugging Face token provided. Use --token or run `huggingface-cli login`."
        )


def download(args: argparse.Namespace) -> Path:
    target = args.target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=target,
        local_dir_use_symlinks=False,
        token=HfFolder.get_token(),
    )
    return target


def validate_directory(path: Path) -> None:
    missing = {name for name in EXPECTED_FILES if not (path / name).exists()}
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise SystemExit(f"Gemma3 checkpoint validation failed. Missing: {missing_list}")


def main() -> None:
    args = parse_args()
    ensure_login(args.token)
    destination = download(args)
    if not args.skip_validation:
        validate_directory(destination)
    print(f"Gemma3 checkpoint available at {destination}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI failure path
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
