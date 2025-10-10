"""Utility script to pre-download embedding models for offline environments."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download embedding models")
    parser.add_argument("--models", nargs="+", required=True, help="Model IDs to download")
    parser.add_argument(
        "--cache-dir",
        default=Path.home() / ".cache" / "medical_kg_rev" / "models",
        help="Target directory for cached models",
    )
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        snapshot_download(
            repo_id=model,
            local_dir=cache_dir / model.replace("/", "__"),
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded {model} to {cache_dir}")


if __name__ == "__main__":
    main()
