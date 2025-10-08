"""Verify the optional parsing/chunking dependency stack is available."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Iterable


def _check_modules(modules: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:  # pragma: no cover - exercised in deployment checks
            missing.append(f"{module}: {exc}")
    return missing


def _check_huggingface_model() -> list[str]:
    model_name = os.getenv("MEDICAL_KG_SENTENCE_MODEL")
    if not model_name:
        return [
            "MEDICAL_KG_SENTENCE_MODEL is not configured; the Hugging Face "
            "sentence segmenter will fall back to heuristic splitting.",
        ]

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - handled by module checks
        return [f"transformers: {exc}"]

    try:
        AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=True,
        )
    except OSError as exc:
        return [
            f"Hugging Face tokenizer '{model_name}' not found locally: {exc}. "
            "Download the model prior to deployment.",
        ]
    except Exception as exc:  # pragma: no cover - unexpected failure
        return [f"Hugging Face tokenizer '{model_name}': {exc}"]
    return []


def main() -> int:
    modules = [
        "langchain_text_splitters",
        "llama_index.core",
        "syntok.segmenter",
        "unstructured.partition",
        "tiktoken",
        "transformers",
        "pydantic",
    ]

    missing = _check_modules(modules)
    missing.extend(_check_huggingface_model())

    if missing:
        print("Chunking dependency check failed:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print("All chunking dependencies resolved successfully.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
