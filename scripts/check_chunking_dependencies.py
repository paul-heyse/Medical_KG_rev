#!/usr/bin/env python
"""Lightweight sanity check for chunking-related dependencies.

This helper is intended for deployment pipelines so that engineers can verify
that the optional libraries required by the new chunking stack are available in
the runtime environment.  It validates both Python packages and, where
applicable, required resources (such as the scispaCy language model).
"""

from __future__ import annotations

import importlib
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


def _check_scispacy_model() -> list[str]:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        return [f"spacy: {exc}"]

    model_name = "en_core_sci_sm"
    try:
        spacy.load(model_name)
    except Exception as exc:  # pragma: no cover - executed if model missing
        return [f"spacy model '{model_name}': {exc}"]
    return []


def main() -> int:
    modules = [
        "langchain_text_splitters",
        "llama_index.core",
        "scispacy",
        "syntok.segmenter",
        "unstructured.partition",
        "tiktoken",
        "transformers",
    ]

    missing = _check_modules(modules)
    missing.extend(_check_scispacy_model())

    if missing:
        print("Chunking dependency check failed:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print("All chunking dependencies resolved successfully.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
