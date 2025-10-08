"""Registration helpers for built-in embedder adapters."""

from __future__ import annotations

import importlib
import structlog

from .dense.openai_compat import register_openai_compat
from .registry import EmbedderRegistry
from .sparse.splade import register_sparse


logger = structlog.get_logger(__name__)


def _try_import(module_path: str, symbol: str) -> object | None:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, symbol)
    except ModuleNotFoundError:
        logger.debug("embedding.provider.optional_missing", module=module_path, symbol=symbol)
        return None
    except AttributeError:  # pragma: no cover - defensive guard
        logger.warning("embedding.provider.symbol_missing", module=module_path, symbol=symbol)
        return None


def register_builtin_embedders(registry: EmbedderRegistry) -> None:
    """Register embedders that are always available plus optional ones when dependencies exist."""

    register_openai_compat(registry)
    register_sparse(registry)

    optional_modules = [
        ("Medical_KG_rev.embeddings.dense.sentence_transformers", "register_sentence_transformers"),
        ("Medical_KG_rev.embeddings.dense.tei", "register_tei"),
        ("Medical_KG_rev.embeddings.multi_vector.colbert", "register_colbert"),
        ("Medical_KG_rev.embeddings.neural_sparse.opensearch", "register_neural_sparse"),
        ("Medical_KG_rev.embeddings.frameworks.langchain", "register_langchain"),
        ("Medical_KG_rev.embeddings.frameworks.llama_index", "register_llama_index"),
        ("Medical_KG_rev.embeddings.frameworks.haystack", "register_haystack"),
        ("Medical_KG_rev.embeddings.experimental.simlm", "register_simlm"),
        ("Medical_KG_rev.embeddings.experimental.retromae", "register_retromae"),
        ("Medical_KG_rev.embeddings.experimental.gtr", "register_gtr"),
        ("Medical_KG_rev.embeddings.experimental.dsi", "register_dsi"),
    ]

    for module_path, symbol in optional_modules:
        registrar = _try_import(module_path, symbol)
        if registrar is None:
            continue
        try:
            registrar(registry)  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "embedding.provider.registration_failed",
                module=module_path,
                symbol=symbol,
                error=str(exc),
            )


__all__ = ["register_builtin_embedders"]
