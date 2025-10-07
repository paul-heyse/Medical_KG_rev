"""Registration helpers for built-in embedder adapters."""

from __future__ import annotations

from .dense.openai_compat import register_openai_compat
from .dense.sentence_transformers import register_sentence_transformers
from .dense.tei import register_tei
from .experimental.dsi import register_dsi
from .experimental.gtr import register_gtr
from .experimental.retromae import register_retromae
from .experimental.simlm import register_simlm
from .frameworks.haystack import register_haystack
from .frameworks.langchain import register_langchain
from .frameworks.llama_index import register_llama_index
from .multi_vector.colbert import register_colbert
from .neural_sparse.opensearch import register_neural_sparse
from .registry import EmbedderRegistry
from .sparse.splade import register_sparse


def register_builtin_embedders(registry: EmbedderRegistry) -> None:
    register_sentence_transformers(registry)
    register_tei(registry)
    register_openai_compat(registry)
    register_colbert(registry)
    register_sparse(registry)
    register_neural_sparse(registry)
    register_langchain(registry)
    register_llama_index(registry)
    register_haystack(registry)
    register_simlm(registry)
    register_retromae(registry)
    register_gtr(registry)
    register_dsi(registry)
