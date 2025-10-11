"""Chunking wrapper base classes and reference implementations.

Key Responsibilities:
    - Define a common interface used by gateway services to invoke chunking
      backends
    - Provide shared helpers for deriving metadata and selecting sentence
      splitters
    - Offer lightweight reference implementations used during early
      integration tests

Collaborators:
    - Upstream: Gateway GraphQL/REST handlers coordinate chunking through
      these wrappers
    - Downstream: Chunking profiles, sentence splitters, and tokenizers

Side Effects:
    - Emits structured logs on chunking failure or configuration updates

Thread Safety:
    - Instances are not thread-safe; callers should avoid mutating
      configuration concurrently

Performance Characteristics:
    - Chunk creation complexity is proportional to input sentences or tokens
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================

import logging
from abc import ABC, abstractmethod
from typing import Any

import structlog

from Medical_KG_rev.chunking.models import Chunk
from Medical_KG_rev.models.ir import Document
from Medical_KG_rev.services.chunking.config import ChunkingConfig

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logger = structlog.get_logger(__name__)


# ==============================================================================
# BASE INTERFACES
# ==============================================================================


class BaseChunkingWrapper(ABC):
    """Base class for chunking service wrappers.

    Attributes:
        config: Chunking configuration describing available profiles.
        logger: Structured logger for emitting operational events.

    Invariants:
        - ``config`` references a valid ``ChunkingConfig`` after initialisation
    """

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize the chunking wrapper.

        Args:
            config: Loaded chunking configuration containing profile settings.
        """
        self.config = config
        self.logger = logger

    @abstractmethod
    def chunk(
        self, document: Document, *, profile: str
    ) -> list[Chunk]:  # pragma: no cover - defined in subclasses
        """Chunk a document using the specified profile.

        Args:
            document: Source document to be chunked.
            profile: Chunking profile identifier.

        Returns:
            List of generated chunks.
        """
        raise NotImplementedError

    def _metadata_provider(self, document: Document):
        """Create metadata provider for document.

        Args:
            document: Document whose contextual metadata should be included in
                generated chunks.

        Returns:
            Callable producing metadata dictionaries for chunk assembly.
        """
        source_system = document.source

        def provider(contexts):
            metadata: dict[str, str] = {
                "source_system": source_system,
                "document_id": document.id,
                "chunker": self.__class__.__name__,
            }
            return metadata

        return provider

    def _get_sentence_splitter(self, profile: dict[str, Any]):
        """Resolve the configured sentence splitter for a profile.

        Args:
            profile: Chunking profile definition.

        Returns:
            Sentence splitter instance contributed by the wrapper.
        """
        splitter_name = profile.get("sentence_splitter", "syntok")
        return self._create_sentence_splitter(splitter_name)

    def _create_sentence_splitter(self, splitter_name: str):
        """Create sentence splitter instance.

        Args:
            splitter_name: Name of the splitter specified by the profile.

        Raises:
            NotImplementedError: Always raised until a concrete wrapper
                provides the implementation.
        """
        raise NotImplementedError(
            f"Sentence splitter '{splitter_name}' not implemented. "
            "This chunking wrapper requires a real sentence splitter implementation. "
            "Please implement or configure a proper sentence splitter."
        )

    def health_check(self) -> dict[str, Any]:
        """Return a health payload describing available profiles.

        Returns:
            Mapping containing wrapper name, status, and configured profiles.
        """
        return {
            "wrapper": self.__class__.__name__,
            "status": "healthy",
            "config": {
                "profiles": list(self.config.profiles.keys()),
            },
        }

    def get_config(self) -> ChunkingConfig:
        """Return the currently active chunking configuration."""
        return self.config

    def update_config(self, config: ChunkingConfig) -> None:
        """Replace the chunking configuration with a new instance.

        Args:
            config: New chunking configuration to apply.
        """
        self.config = config
        self.logger.info(f"Updated configuration for {self.__class__.__name__}")


# ==============================================================================
# IMPLEMENTATIONS
# ==============================================================================


class SyntokChunkingWrapper(BaseChunkingWrapper):
    """Syntok-based chunking wrapper."""

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:
        """Chunk document using Syntok.

        Args:
            document: Source document awaiting segmentation.
            profile: Chunking profile to apply.

        Returns:
            List of generated chunks.

        Raises:
            ValueError: If the requested profile is not defined.
            Exception: Propagates unexpected processing errors.
        """
        try:
            # Get profile configuration
            profile_config = self.config.profiles.get(profile)
            if not profile_config:
                raise ValueError(f"Profile not found: {profile}")

            # Get sentence splitter
            splitter = self._get_sentence_splitter(profile_config)

            # Split document content
            content = document.content[0] if document.content else ""
            sentences = splitter.split(content)

            # Create chunks
            chunks = []
            chunk_size = profile_config.get("chunk_size", 1000)
            chunk_overlap = profile_config.get("chunk_overlap", 200)

            current_chunk = []
            current_size = 0

            for sentence in sentences:
                sentence_size = len(sentence.split())

                if current_size + sentence_size > chunk_size and current_chunk:
                    # Create chunk
                    chunk_text = " ".join(current_chunk)
                    chunk = Chunk(
                        id=f"chunk-{len(chunks)}",
                        text=chunk_text,
                        metadata={
                            "chunker": "syntok",
                            "profile": profile,
                            "chunk_size": len(chunk_text),
                            "sentence_count": len(current_chunk),
                        },
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    overlap_sentences = int(chunk_overlap * 0.1)  # 10% overlap
                    current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                    current_size = sum(len(s.split()) for s in current_chunk)

                current_chunk.append(sentence)
                current_size += sentence_size

            # Add final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk = Chunk(
                    id=f"chunk-{len(chunks)}",
                    text=chunk_text,
                    metadata={
                        "chunker": "syntok",
                        "profile": profile,
                        "chunk_size": len(chunk_text),
                        "sentence_count": len(current_chunk),
                    },
                )
                chunks.append(chunk)

            return chunks

        except Exception as exc:
            self.logger.error(f"Syntok chunking failed: {exc}")
            raise exc


class HuggingFaceChunkingWrapper(BaseChunkingWrapper):
    """HuggingFace-based chunking wrapper."""

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:
        """Chunk document using HuggingFace tokenizer.

        Args:
            document: Source document awaiting token-based chunking.
            profile: Chunking profile to apply.

        Returns:
            List of generated chunks.

        Raises:
            ValueError: If the requested profile is not defined.
            Exception: Propagates unexpected processing errors.
        """
        try:
            # Get profile configuration
            profile_config = self.config.profiles.get(profile)
            if not profile_config:
                raise ValueError(f"Profile not found: {profile}")

            # Get tokenizer
            tokenizer_name = profile_config.get("tokenizer", "bert-base-uncased")
            tokenizer = self._create_tokenizer(tokenizer_name)

            # Tokenize document content
            content = document.content[0] if document.content else ""
            tokens = tokenizer.tokenize(content)

            # Create chunks
            chunks = []
            chunk_size = profile_config.get("chunk_size", 512)
            chunk_overlap = profile_config.get("chunk_overlap", 50)

            for i in range(0, len(tokens), chunk_size - chunk_overlap):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)

                chunk = Chunk(
                    id=f"chunk-{len(chunks)}",
                    text=chunk_text,
                    metadata={
                        "chunker": "huggingface",
                        "profile": profile,
                        "tokenizer": tokenizer_name,
                        "token_count": len(chunk_tokens),
                        "chunk_size": len(chunk_text),
                    },
                )
                chunks.append(chunk)

            return chunks

        except Exception as exc:
            self.logger.error(f"HuggingFace chunking failed: {exc}")
            raise exc

    def _create_tokenizer(self, tokenizer_name: str):
        """Create tokenizer instance.

        Args:
            tokenizer_name: Hugging Face model identifier.

        Raises:
            NotImplementedError: Always raised until a concrete implementation
                is supplied by a deployment-specific wrapper.
        """
        raise NotImplementedError(
            f"Tokenizer '{tokenizer_name}' not implemented. "
            "This chunking wrapper requires a real tokenizer implementation. "
            "Please implement or configure a proper tokenizer."
        )


class SciSpacyChunkingWrapper(BaseChunkingWrapper):
    """SciSpacy-based chunking wrapper."""

    def chunk(self, document: Document, *, profile: str) -> list[Chunk]:
        """Chunk document using SciSpacy.

        Args:
            document: Source document awaiting SciSpaCy-driven chunking.
            profile: Chunking profile to apply.

        Returns:
            List of generated chunks.

        Raises:
            ValueError: If the requested profile is not defined.
            Exception: Propagates unexpected processing errors.
        """
        try:
            # Get profile configuration
            profile_config = self.config.profiles.get(profile)
            if not profile_config:
                raise ValueError(f"Profile not found: {profile}")

            # Get SciSpacy model
            model_name = profile_config.get("model", "en_core_sci_sm")
            nlp = self._create_spacy_model(model_name)

            # Process document content
            content = document.content[0] if document.content else ""
            doc = nlp(content)

            # Create chunks based on sentences
            chunks = []
            chunk_size = profile_config.get("chunk_size", 1000)

            current_chunk = []
            current_size = 0

            for sent in doc.sents:
                sent_text = sent.text
                sent_size = len(sent_text.split())

                if current_size + sent_size > chunk_size and current_chunk:
                    # Create chunk
                    chunk_text = " ".join(current_chunk)
                    chunk = Chunk(
                        id=f"chunk-{len(chunks)}",
                        text=chunk_text,
                        metadata={
                            "chunker": "scispacy",
                            "profile": profile,
                            "model": model_name,
                            "chunk_size": len(chunk_text),
                            "sentence_count": len(current_chunk),
                        },
                    )
                    chunks.append(chunk)

                    # Start new chunk
                    current_chunk = [sent_text]
                    current_size = sent_size
                else:
                    current_chunk.append(sent_text)
                    current_size += sent_size

            # Add final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk = Chunk(
                    id=f"chunk-{len(chunks)}",
                    text=chunk_text,
                    metadata={
                        "chunker": "scispacy",
                        "profile": profile,
                        "model": model_name,
                        "chunk_size": len(chunk_text),
                        "sentence_count": len(current_chunk),
                    },
                )
                chunks.append(chunk)

            return chunks

        except Exception as exc:
            self.logger.error(f"SciSpacy chunking failed: {exc}")
            raise exc

    def _create_spacy_model(self, model_name: str):
        """Create spaCy model instance.

        Args:
            model_name: Name of the SciSpaCy model to load.

        Raises:
            NotImplementedError: Always raised until a concrete implementation
                is supplied by a deployment-specific wrapper.
        """
        raise NotImplementedError(
            f"spaCy model '{model_name}' not implemented. "
            "This chunking wrapper requires a real spaCy model implementation. "
            "Please implement or configure a proper spaCy model."
        )


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


def create_chunking_wrapper(wrapper_type: str, config: ChunkingConfig) -> BaseChunkingWrapper:
    """Create a chunking wrapper instance.

    Args:
        wrapper_type: Identifier of the wrapper implementation to load.
        config: Chunking configuration shared with the wrapper.

    Returns:
        Concrete ``BaseChunkingWrapper`` implementation.

    Raises:
        ValueError: If ``wrapper_type`` is not recognised.
    """
    if wrapper_type == "syntok":
        return SyntokChunkingWrapper(config)
    elif wrapper_type == "huggingface":
        return HuggingFaceChunkingWrapper(config)
    elif wrapper_type == "scispacy":
        return SciSpacyChunkingWrapper(config)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")


def get_available_wrappers() -> list[str]:
    """Return the list of available wrapper type identifiers."""
    return ["syntok", "huggingface", "scispacy"]
