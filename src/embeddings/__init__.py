"""Embedding abstraction layer."""

from .interface import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    MockEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "get_embedding_provider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "MockEmbeddingProvider",
]
