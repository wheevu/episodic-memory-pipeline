"""
Embedding provider abstraction.

This module provides a unified interface for generating embeddings,
supporting multiple backends (OpenAI, local sentence-transformers, Ollama).

Default Configuration:
- Provider: local (SentenceTransformers with BAAI/bge-m3)
- Device: auto-detect (CPU, MPS on Mac, CUDA if available)
- All embeddings are L2-normalized for cosine similarity via inner product
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def _normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors so inner product corresponds to cosine similarity.

    Args:
        vectors: Vector array shaped (d,) or (n, d).

    Returns:
        A normalized array with the same shape as `vectors`.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / norm if norm > 0 else vectors
    
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            The embedding vector dimensionality.
        """
        pass
    
    @property
    def provider_name(self) -> str:
        """Return provider name for logging/debugging.

        Returns:
            A human-readable provider identifier.
        """
        return self.__class__.__name__
    
    @property
    def is_mock(self) -> bool:
        """Return True if this is a mock provider.

        Returns:
            True if the provider is a mock; otherwise False.
        """
        return False
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            L2-normalized embedding vector as a `np.float32` array.
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            L2-normalized 2D numpy array of shape `(len(texts), dimension)`,
            dtype `np.float32`.
        """
        pass
    
    # Aliases for common naming conventions
    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        """Alias for `embed_batch` that returns a list of arrays.

        Args:
            texts: Documents to embed.

        Returns:
            A list of per-document embedding arrays.
        """
        embeddings = self.embed_batch(texts)
        return [embeddings[i] for i in range(embeddings.shape[0])]
    
    def embed_query(self, text: str) -> np.ndarray:
        """Alias for `embed_text`.

        Args:
            text: Query text to embed.

        Returns:
            The embedding vector for the query.
        """
        return self.embed_text(text)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536
    ) -> None:
        """Initialize the OpenAI embedding provider.

        Args:
            api_key: OpenAI API key.
            model: Model name (e.g., "text-embedding-3-small").
            dimension: Output dimension.

        Raises:
            ImportError: If the `openai` package is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension produced by this provider.

        Returns:
            The embedding vector dimensionality.
        """
        return self._dimension
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding using the OpenAI embeddings API.

        Args:
            text: Input text to embed.

        Returns:
            L2-normalized embedding vector.
        """
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimension
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return _normalize_l2(embedding)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a batch using the OpenAI embeddings API.

        Args:
            texts: Input texts to embed.

        Returns:
            L2-normalized embedding matrix of shape `(len(texts), dimension)`.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimension
        )
        
        embeddings = [item.embedding for item in response.data]
        embeddings = np.array(embeddings, dtype=np.float32)
        return _normalize_l2(embeddings)


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    
    Default model: BAAI/bge-m3 (excellent multilingual embeddings)
    
    Features:
    - Automatic device selection (CPU, MPS, CUDA)
    - L2-normalized output for cosine similarity
    - Deterministic embeddings
    """
    
    # Default model for high-quality semantic search
    DEFAULT_MODEL = "BAAI/bge-m3"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True
    ) -> None:
        """Initialize the local sentence-transformers embedding provider.

        Args:
            model_name: Sentence-transformers model name (default: BAAI/bge-m3).
            device: Device to run on ("cpu", "cuda", "mps", or None for auto).
            normalize: Whether to L2-normalize embeddings.

        Raises:
            ImportError: If `sentence-transformers` is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package required. "
                "Install with: pip install sentence-transformers"
            )
        
        # Use environment variable or default
        model_name = model_name or os.getenv("EMBEDDING_MODEL", self.DEFAULT_MODEL)
        device = device or os.getenv("EMBEDDING_DEVICE", None)
        
        # Auto-detect device if not specified
        if device is None:
            device = self._detect_device()
        
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        
        logger.info(f"Loading embedding model '{model_name}' on device '{device}'")
        
        # Load model with specified device
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding model loaded: dimension={self._dimension}")
    
    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available compute device.

        Returns:
            "cuda" if available, else "mps" (Apple Silicon), else "cpu".
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension produced by the loaded model.

        Returns:
            The embedding vector dimensionality.
        """
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Return a descriptive provider name including the model identifier.

        Returns:
            A descriptive provider name including the model identifier.
        """
        return f"LocalEmbedding({self._model_name})"
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding using the local model.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as `np.float32`.
        """
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize
        )
        embedding = embedding.astype(np.float32)
        
        # Ensure normalized even if model doesn't do it
        if self._normalize:
            embedding = _normalize_l2(embedding)
        
        return embedding
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a batch using the local model.

        Args:
            texts: Input texts to embed.

        Returns:
            Embedding matrix as `np.float32` of shape `(len(texts), dimension)`.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=len(texts) > 10
        )
        embeddings = embeddings.astype(np.float32)
        
        # Ensure normalized even if model doesn't do it
        if self._normalize:
            embeddings = _normalize_l2(embeddings)
        
        return embeddings


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama embedding provider for local embedding models.
    
    Uses Ollama's embeddings API endpoint. Supports any embedding model
    available in Ollama (e.g., nomic-embed-text, mxbai-embed-large).
    """
    
    DEFAULT_MODEL = "nomic-embed-text"
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None
    ) -> None:
        """Initialize the Ollama embedding provider.

        Args:
            model: Ollama embedding model name (default: nomic-embed-text).
            base_url: Ollama server URL (default: http://localhost:11434).
            dimension: Expected embedding dimension (auto-detected if not specified).
        """
        self._model = model or os.getenv("OLLAMA_EMBED_MODEL", self.DEFAULT_MODEL)
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self._dimension = dimension
        
        # Auto-detect dimension by embedding a test string
        if self._dimension is None:
            self._dimension = self._detect_dimension()
        
        logger.info(f"Ollama embeddings: model={self._model}, dimension={self._dimension}")
    
    def _detect_dimension(self) -> int:
        """Detect embedding dimension by running a test embedding request.

        Returns:
            The detected embedding vector length, or a conservative default on failure.
        """
        import httpx
        
        try:
            response = httpx.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": "test"},
                timeout=30.0
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return len(embedding)
        except Exception as e:
            logger.warning(f"Could not detect Ollama embedding dimension: {e}")
            # Default dimension for nomic-embed-text
            return 768
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension produced by the Ollama model.

        Returns:
            The embedding vector dimensionality.
        """
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Return a descriptive provider name including the model identifier.

        Returns:
            A descriptive provider name including the model identifier.
        """
        return f"OllamaEmbedding({self._model})"
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding using Ollama's embeddings endpoint.

        Args:
            text: Input text to embed.

        Returns:
            L2-normalized embedding vector.
        """
        import httpx
        
        response = httpx.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=30.0
        )
        response.raise_for_status()
        
        embedding = np.array(response.json()["embedding"], dtype=np.float32)
        return _normalize_l2(embedding)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a batch by calling Ollama per item.

        Args:
            texts: Input texts to embed.

        Returns:
            L2-normalized embedding matrix.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        
        # Ollama doesn't have batch endpoint, so embed one by one
        embeddings = [self.embed_text(text) for text in texts]
        return np.vstack(embeddings)


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing without external dependencies.
    
    Generates deterministic random embeddings based on text hash.
    WARNING: These embeddings have NO semantic meaning - similar texts
    will NOT have similar embeddings. Use only for testing data flow.
    """
    
    def __init__(self, dimension: int = 1024) -> None:
        """Initialize the mock embedding provider.

        Args:
            dimension: Embedding dimension (default: 1024 to match BGE-M3).
        """
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension produced by this mock provider.

        Returns:
            The embedding vector dimensionality.
        """
        return self._dimension
    
    @property
    def is_mock(self) -> bool:
        """Return True for the mock provider.

        Returns:
            True.
        """
        return True
    
    @property
    def provider_name(self) -> str:
        """Return a stable provider name for logging/debugging.

        Returns:
            A stable provider name.
        """
        return "MockEmbedding"
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a deterministic mock embedding.

        Args:
            text: Input text to "embed".

        Returns:
            A normalized random vector derived from the text hash.
        """
        # Use hash to generate reproducible "embeddings"
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self._dimension).astype(np.float32)
        # Normalize to unit length
        return _normalize_l2(embedding)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate mock embeddings for a batch.

        Args:
            texts: Input texts to embed.

        Returns:
            A matrix of embeddings stacked row-wise.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        return np.vstack([self.embed_text(t) for t in texts])


def get_embedding_provider(
    provider: str = "local",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    dimension: Optional[int] = None,
    device: Optional[str] = None,
    base_url: Optional[str] = None,
) -> EmbeddingProvider:
    """Factory to create an `EmbeddingProvider` implementation.

    Args:
        provider: Provider name: "local" (default), "openai", "ollama", or "mock".
        api_key: API key (required for OpenAI).
        model: Model name (optional; uses provider defaults).
        dimension: Embedding dimension (optional; auto-detected for most providers).
        device: Device for local provider ("cpu", "cuda", "mps", or None for auto).
        base_url: Base URL for Ollama provider.

    Returns:
        A configured `EmbeddingProvider` instance.

    Raises:
        ValueError: If the provider is unknown or required configuration is missing.

    Environment variables:
        EMBEDDING_PROVIDER: Default provider selection.
        EMBEDDING_MODEL: Default model name.
        EMBEDDING_DEVICE: Device for local embeddings.
        OLLAMA_BASE_URL: Ollama server URL.
        OLLAMA_EMBED_MODEL: Ollama embedding model.
    """
    # Get provider from environment if not explicitly set
    provider = os.getenv("EMBEDDING_PROVIDER", provider)
    
    if provider == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI provider")
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            dimension=dimension or 1536
        )
    elif provider == "local":
        return LocalEmbeddingProvider(
            model_name=model,  # Will use default if None
            device=device,
            normalize=True
        )
    elif provider == "ollama":
        return OllamaEmbeddingProvider(
            model=model,
            base_url=base_url,
            dimension=dimension
        )
    elif provider == "mock":
        return MockEmbeddingProvider(dimension=dimension or 1024)
    else:
        raise ValueError(f"Unknown provider: {provider}")

