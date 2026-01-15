"""
Bootstrap module for the episodic memory pipeline.

=============================================================================
CRITICAL: INITIALIZATION ORDER FOR NATIVE LIBRARIES
=============================================================================

This module exists to solve a macOS segfault issue when using SentenceTransformers
(which uses PyTorch + HuggingFace tokenizers) alongside FAISS (native C++ library).

THE PROBLEM:
- SentenceTransformers loads PyTorch and tokenizers (with parallel workers)
- FAISS initializes its own native code with memory management
- When Python exits, the cleanup order of these libraries can conflict
- On macOS specifically, this manifests as SIGSEGV (segfault) during atexit

THE SOLUTION:
- Set TOKENIZERS_PARALLELISM=false BEFORE any imports
- Initialize SentenceTransformers models BEFORE importing FAISS-related code
- Cache the preloaded model for reuse throughout the application lifecycle

This module provides a single entry point (get_components) that:
1. Sets safety environment flags
2. Preloads embedding models (if using local embeddings)
3. Imports FAISS-related modules in the correct order
4. Returns fully initialized pipeline components

All CLI and application code should use this module instead of importing
pipeline components directly.

=============================================================================
"""

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# =============================================================================
# STEP 1: SET ENVIRONMENT FLAGS BEFORE ANY HEAVY IMPORTS
# =============================================================================
# This MUST happen before importing anything that might load tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =============================================================================
# STEP 2: IMPORT CONFIG AND EMBEDDING PROVIDER (LIGHTWEIGHT)
# =============================================================================
# These imports are safe - they don't load FAISS
from config import Config
from config import config as default_config
from src.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    get_embedding_provider,
)
from src.llm import LLMProvider, get_llm_provider

# =============================================================================
# MODULE-LEVEL CACHE FOR PRELOADED MODELS
# =============================================================================
_preloaded_embedding_model: Optional[LocalEmbeddingProvider] = None
_bootstrap_initialized: bool = False


@dataclass
class PipelineComponents:
    """Container for all initialized pipeline components."""

    database: Any  # Database
    vector_store: Any  # VectorStore
    embedding_provider: EmbeddingProvider
    llm: LLMProvider

    # Pipeline classes (for lazy instantiation)
    IngestionPipeline: type
    ConsolidationPipeline: type
    RetrievalEngine: type
    EvaluationRunner: type
    get_scenario: callable


def _preload_local_embeddings(config: Config) -> Optional[LocalEmbeddingProvider]:
    """Preload local embedding model BEFORE FAISS imports.

    This is the critical step that prevents segfaults on macOS. The model is cached
    globally so it only loads once.

    Args:
        config: Pipeline configuration.

    Returns:
        The preloaded `LocalEmbeddingProvider` if local embeddings are selected;
        otherwise None.
    """
    global _preloaded_embedding_model

    if _preloaded_embedding_model is not None:
        return _preloaded_embedding_model

    if config.embedding_provider == "local":
        # Load the model NOW, before any FAISS imports
        _preloaded_embedding_model = LocalEmbeddingProvider(
            model_name=config.embedding_model, device=config.embedding_device
        )
        return _preloaded_embedding_model

    return None


def _import_faiss_modules() -> Tuple[Any, ...]:
    """Import FAISS-related modules AFTER embedding model is loaded.

    These imports trigger FAISS native library initialization. They MUST happen after
    SentenceTransformers is loaded to avoid cleanup conflicts on macOS.

    Returns:
        A tuple of imported classes/modules in the order expected by `get_components`.
    """
    # These imports load FAISS
    from src.consolidation import ConsolidationPipeline
    from src.evaluation import EvaluationRunner
    from src.evaluation.runner import get_scenario
    from src.ingestion import IngestionPipeline
    from src.retrieval import RetrievalEngine
    from src.storage import Database, VectorStore

    return (
        Database,
        VectorStore,
        IngestionPipeline,
        ConsolidationPipeline,
        RetrievalEngine,
        EvaluationRunner,
        get_scenario,
    )


def get_components(
    config: Optional[Config] = None, force_mock: bool = False, verbose: bool = True
) -> PipelineComponents:
    """
    Initialize and return all pipeline components with correct import ordering.

    This is the main entry point for obtaining pipeline components.
    It handles the FAISS/SentenceTransformers initialization order automatically.

    Args:
        config: Configuration object. Uses default config if not provided.
        force_mock: If True, use mock providers regardless of config.
        verbose: If True, print status messages about provider selection.

    Returns:
        PipelineComponents containing all initialized components.

    Example:
        >>> from src.bootstrap import get_components
        >>> components = get_components()
        >>> result = components.IngestionPipeline(
        ...     components.database,
        ...     components.vector_store,
        ...     components.embedding_provider,
        ...     components.llm
        ... ).ingest("Some text")
    """
    global _bootstrap_initialized

    if config is None:
        config = default_config

    # ==========================================================================
    # STEP 3: PRELOAD EMBEDDING MODEL (if using local embeddings)
    # ==========================================================================
    # This MUST happen before importing FAISS modules
    preloaded_model = None
    if not force_mock and config.embedding_provider == "local":
        preloaded_model = _preload_local_embeddings(config)

    # ==========================================================================
    # STEP 4: NOW SAFE TO IMPORT FAISS-RELATED MODULES
    # ==========================================================================
    (
        Database,
        VectorStore,
        IngestionPipeline,
        ConsolidationPipeline,
        RetrievalEngine,
        EvaluationRunner,
        get_scenario,
    ) = _import_faiss_modules()

    _bootstrap_initialized = True

    # ==========================================================================
    # STEP 5: CREATE STORAGE COMPONENTS
    # ==========================================================================
    database = Database(config.database_path)
    vector_store = VectorStore(config.vector_index_path, dimension=config.embedding_dimension)

    # ==========================================================================
    # STEP 6: CREATE EMBEDDING PROVIDER
    # ==========================================================================
    embedding_provider = _create_embedding_provider(config, force_mock, preloaded_model, verbose)

    # ==========================================================================
    # STEP 7: CREATE LLM PROVIDER
    # ==========================================================================
    llm = _create_llm_provider(config, force_mock, verbose)

    return PipelineComponents(
        database=database,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        llm=llm,
        IngestionPipeline=IngestionPipeline,
        ConsolidationPipeline=ConsolidationPipeline,
        RetrievalEngine=RetrievalEngine,
        EvaluationRunner=EvaluationRunner,
        get_scenario=get_scenario,
    )


def _create_embedding_provider(
    config: Config,
    force_mock: bool,
    preloaded_model: Optional[LocalEmbeddingProvider],
    verbose: bool,
) -> EmbeddingProvider:
    """Create the appropriate embedding provider based on config.

    Args:
        config: Pipeline configuration.
        force_mock: If True, force mock embeddings regardless of config.
        preloaded_model: Preloaded local model instance, if available.
        verbose: If True, print status messages.

    Returns:
        An `EmbeddingProvider` instance.

    Raises:
        ValueError: If OpenAI embeddings are selected but `OPENAI_API_KEY` is missing.
    """

    if force_mock or config.embedding_provider == "mock":
        if verbose:
            _log(
                "[yellow]Using mock embedding provider (retrieval metrics will be meaningless)[/yellow]"
            )
        return get_embedding_provider("mock", dimension=config.embedding_dimension)

    if config.embedding_provider == "openai":
        if not config.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY required for OpenAI embeddings. "
                "Use EMBEDDING_PROVIDER=local for local embeddings."
            )
        if verbose:
            _log(f"[cyan]Using OpenAI embeddings: {config.embedding_model}[/cyan]")
        return get_embedding_provider(
            "openai",
            api_key=config.openai_api_key,
            model=config.embedding_model,
            dimension=config.embedding_dimension,
        )

    if config.embedding_provider == "ollama":
        if verbose:
            _log(f"[cyan]Using Ollama embeddings: {config.ollama_embed_model}[/cyan]")
        return get_embedding_provider(
            "ollama", model=config.ollama_embed_model, base_url=config.ollama_base_url
        )

    # Default: local embeddings - use preloaded model if available
    if verbose:
        _log(
            f"[green]Using local embeddings: {config.embedding_model} on {config.embedding_device}[/green]"
        )

    if preloaded_model is not None:
        return preloaded_model

    return get_embedding_provider(
        "local", model=config.embedding_model, device=config.embedding_device
    )


def _create_llm_provider(config: Config, force_mock: bool, verbose: bool) -> LLMProvider:
    """Create the appropriate LLM provider based on config.

    Args:
        config: Pipeline configuration.
        force_mock: If True, force mock LLM regardless of config.
        verbose: If True, print status messages.

    Returns:
        An `LLMProvider` instance.
    """

    if force_mock:
        if verbose:
            _log("[yellow]Using mock LLM provider[/yellow]")
        return get_llm_provider("mock")

    if config.llm_provider == "ollama":
        if verbose:
            _log(f"[cyan]Using Ollama LLM: {config.ollama_model}[/cyan]")
        return get_llm_provider(
            "ollama",
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=config.llm_temperature,
        )

    if config.openai_api_key:
        if verbose:
            _log(f"[cyan]Using OpenAI LLM: {config.llm_model}[/cyan]")
        return get_llm_provider("openai", api_key=config.openai_api_key, model=config.llm_model)

    # Fallback to mock
    if verbose:
        _log("[yellow]Using mock LLM provider (no API key configured)[/yellow]")
    return get_llm_provider("mock")


def _log(message: str) -> None:
    """Print a message using Rich if available, else plain print.

    Args:
        message: Rich markup string (or plain text) to print.

    Returns:
        None.
    """
    try:
        from rich.console import Console

        console = Console()
        console.print(message)
    except ImportError:
        # Strip rich markup for plain print
        import re

        plain = re.sub(r"\[/?[^\]]+\]", "", message)
        print(plain)


def is_initialized() -> bool:
    """Check if bootstrap has been run.

    Returns:
        True if `get_components()` has been invoked in this process; otherwise False.
    """
    return _bootstrap_initialized


def get_cached_embedding_model() -> Optional[LocalEmbeddingProvider]:
    """Get the cached preloaded embedding model, if any.

    Returns:
        Cached `LocalEmbeddingProvider` instance if preloaded; otherwise None.
    """
    return _preloaded_embedding_model
