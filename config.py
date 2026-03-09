"""
Configuration management for the episodic memory pipeline.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Central configuration for the memory pipeline."""

    # Paths
    base_path: Path = field(default_factory=lambda: Path(__file__).parent)
    lance_db_path: Optional[Path] = field(default=None)

    # Embedding configuration (default: local with BGE-M3)
    embedding_provider: str = "local"  # "local", "openai", "ollama", or "mock"
    embedding_model: str = "BAAI/bge-m3"  # Default local model
    embedding_dimension: int = 1024  # BGE-M3 dimension

    # Ollama embeddings (alternative local embedding runtime)
    ollama_embed_model: str = "nomic-embed-text"  # For EMBEDDING_PROVIDER=ollama

    # LLM configuration
    llm_provider: str = "openai"  # "openai" or "ollama"
    llm_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"  # Default local model
    llm_temperature: float = 0.2  # Low for determinism

    # API keys (from environment)
    openai_api_key: Optional[str] = field(default=None)

    # Memory pipeline settings
    memory_worthiness_threshold: float = 0.6  # minimum score to store
    consolidation_episode_threshold: int = 5  # episodes before consolidation
    consolidation_age_days: int = 7  # consolidate weekly
    max_episodes_per_summary: int = 20

    # Retrieval settings
    semantic_top_k: int = 10
    narrative_max_episodes: int = 50
    similarity_threshold: float = 0.7

    def __post_init__(self) -> None:
        """Finalize configuration by applying defaults and environment overrides.

        This runs after dataclass initialization to:
        - Fill in default filesystem paths.
        - Read configuration overrides from environment variables.
        - Derive embedding dimensions for known models/providers (when not explicitly set).

        Note: Directory creation is deferred until first access via ensure_directories().

        Returns:
            None.
        """
        # Set default paths relative to base
        if self.lance_db_path is None:
            self.lance_db_path = self.base_path / "data" / "lancedb"

        # Load from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", self.embedding_provider)
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)

        # Embedding-specific configuration
        if embed_model := os.getenv("EMBEDDING_MODEL"):
            self.embedding_model = embed_model
        if embed_dim := os.getenv("EMBEDDING_DIMENSION"):
            self.embedding_dimension = int(embed_dim)
        if ollama_embed := os.getenv("OLLAMA_EMBED_MODEL"):
            self.ollama_embed_model = ollama_embed

        # LLM model override
        if llm_model := os.getenv("LLM_MODEL"):
            self.llm_model = llm_model

        # Ollama-specific configuration
        if ollama_model := os.getenv("OLLAMA_MODEL"):
            self.ollama_model = ollama_model
        if ollama_url := os.getenv("OLLAMA_BASE_URL"):
            self.ollama_base_url = ollama_url
        if llm_temp := os.getenv("LLM_TEMPERATURE"):
            self.llm_temperature = float(llm_temp)

        if memory_threshold := os.getenv("MEMORY_WORTHINESS_THRESHOLD"):
            self.memory_worthiness_threshold = float(memory_threshold)
        if consolidation_episode_threshold := os.getenv("CONSOLIDATION_EPISODE_THRESHOLD"):
            self.consolidation_episode_threshold = int(consolidation_episode_threshold)
        if consolidation_age_days := os.getenv("CONSOLIDATION_AGE_DAYS"):
            self.consolidation_age_days = int(consolidation_age_days)

        if lance_path := os.getenv("LANCE_DB_PATH"):
            self.lance_db_path = Path(lance_path)

        # Update embedding dimension based on provider/model
        self._configure_embedding_dimension()

    def _configure_embedding_dimension(self) -> None:
        """Set embedding dimension based on provider and model.

        Returns:
            None.
        """
        # Only update if not explicitly set via environment
        if os.getenv("EMBEDDING_DIMENSION"):
            return

        # Known model dimensions
        model_dimensions = {
            # FastEmbed-compatible local models
            "BAAI/bge-m3": 1024,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            # OpenAI models
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            # Ollama models
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
        }

        if self.embedding_provider == "local":
            self.embedding_dimension = model_dimensions.get(self.embedding_model, 1024)
        elif self.embedding_provider == "openai":
            self.embedding_dimension = model_dimensions.get(self.embedding_model, 1536)
        elif self.embedding_provider == "ollama":
            self.embedding_dimension = model_dimensions.get(self.ollama_embed_model, 768)
        elif self.embedding_provider == "mock":
            self.embedding_dimension = 1024  # Match BGE-M3 for consistency

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist.

        This is called lazily by components that need to write to disk,
        rather than eagerly during config initialization.

        Returns:
            None.
        """
        try:
            self.lance_db_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", self.lance_db_path)
        except Exception as e:
            logger.warning("Failed to create directory %s: %s", self.lance_db_path, e)
            raise


# Global config instance
config = Config()
