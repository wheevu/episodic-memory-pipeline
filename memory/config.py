"""
Configuration management for the memory pipeline.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Central configuration for the memory pipeline."""
    
    # Embedding settings
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "mock")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    local_embedding_url: str = field(
        default_factory=lambda: os.getenv("LOCAL_EMBEDDING_URL", "http://localhost:8080/embed")
    )
    
    # LLM settings
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini")
    )
    local_llm_url: str = field(
        default_factory=lambda: os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
    )
    
    # Storage paths
    database_path: Path = field(
        default_factory=lambda: Path(os.getenv("DATABASE_PATH", "./data/memory.db"))
    )
    faiss_index_path: Path = field(
        default_factory=lambda: Path(os.getenv("FAISS_INDEX_PATH", "./data/memory.faiss"))
    )
    
    # Memory settings
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    )
    min_memory_confidence: float = field(
        default_factory=lambda: float(os.getenv("MIN_MEMORY_CONFIDENCE", "0.6"))
    )
    consolidation_window_days: int = field(
        default_factory=lambda: int(os.getenv("CONSOLIDATION_WINDOW_DAYS", "7"))
    )
    
    def __post_init__(self) -> None:
        """Ensure storage directories exist.

        Returns:
            None.
        """
        self.database_path = Path(self.database_path)
        self.faiss_index_path = Path(self.faiss_index_path)
        
        # Create data directory if needed
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global config instance.

    Returns:
        Global `Config` instance.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset config (useful for testing).

    Returns:
        None.
    """
    global _config
    _config = None

