"""
Diagnostics module for the episodic memory pipeline.

This module provides helper functions for the `doctor` command, including:
- Provider inspection and description
- Configuration analysis
- Fix suggestion generation
- Dry-run diagnostics (config-only, no initialization)

The module is designed to be imported by cli.py and provides both
full diagnostics (with initialized components) and dry-run diagnostics
(config inspection only, zero side effects).
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional

from config import Config


@dataclass
class ProviderInfo:
    """Describes a provider's configuration."""

    type: str
    model: str
    temperature: str = "N/A"
    device: str = "N/A"
    dimension: int = 0
    normalized: bool = True
    is_mock: bool = False
    base_url: Optional[str] = None


@dataclass
class ConfigDiagnostics:
    """
    Configuration diagnostics from environment and config defaults.

    This is used for dry-run mode where we don't initialize any components.
    """

    # Environment variables
    env_embedding_provider: Optional[str]
    env_embedding_model: Optional[str]
    env_embedding_dimension: Optional[str]
    env_llm_provider: Optional[str]
    env_ollama_model: Optional[str]
    env_ollama_base_url: Optional[str]
    env_openai_api_key_set: bool
    env_tokenizers_parallelism: Optional[str]

    # Resolved config values
    resolved_embedding_provider: str
    resolved_embedding_model: str
    resolved_embedding_dimension: int
    resolved_llm_provider: str
    resolved_llm_model: str
    resolved_llm_temperature: float

    # Derived flags
    will_use_mock_embeddings: bool
    will_use_mock_llm: bool

    @property
    def has_issues(self) -> bool:
        """Return True if there are any configuration issues.

        Returns:
            True if mock embeddings or mock LLM would be used; otherwise False.
        """
        return self.will_use_mock_embeddings or self.will_use_mock_llm


def get_config_diagnostics(config: Config, force_mock: bool = False) -> ConfigDiagnostics:
    """
    Inspect configuration WITHOUT initializing any components.

    This is the core function for dry-run mode. It reads environment
    variables and config defaults to determine what providers WOULD
    be selected, without actually loading models or connecting to services.

    Args:
        config: The Config object to inspect
        force_mock: Whether --mock flag was passed

    Returns:
        ConfigDiagnostics with all inspection results
    """
    # Read raw environment variables
    env_embedding_provider = os.getenv("EMBEDDING_PROVIDER")
    env_embedding_model = os.getenv("EMBEDDING_MODEL")
    env_embedding_dimension = os.getenv("EMBEDDING_DIMENSION")
    env_llm_provider = os.getenv("LLM_PROVIDER")
    env_ollama_model = os.getenv("OLLAMA_MODEL")
    env_ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    env_openai_api_key = os.getenv("OPENAI_API_KEY")
    env_tokenizers_parallelism = os.getenv("TOKENIZERS_PARALLELISM")

    # Determine resolved values (what config actually uses)
    resolved_embedding_provider = config.embedding_provider
    resolved_embedding_model = config.embedding_model
    resolved_embedding_dimension = config.embedding_dimension
    resolved_llm_provider = config.llm_provider
    resolved_llm_temperature = config.llm_temperature

    # Determine LLM model based on provider
    if config.llm_provider == "ollama":
        resolved_llm_model = config.ollama_model
    else:
        resolved_llm_model = config.llm_model

    # Determine if mock providers will be used
    will_use_mock_embeddings = force_mock or resolved_embedding_provider == "mock"

    will_use_mock_llm = force_mock
    if not force_mock:
        # LLM falls back to mock if no API key and not using ollama
        if config.llm_provider == "ollama":
            will_use_mock_llm = False
        elif config.openai_api_key:
            will_use_mock_llm = False
        else:
            will_use_mock_llm = True

    return ConfigDiagnostics(
        env_embedding_provider=env_embedding_provider,
        env_embedding_model=env_embedding_model,
        env_embedding_dimension=env_embedding_dimension,
        env_llm_provider=env_llm_provider,
        env_ollama_model=env_ollama_model,
        env_ollama_base_url=env_ollama_base_url,
        env_openai_api_key_set=bool(env_openai_api_key),
        env_tokenizers_parallelism=env_tokenizers_parallelism,
        resolved_embedding_provider=resolved_embedding_provider,
        resolved_embedding_model=resolved_embedding_model,
        resolved_embedding_dimension=resolved_embedding_dimension,
        resolved_llm_provider=resolved_llm_provider,
        resolved_llm_model=resolved_llm_model,
        resolved_llm_temperature=resolved_llm_temperature,
        will_use_mock_embeddings=will_use_mock_embeddings,
        will_use_mock_llm=will_use_mock_llm,
    )


def describe_llm_provider(llm: Any, config: Config) -> ProviderInfo:
    """
    Extract descriptive info from an initialized LLM provider.

    Args:
        llm: The LLM provider instance
        config: Config object for fallback values

    Returns:
        ProviderInfo describing the LLM provider
    """
    from src.llm.interface import MockLLMProvider, OllamaLLMProvider, OpenAILLMProvider

    info = ProviderInfo(
        type="UNKNOWN",
        model="UNKNOWN",
        temperature="UNKNOWN",
        is_mock=llm.is_mock,
    )

    if isinstance(llm, MockLLMProvider):
        info.type = "mock"
        info.model = "MockLLM (deterministic)"
        info.temperature = "N/A"
    elif isinstance(llm, OllamaLLMProvider):
        info.type = "ollama"
        info.model = getattr(llm, "_model", config.ollama_model)
        info.temperature = str(getattr(llm, "_default_temperature", config.llm_temperature))
        info.base_url = getattr(llm, "_base_url", config.ollama_base_url)
    elif isinstance(llm, OpenAILLMProvider):
        info.type = "openai"
        info.model = getattr(llm, "_model", config.llm_model)
        info.temperature = str(config.llm_temperature)
        info.base_url = "https://api.openai.com"
    else:
        info.type = llm.__class__.__name__

    return info


def describe_embedding_provider(emb: Any, config: Config) -> ProviderInfo:
    """
    Extract descriptive info from an initialized embedding provider.

    Args:
        emb: The embedding provider instance
        config: Config object for fallback values

    Returns:
        ProviderInfo describing the embedding provider
    """
    from src.embeddings.interface import (
        LocalEmbeddingProvider,
        MockEmbeddingProvider,
        OllamaEmbeddingProvider,
    )

    # Try to import OpenAI provider (may not exist)
    OpenAIEmbeddingProvider = None
    try:
        from src.embeddings.interface import OpenAIEmbeddingProvider
    except ImportError:
        pass

    info = ProviderInfo(
        type="UNKNOWN",
        model="UNKNOWN",
        device="N/A",
        dimension=emb.dimension,
        normalized=True,  # All our providers normalize
        is_mock=emb.is_mock,
    )

    if isinstance(emb, MockEmbeddingProvider):
        info.type = "mock"
        info.model = "MockEmbedding (random normalized)"
    elif isinstance(emb, LocalEmbeddingProvider):
        info.type = "local"
        model_attr = getattr(emb, "_model", None)
        if model_attr is None:
            info.model = config.embedding_model
        elif hasattr(model_attr, "name_or_path"):
            info.model = model_attr.name_or_path or config.embedding_model
        else:
            info.model = config.embedding_model
        info.device = "N/A"
    elif isinstance(emb, OllamaEmbeddingProvider):
        info.type = "ollama"
        info.model = getattr(emb, "_model", config.ollama_embed_model)
        info.device = "remote (Ollama server)"
    elif OpenAIEmbeddingProvider is not None and isinstance(emb, OpenAIEmbeddingProvider):
        info.type = "openai"
        info.model = config.embedding_model
        info.device = "remote (OpenAI API)"
    else:
        info.type = emb.__class__.__name__

    return info


def generate_fix_suggestions(
    config_diag: Optional[ConfigDiagnostics] = None,
    llm_info: Optional[ProviderInfo] = None,
    emb_info: Optional[ProviderInfo] = None,
    force_mock: bool = False,
) -> List[str]:
    """
    Generate copy-pasteable shell commands to fix configuration issues.

    This function analyzes either:
    - ConfigDiagnostics (dry-run mode, config inspection only)
    - ProviderInfo from initialized providers (full mode)

    Args:
        config_diag: Config diagnostics from dry-run inspection
        llm_info: LLM provider info (from initialized provider)
        emb_info: Embedding provider info (from initialized provider)
        force_mock: Whether --mock flag was explicitly passed

    Returns:
        List of suggestion strings (may include shell commands)
    """
    suggestions = []

    # If --mock was explicitly used, no fixes needed
    if force_mock:
        return [
            "# Mock mode explicitly enabled via --mock flag",
            "# No action needed (intentional mock providers)",
        ]

    # Determine mock status from either source
    is_mock_embeddings = False
    is_mock_llm = False

    if config_diag:
        is_mock_embeddings = config_diag.will_use_mock_embeddings
        is_mock_llm = config_diag.will_use_mock_llm
    elif emb_info and llm_info:
        is_mock_embeddings = emb_info.is_mock
        is_mock_llm = llm_info.is_mock

    # Generate embedding fix suggestions
    if is_mock_embeddings:
        suggestions.append("# Fix mock embeddings - use local FastEmbed:")
        suggestions.append("export EMBEDDING_PROVIDER=local")
        suggestions.append("export EMBEDDING_MODEL=BAAI/bge-m3")
        suggestions.append("")

    # Generate LLM fix suggestions
    if is_mock_llm:
        suggestions.append("# Fix mock LLM - Option 1: Use Ollama (local):")
        suggestions.append("export LLM_PROVIDER=ollama")
        suggestions.append("export OLLAMA_MODEL=qwen2.5:7b-instruct")
        suggestions.append("export OLLAMA_BASE_URL=http://localhost:11434")
        suggestions.append("")
        suggestions.append("# Fix mock LLM - Option 2: Use OpenAI (API):")
        suggestions.append("export LLM_PROVIDER=openai")
        suggestions.append("export OPENAI_API_KEY=sk-your-key-here")
        suggestions.append("")

    # If no issues found
    if not suggestions:
        suggestions.append("# ✓ Configuration looks good!")
        suggestions.append("# All providers are configured with real models.")

    return suggestions


def format_status_icon(status: bool, warning_if_false: bool = False) -> str:
    """Return a Rich-formatted YES/NO status icon.

    Args:
        status: Boolean status value to render.
        warning_if_false: If True, render a warning style when `status` is False.

    Returns:
        A Rich markup string representing the status.
    """
    if status:
        return "[green]✓ YES[/green]"
    elif warning_if_false:
        return "[yellow]⚠ NO[/yellow]"
    else:
        return "[dim]NO[/dim]"


def format_bool_display(value: bool) -> str:
    """Display a boolean as colored YES/NO.

    Args:
        value: Boolean value to render.

    Returns:
        A Rich markup string representing the boolean.
    """
    return "[green]YES[/green]" if value else "[dim]NO[/dim]"


def format_env_value(value: Optional[str], default: str = "[dim]not set[/dim]") -> str:
    """Format an environment variable value for display.

    Args:
        value: Environment variable value, or None if not set.
        default: Default Rich markup string if `value` is None.

    Returns:
        A Rich markup string for the environment variable value.
    """
    if value is None:
        return default
    return f"[cyan]{value}[/cyan]"
