"""Bootstrap helpers for initializing pipeline components."""

from dataclasses import dataclass
from typing import Callable, Optional

from config import Config
from config import config as default_config
from src.consolidation import ConsolidationPipeline
from src.embeddings import EmbeddingProvider, get_embedding_provider
from src.evaluation import EvaluationRunner
from src.evaluation.runner import get_scenario
from src.ingestion import IngestionPipeline
from src.llm import LLMProvider, get_llm_provider
from src.retrieval import RetrievalEngine
from src.storage import LanceStore


@dataclass
class PipelineComponents:
    """Container for initialized pipeline components."""

    lance_store: LanceStore
    embedding_provider: EmbeddingProvider
    llm: LLMProvider

    IngestionPipeline: type
    ConsolidationPipeline: type
    RetrievalEngine: type
    EvaluationRunner: type
    get_scenario: Callable


def get_components(
    config: Optional[Config] = None, force_mock: bool = False, verbose: bool = True
) -> PipelineComponents:
    """Initialize and return all pipeline components."""
    cfg = config or default_config
    cfg.ensure_directories()

    lance_store = LanceStore(cfg.lance_db_path, cfg.embedding_dimension)
    embedding_provider = _create_embedding_provider(cfg, force_mock, verbose)
    llm = _create_llm_provider(cfg, force_mock, verbose)

    return PipelineComponents(
        lance_store=lance_store,
        embedding_provider=embedding_provider,
        llm=llm,
        IngestionPipeline=IngestionPipeline,
        ConsolidationPipeline=ConsolidationPipeline,
        RetrievalEngine=RetrievalEngine,
        EvaluationRunner=EvaluationRunner,
        get_scenario=get_scenario,
    )


def _create_embedding_provider(
    config: Config, force_mock: bool, verbose: bool
) -> EmbeddingProvider:
    """Create the configured embedding provider."""
    if force_mock or config.embedding_provider == "mock":
        if verbose:
            _log("[yellow]Using mock embedding provider[/yellow]")
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

    if verbose:
        _log(f"[green]Using local embeddings: {config.embedding_model}[/green]")
    return get_embedding_provider("local", model=config.embedding_model)


def _create_llm_provider(config: Config, force_mock: bool, verbose: bool) -> LLMProvider:
    """Create the configured LLM provider."""
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

    if verbose:
        _log("[yellow]Using mock LLM provider (no API key configured)[/yellow]")
    return get_llm_provider("mock")


def _log(message: str) -> None:
    """Print using Rich when available, otherwise plain text."""
    try:
        from rich.console import Console

        Console().print(message)
    except ImportError:
        import re

        plain = re.sub(r"\[/?[^\]]+\]", "", message)
        print(plain)
