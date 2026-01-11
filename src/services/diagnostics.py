"""
Diagnostics service for the episodic memory pipeline.

This module contains business logic for system diagnostics.
Returns plain dataclasses - no Rich/Typer imports.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.bootstrap import PipelineComponents
    from config import Config


@dataclass
class ProviderStatus:
    """Status of a provider."""
    type: str
    model: str
    is_mock: bool
    device: str = "N/A"
    dimension: int = 0
    temperature: str = "N/A"
    base_url: Optional[str] = None


@dataclass
class BootstrapStatus:
    """Status of the bootstrap system."""
    is_initialized: bool
    has_cached_model: bool
    tokenizers_parallelism_disabled: bool


@dataclass
class VectorStoreStatus:
    """Status of the vector store."""
    index_type: str
    similarity_metric: str
    dimension: int
    dimension_match: bool
    indexes: Dict[str, int]  # name -> count
    total_vectors: int
    has_unsaved_changes: bool = False


@dataclass
class EvalReadiness:
    """Evaluation readiness status."""
    is_ready: bool
    warnings: List[str]


@dataclass
class DiagnosticsResult:
    """Complete diagnostics result."""
    bootstrap: BootstrapStatus
    llm: ProviderStatus
    embedding: ProviderStatus
    vector_store: VectorStoreStatus
    eval_readiness: EvalReadiness
    suggestions: List[str]


class DiagnosticsService:
    """
    Service for running system diagnostics.
    
    This service inspects the pipeline components and configuration
    to provide diagnostic information for debugging and verification.
    """
    
    def __init__(
        self,
        components: Optional["PipelineComponents"] = None,
        config: Optional["Config"] = None
    ) -> None:
        """
        Initialize the diagnostics service.
        
        Args:
            components: Pipeline components (optional, for full diagnostics)
            config: Config object (for dry-run diagnostics)
        
        Returns:
            None.
        """
        self.components = components
        self._config = config
    
    @property
    def config(self) -> "Config":
        """Get config, importing if needed.

        Returns:
            The resolved `Config` instance used for diagnostics.
        """
        if self._config is None:
            from config import config
            self._config = config
        return self._config
    
    def run_full_diagnostics(self) -> DiagnosticsResult:
        """
        Run full diagnostics with initialized components.
        
        Returns:
            DiagnosticsResult with all diagnostic information
        
        Raises:
            ValueError: If `components` were not provided to the service.
        """
        if self.components is None:
            raise ValueError("Components required for full diagnostics")
        
        bootstrap = self._get_bootstrap_status()
        llm = self._get_llm_status()
        embedding = self._get_embedding_status()
        vector_store = self._get_vector_store_status(embedding.dimension)
        eval_readiness = self._get_eval_readiness(llm, embedding, vector_store)
        suggestions = self._generate_suggestions(llm, embedding, vector_store)
        
        return DiagnosticsResult(
            bootstrap=bootstrap,
            llm=llm,
            embedding=embedding,
            vector_store=vector_store,
            eval_readiness=eval_readiness,
            suggestions=suggestions
        )
    
    def run_dry_diagnostics(self, force_mock: bool = False) -> Dict[str, Any]:
        """
        Run config-only diagnostics without initializing components.
        
        Args:
            force_mock: Whether --mock flag is set
            
        Returns:
            Dictionary with config inspection results
        """
        import os
        
        config = self.config
        
        # Environment variables
        env_vars = {
            "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER"),
            "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
            "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE"),
            "EMBEDDING_DIMENSION": os.getenv("EMBEDDING_DIMENSION"),
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER"),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
            "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
            "OPENAI_API_KEY": "***" if os.getenv("OPENAI_API_KEY") else None,
            "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM"),
        }
        
        # Resolved values
        resolved = {
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "embedding_device": config.embedding_device,
            "embedding_dimension": config.embedding_dimension,
            "llm_provider": config.llm_provider,
            "llm_model": (
                config.ollama_model 
                if config.llm_provider == "ollama" 
                else config.llm_model
            ),
            "llm_temperature": config.llm_temperature,
        }
        
        # Predict mock usage
        will_use_mock_embeddings = (
            force_mock or 
            config.embedding_provider == "mock"
        )
        
        will_use_mock_llm = force_mock
        if not force_mock:
            if config.llm_provider == "ollama":
                will_use_mock_llm = False
            elif config.openai_api_key:
                will_use_mock_llm = False
            else:
                will_use_mock_llm = True
        
        predictions = {
            "will_use_mock_embeddings": will_use_mock_embeddings,
            "will_use_mock_llm": will_use_mock_llm,
        }
        
        # Suggestions
        suggestions = []
        if will_use_mock_embeddings:
            suggestions.append("export EMBEDDING_PROVIDER=local")
            suggestions.append("export EMBEDDING_MODEL=BAAI/bge-m3")
        if will_use_mock_llm:
            suggestions.append("export LLM_PROVIDER=ollama")
            suggestions.append("export OLLAMA_MODEL=qwen2.5:7b-instruct")
        
        return {
            "env_vars": env_vars,
            "resolved": resolved,
            "predictions": predictions,
            "suggestions": suggestions,
        }
    
    def _get_bootstrap_status(self) -> BootstrapStatus:
        """Get bootstrap system status.

        Returns:
            `BootstrapStatus` describing whether bootstrap is initialized and cached.
        """
        import os
        from src.bootstrap import is_initialized, get_cached_embedding_model
        
        return BootstrapStatus(
            is_initialized=is_initialized(),
            has_cached_model=get_cached_embedding_model() is not None,
            tokenizers_parallelism_disabled=(
                os.environ.get('TOKENIZERS_PARALLELISM', '').lower() == 'false'
            )
        )
    
    def _get_llm_status(self) -> ProviderStatus:
        """Get LLM provider status.

        Returns:
            `ProviderStatus` describing the configured LLM provider.
        """
        from src.llm.interface import OllamaLLMProvider, OpenAILLMProvider, MockLLMProvider
        
        llm = self.components.llm
        config = self.config
        
        if isinstance(llm, MockLLMProvider):
            return ProviderStatus(
                type="mock",
                model="MockLLM (deterministic)",
                is_mock=True,
                temperature="N/A"
            )
        elif isinstance(llm, OllamaLLMProvider):
            return ProviderStatus(
                type="ollama",
                model=getattr(llm, '_model', config.ollama_model),
                is_mock=False,
                temperature=str(getattr(llm, '_default_temperature', config.llm_temperature)),
                base_url=getattr(llm, '_base_url', config.ollama_base_url)
            )
        elif isinstance(llm, OpenAILLMProvider):
            return ProviderStatus(
                type="openai",
                model=getattr(llm, '_model', config.llm_model),
                is_mock=False,
                temperature=str(config.llm_temperature),
                base_url="https://api.openai.com"
            )
        else:
            return ProviderStatus(
                type=llm.__class__.__name__,
                model="unknown",
                is_mock=getattr(llm, 'is_mock', True)
            )
    
    def _get_embedding_status(self) -> ProviderStatus:
        """Get embedding provider status.

        Returns:
            `ProviderStatus` describing the configured embedding provider.
        """
        from src.embeddings.interface import (
            LocalEmbeddingProvider,
            OllamaEmbeddingProvider,
            MockEmbeddingProvider,
        )
        
        emb = self.components.embedding_provider
        config = self.config
        
        if isinstance(emb, MockEmbeddingProvider):
            return ProviderStatus(
                type="mock",
                model="MockEmbedding (random normalized)",
                is_mock=True,
                dimension=emb.dimension
            )
        elif isinstance(emb, LocalEmbeddingProvider):
            model_name = config.embedding_model
            model_attr = getattr(emb, '_model', None)
            if model_attr and hasattr(model_attr, 'name_or_path'):
                model_name = model_attr.name_or_path or model_name
            
            return ProviderStatus(
                type="local",
                model=model_name,
                is_mock=False,
                device=config.embedding_device,
                dimension=emb.dimension
            )
        elif isinstance(emb, OllamaEmbeddingProvider):
            return ProviderStatus(
                type="ollama",
                model=getattr(emb, '_model', config.ollama_embed_model),
                is_mock=False,
                device="remote (Ollama server)",
                dimension=emb.dimension
            )
        else:
            return ProviderStatus(
                type=emb.__class__.__name__,
                model="unknown",
                is_mock=getattr(emb, 'is_mock', True),
                dimension=emb.dimension
            )
    
    def _get_vector_store_status(self, embedding_dim: int) -> VectorStoreStatus:
        """Get vector store status.

        Args:
            embedding_dim: Expected embedding dimension for consistency checks.

        Returns:
            `VectorStoreStatus` describing the current FAISS indexes and health.
        """
        vs = self.components.vector_store
        vs_stats = vs.get_statistics()
        
        indexes = {}
        total = 0
        has_unsaved = False
        for name, info in vs_stats.items():
            count = info.get('count', 0)
            indexes[name] = count
            total += count
            if info.get('has_unsaved_changes', False):
                has_unsaved = True
        
        return VectorStoreStatus(
            index_type="IndexFlatIP (Inner Product)",
            similarity_metric="Cosine (via inner product on L2-normalized vectors)",
            dimension=vs.dimension,
            dimension_match=(vs.dimension == embedding_dim),
            indexes=indexes,
            total_vectors=total,
            has_unsaved_changes=has_unsaved
        )
    
    def _get_eval_readiness(
        self,
        llm: ProviderStatus,
        embedding: ProviderStatus,
        vector_store: VectorStoreStatus
    ) -> EvalReadiness:
        """Determine evaluation readiness.

        Args:
            llm: LLM provider status.
            embedding: Embedding provider status.
            vector_store: Vector store status.

        Returns:
            `EvalReadiness` indicating whether evaluation results would be meaningful.
        """
        warnings = []
        
        if embedding.is_mock:
            warnings.append("Mock embeddings → Retrieval precision will be SKIPPED")
        if llm.is_mock:
            warnings.append("Mock LLM → Fact extraction metrics may not be meaningful")
        if not vector_store.dimension_match:
            warnings.append(f"Dimension mismatch → Vector search will fail! "
                          f"(vector_store={vector_store.dimension}, "
                          f"embedding={embedding.dimension})")
        
        return EvalReadiness(
            is_ready=len(warnings) == 0,
            warnings=warnings
        )
    
    def _generate_suggestions(
        self,
        llm: ProviderStatus,
        embedding: ProviderStatus,
        vector_store: VectorStoreStatus
    ) -> List[str]:
        """Generate fix suggestions for common diagnostic issues.

        Args:
            llm: LLM provider status.
            embedding: Embedding provider status.
            vector_store: Vector store status.

        Returns:
            A list of shell commands/comments the user can copy-paste.
        """
        suggestions = []
        
        if embedding.is_mock:
            suggestions.append("# Fix mock embeddings - use local SentenceTransformers:")
            suggestions.append("export EMBEDDING_PROVIDER=local")
            suggestions.append("export EMBEDDING_MODEL=BAAI/bge-m3")
            suggestions.append("export EMBEDDING_DEVICE=cpu  # or 'mps' for Apple Silicon")
        
        if llm.is_mock:
            suggestions.append("")
            suggestions.append("# Fix mock LLM - use Ollama (local):")
            suggestions.append("export LLM_PROVIDER=ollama")
            suggestions.append("export OLLAMA_MODEL=qwen2.5:7b-instruct")
        
        if vector_store.has_unsaved_changes:
            suggestions.append("")
            suggestions.append("# Persist unsaved vectors to disk:")
            suggestions.append("# In Python: vector_store.save()")
            suggestions.append("# Or use auto_save=True when initializing VectorStore")
        
        return suggestions

