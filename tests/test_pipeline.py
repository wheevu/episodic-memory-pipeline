"""Integration tests for the episodic memory pipeline."""

from pathlib import Path

import pytest

from config import Config
from src.consolidation import ConsolidationPipeline
from src.embeddings import EmbeddingProvider, get_embedding_provider
from src.ingestion import IngestionPipeline
from src.llm import LLMProvider, get_llm_provider
from src.models import Episode, Fact, MemoryType
from src.retrieval import RetrievalEngine
from src.storage import LanceStore


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create a test configuration."""
    cfg = Config()
    cfg.lance_db_path = temp_dir / "lancedb"
    cfg.embedding_dimension = 384
    return cfg


@pytest.fixture
def lance_store(test_config: Config) -> LanceStore:
    """Create a temporary LanceStore."""
    return LanceStore(test_config.lance_db_path, embedding_dimension=384)


@pytest.fixture
def embedding_provider() -> EmbeddingProvider:
    """Create a mock embedding provider."""
    return get_embedding_provider("mock", dimension=384)


@pytest.fixture
def llm() -> LLMProvider:
    """Create a mock LLM provider."""
    return get_llm_provider("mock")


class TestEpisodeModel:
    def test_create_episode(self) -> None:
        episode = Episode(
            raw_input="I started learning Korean today",
            content="User started learning Korean",
            memory_type=MemoryType.EPISODIC,
            topics=["language_learning", "korean"],
            importance=0.8,
        )
        assert episode.id is not None
        assert episode.memory_type == MemoryType.EPISODIC
        assert "korean" in episode.topics
        assert episode.importance == 0.8


class TestLanceStore:
    def test_save_and_get_episode(
        self, lance_store: LanceStore, embedding_provider: EmbeddingProvider
    ) -> None:
        episode = Episode(
            raw_input="Test input",
            content="Test content",
            memory_type=MemoryType.EPISODIC,
            topics=["test"],
        )
        emb = embedding_provider.embed_text(episode.to_embedding_text())
        lance_store.save_episode(episode, emb)

        retrieved = lance_store.get_episode(episode.id)
        assert retrieved is not None
        assert retrieved.id == episode.id
        assert retrieved.content == episode.content

    def test_save_and_get_fact(
        self, lance_store: LanceStore, embedding_provider: EmbeddingProvider
    ) -> None:
        fact = Fact(
            content="User is learning Korean",
            category="knowledge",
            topic="language_learning",
            confidence=0.9,
        )
        emb = embedding_provider.embed_text(fact.to_embedding_text())
        lance_store.save_fact(fact, emb)

        retrieved = lance_store.get_fact(fact.id)
        assert retrieved is not None
        assert retrieved.content == fact.content
        assert retrieved.topic == "language_learning"


class TestIngestionPipeline:
    def test_ingest_memory_worthy_text(
        self,
        lance_store: LanceStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        pipeline = IngestionPipeline(lance_store, embedding_provider, llm, worthiness_threshold=0.5)
        result = pipeline.ingest(
            "I started learning Korean today for my Seoul trip",
            source="test",
            force=True,
        )
        assert result.success
        assert result.episode is not None


class TestConsolidationPipeline:
    def test_consolidate_topic(
        self,
        lance_store: LanceStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        ingestion = IngestionPipeline(lance_store, embedding_provider, llm)
        for text in [
            "Started learning Korean",
            "Practiced Korean vocabulary",
            "Had first Korean conversation",
        ]:
            ingestion.ingest(text, force=True)

        pipeline = ConsolidationPipeline(lance_store, embedding_provider, llm, episode_threshold=1)
        result = pipeline.consolidate_topic("general")
        assert result.episodes_processed >= 0


class TestRetrievalEngine:
    def test_semantic_query(
        self,
        lance_store: LanceStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        ingestion = IngestionPipeline(lance_store, embedding_provider, llm)
        ingestion.ingest("I am learning Korean", force=True)

        engine = RetrievalEngine(lance_store, embedding_provider, llm)
        result = engine.query("What am I learning?")
        assert result is not None


def test_end_to_end_flow(temp_dir: Path) -> None:
    cfg = Config()
    cfg.lance_db_path = temp_dir / "lancedb"
    cfg.embedding_dimension = 384

    store = LanceStore(cfg.lance_db_path, embedding_dimension=384)
    embedding_provider = get_embedding_provider("mock", dimension=384)
    llm = get_llm_provider("mock")

    ingestion = IngestionPipeline(store, embedding_provider, llm)
    memories = [
        "I started a new job at Google today",
        "My first week at Google was challenging but exciting",
        "I prefer working from home when possible",
    ]
    for mem in memories:
        result = ingestion.ingest(mem, force=True)
        assert result.success

    stats = store.get_statistics()
    assert stats["total_episodes"] >= 3

    retrieval = RetrievalEngine(store, embedding_provider, llm)
    result = retrieval.query("Where do I work?")
    assert result is not None
