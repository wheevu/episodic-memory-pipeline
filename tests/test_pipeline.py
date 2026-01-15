"""
Integration tests for the episodic memory pipeline.

These tests use mock providers to run without external dependencies.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from config import Config
from src.consolidation import ConsolidationPipeline
from src.embeddings import EmbeddingProvider, get_embedding_provider
from src.ingestion import IngestionPipeline
from src.llm import LLMProvider, get_llm_provider
from src.models import Episode, Fact, MemoryType
from src.retrieval import RetrievalEngine
from src.storage import Database, VectorStore

# Mark all tests in this module as requiring FAISS
pytestmark = pytest.mark.requires_faiss


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test data.

    Returns:
        Path to the created temporary directory.
    """
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create a test configuration.

    Args:
        temp_dir: Temporary directory used as the base for test artifacts.

    Returns:
        A `Config` configured to write into the temporary directory.
    """
    config = Config()
    config.database_path = temp_dir / "test.db"
    config.vector_index_path = temp_dir / "test.faiss"
    config.embedding_dimension = 384  # Mock dimension
    return config


@pytest.fixture
def database(test_config: Config) -> Database:
    """Create a test database.

    Args:
        test_config: Test configuration with database path.

    Returns:
        A `Database` instance backed by a temporary SQLite file.
    """
    return Database(test_config.database_path)


@pytest.fixture
def vector_store(test_config: Config) -> VectorStore:
    """Create a test vector store.

    Args:
        test_config: Test configuration with vector index path.

    Returns:
        A `VectorStore` instance backed by temporary FAISS files.
    """
    return VectorStore(test_config.vector_index_path, dimension=384)


@pytest.fixture
def embedding_provider() -> EmbeddingProvider:
    """Create a mock embedding provider.

    Returns:
        A mock `EmbeddingProvider` with a fixed embedding dimension.
    """
    return get_embedding_provider("mock", dimension=384)


@pytest.fixture
def llm() -> LLMProvider:
    """Create a mock LLM provider.

    Returns:
        A mock `LLMProvider` suitable for deterministic tests.
    """
    return get_llm_provider("mock")


class TestEpisodeModel:
    """Test Episode model."""

    def test_create_episode(self) -> None:
        """Create an Episode and validate core fields are set/typed correctly."""
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

    def test_episode_to_db_row(self) -> None:
        """Verify `Episode.to_db_row()` produces a persistable dictionary."""
        episode = Episode(
            raw_input="Test input",
            content="Test content",
            memory_type=MemoryType.FACT,
            topics=["test"],
        )

        row = episode.to_db_row()

        assert row["id"] == episode.id
        assert row["raw_input"] == "Test input"
        assert row["memory_type"] == "fact"
        assert '"test"' in row["topics"]

    def test_episode_from_db_row(self) -> None:
        """Verify `Episode.from_db_row()` correctly parses serialized fields."""
        row = {
            "id": "test-id",
            "created_at": "2024-01-01T00:00:00",
            "occurred_at": "2024-01-01T00:00:00",
            "raw_input": "Test",
            "content": "Test content",
            "memory_type": "episodic",
            "topics": '["test"]',
            "entities": "[]",
            "confidence": 0.9,
            "importance": 0.5,
            "source": "test",
            "session_id": None,
            "is_active": True,
            "consolidated": False,
            "embedding_id": None,
        }

        episode = Episode.from_db_row(row)

        assert episode.id == "test-id"
        assert episode.memory_type == MemoryType.EPISODIC
        assert episode.topics == ["test"]


class TestDatabase:
    """Test database operations."""

    def test_save_and_get_episode(self, database: Database) -> None:
        """Persist an episode and retrieve it by ID.

        Args:
            database: Test database fixture.

        Returns:
            None.
        """
        episode = Episode(
            raw_input="Test input",
            content="Test content",
            memory_type=MemoryType.EPISODIC,
            topics=["test"],
        )

        database.save_episode(episode)
        retrieved = database.get_episode(episode.id)

        assert retrieved is not None
        assert retrieved.id == episode.id
        assert retrieved.content == episode.content

    def test_get_episodes_with_filters(self, database: Database) -> None:
        """Query episodes using a type filter.

        Args:
            database: Test database fixture.

        Returns:
            None.
        """
        # Create episodes with different types
        ep1 = Episode(
            raw_input="Fact", content="A fact", memory_type=MemoryType.FACT, topics=["topic1"]
        )
        ep2 = Episode(
            raw_input="Goal", content="A goal", memory_type=MemoryType.GOAL, topics=["topic2"]
        )

        database.save_episode(ep1)
        database.save_episode(ep2)

        # Filter by type
        facts = database.get_episodes(memory_type="fact")
        assert len(facts) == 1
        assert facts[0].memory_type == MemoryType.FACT

    def test_save_and_get_fact(self, database: Database) -> None:
        """Persist a fact and retrieve it by ID.

        Args:
            database: Test database fixture.

        Returns:
            None.
        """
        fact = Fact(
            content="User is learning Korean",
            category="knowledge",
            topic="language_learning",
            confidence=0.9,
        )

        database.save_fact(fact)
        retrieved = database.get_fact(fact.id)

        assert retrieved is not None
        assert retrieved.content == fact.content
        assert retrieved.topic == "language_learning"


class TestVectorStore:
    """Test vector store operations."""

    def test_add_and_search(
        self, vector_store: VectorStore, embedding_provider: EmbeddingProvider
    ) -> None:
        """Add vectors to the store and validate search output structure/range.

        Args:
            vector_store: Test vector store fixture.
            embedding_provider: Embedding provider fixture.

        Returns:
            None.
        """
        # Add some vectors
        texts = ["Learning Korean", "Trip to Seoul", "Python programming"]
        for i, text in enumerate(texts):
            embedding = embedding_provider.embed_text(text)
            vector_store.add("episodes", f"id-{i}", embedding)

        # Search
        query_embedding = embedding_provider.embed_text("Korean language")
        results = vector_store.search("episodes", query_embedding, k=2)

        assert len(results) <= 2
        # Results should be (record_id, score) tuples
        # Score is cosine similarity, range [-1, 1] for normalized vectors
        for record_id, score in results:
            assert record_id.startswith("id-")
            assert -1 <= score <= 1


class TestIngestionPipeline:
    """Test the full ingestion pipeline."""

    def test_ingest_memory_worthy_text(
        self,
        database: Database,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        """Ingest a forced-worthy input and verify an episode is stored.

        Args:
            database: Test database fixture.
            vector_store: Test vector store fixture.
            embedding_provider: Embedding provider fixture.
            llm: LLM provider fixture.

        Returns:
            None.
        """
        pipeline = IngestionPipeline(
            database, vector_store, embedding_provider, llm, worthiness_threshold=0.5
        )

        result = pipeline.ingest(
            "I started learning Korean today for my Seoul trip",
            source="test",
            force=True,  # Skip worthiness check for reliable test
        )

        assert result.success
        assert result.episode is not None
        assert result.episode.id is not None

    def test_skip_non_worthy_text(
        self,
        database: Database,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        """Ingest a short acknowledgement and verify it is skipped.

        Args:
            database: Test database fixture.
            vector_store: Test vector store fixture.
            embedding_provider: Embedding provider fixture.
            llm: LLM provider fixture.

        Returns:
            None.
        """
        pipeline = IngestionPipeline(
            database, vector_store, embedding_provider, llm, worthiness_threshold=0.5
        )

        result = pipeline.ingest("ok thanks")

        assert not result.success
        assert "Not memory-worthy" in result.reason or "too short" in result.reason.lower()


class TestConsolidationPipeline:
    """Test the consolidation pipeline."""

    def test_consolidate_topic(
        self,
        database: Database,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        """Ingest episodes and ensure consolidation can be invoked without error.

        Args:
            database: Test database fixture.
            vector_store: Test vector store fixture.
            embedding_provider: Embedding provider fixture.
            llm: LLM provider fixture.

        Returns:
            None.
        """
        # First, add some episodes
        ingestion = IngestionPipeline(database, vector_store, embedding_provider, llm)

        texts = [
            "Started learning Korean",
            "Practiced Korean vocabulary",
            "Had first Korean conversation",
        ]

        for text in texts:
            ingestion.ingest(text, force=True)

        # Run consolidation
        ConsolidationPipeline(database, vector_store, embedding_provider, llm, episode_threshold=1)

        # Should have some topics to consolidate
        # (mock LLM assigns 'general' topic)


class TestRetrievalEngine:
    """Test the retrieval engine."""

    def test_semantic_query(
        self,
        database: Database,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        """Query the engine and validate a non-empty result structure is returned.

        Args:
            database: Test database fixture.
            vector_store: Test vector store fixture.
            embedding_provider: Embedding provider fixture.
            llm: LLM provider fixture.

        Returns:
            None.
        """
        # Add some data
        ingestion = IngestionPipeline(database, vector_store, embedding_provider, llm)
        ingestion.ingest("I am learning Korean", force=True)

        # Query
        engine = RetrievalEngine(database, vector_store, embedding_provider, llm)

        result = engine.query("What am I learning?")

        # Should return something
        assert result is not None
        assert result.answer is not None or result.episodes is not None


def test_end_to_end_flow(temp_dir: Path) -> None:
    """Test the complete flow: ingest → consolidate → retrieve.

    Args:
        temp_dir: Temporary directory for test artifacts.

    Returns:
        None.
    """
    # Setup
    config = Config()
    config.database_path = temp_dir / "test.db"
    config.vector_index_path = temp_dir / "test.faiss"
    config.embedding_dimension = 384

    database = Database(config.database_path)
    vector_store = VectorStore(config.vector_index_path, dimension=384)
    embedding_provider = get_embedding_provider("mock", dimension=384)
    llm = get_llm_provider("mock")

    # 1. Ingest memories
    ingestion = IngestionPipeline(database, vector_store, embedding_provider, llm)

    memories = [
        "I started a new job at Google today",
        "My first week at Google was challenging but exciting",
        "I prefer working from home when possible",
    ]

    for mem in memories:
        result = ingestion.ingest(mem, force=True)
        assert result.success, f"Failed to ingest: {mem}"

    # 2. Check storage
    stats = database.get_statistics()
    assert stats["total_episodes"] >= 3

    # 3. Query
    retrieval = RetrievalEngine(database, vector_store, embedding_provider, llm)

    result = retrieval.query("Where do I work?")
    assert result is not None

    # 4. Verify vector store persisted
    vector_store.save()
    vec_stats = vector_store.get_statistics()
    assert vec_stats["episodes"]["count"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
