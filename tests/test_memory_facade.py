"""
Tests for the MemorySystem facade (src/memory.py).

These tests verify the full agent-facing API using mock providers so they
run quickly, with no network access or model downloads.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from config import Config

# Mark all tests in this module as requiring FAISS
pytestmark = pytest.mark.requires_faiss


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    tmp = tempfile.mkdtemp(prefix="mem_facade_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    """Create a Config pointing at the temp directory with mock providers."""
    cfg = Config()
    cfg.database_path = temp_dir / "test.db"
    cfg.vector_index_path = temp_dir / "test.faiss"
    cfg.embedding_provider = "mock"
    cfg.embedding_dimension = 384
    return cfg


@pytest.fixture
def mem(test_config):
    """Create a MemorySystem instance backed by mock providers."""
    from src.memory import MemorySystem

    return MemorySystem(config=test_config, force_mock=True)


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestConstruction:
    """MemorySystem construction and lazy init."""

    def test_import_from_package(self):
        """MemorySystem is importable from ``src``."""
        from src import MemorySystem

        assert MemorySystem is not None

    def test_lazy_init(self, test_config):
        """Components are NOT bootstrapped until first access."""
        from src.memory import MemorySystem

        m = MemorySystem(config=test_config, force_mock=True)
        assert m._components is None
        assert repr(m) == "<MemorySystem lazy mock=True>"

    def test_eager_init(self, test_config):
        """eager=True triggers immediate bootstrap."""
        from src.memory import MemorySystem

        m = MemorySystem(config=test_config, force_mock=True, eager=True)
        assert m._components is not None
        assert "initialised" in repr(m)

    def test_repr(self, mem):
        """repr changes after first access."""
        _ = mem.database  # triggers bootstrap
        assert "initialised" in repr(mem)


# ------------------------------------------------------------------
# remember
# ------------------------------------------------------------------


class TestRemember:
    """Test the remember (ingest) API."""

    def test_remember_stores_episode(self, mem):
        """A non-trivial text should be stored successfully."""
        result = mem.remember(
            "I started learning Korean today using Duolingo. "
            "Completed the first lesson on basic greetings.",
            force=True,
        )
        assert result.success
        assert result.episode is not None
        assert result.episode.id

    def test_remember_returns_episode_id(self, mem):
        """The returned episode has a valid UUID-style id."""
        result = mem.remember("Met with the team about the new project timeline.", force=True)
        assert len(result.episode.id) == 36  # UUID

    def test_remember_empty_text_skipped(self, mem):
        """Empty input is rejected."""
        result = mem.remember("", force=True)
        assert not result.success

    def test_remember_whitespace_skipped(self, mem):
        """Whitespace-only input is rejected."""
        result = mem.remember("   \n\t  ", force=True)
        assert not result.success

    def test_remember_batch(self, mem):
        """Batch ingestion stores multiple episodes."""
        texts = [
            "Had coffee with Sarah at the new cafe downtown.",
            "Finished reading chapter 5 of the ML textbook.",
            "Went for a 5K run in the park this morning.",
        ]
        results = mem.remember_batch(texts, force=True)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_remember_with_metadata(self, mem):
        """Source and session_id propagate to the stored episode."""
        result = mem.remember(
            "Deployed v2.1 of the recommendation engine to staging.",
            source="deploy_bot",
            session_id="deploy-session-42",
            force=True,
        )
        assert result.success
        assert result.episode.source == "deploy_bot"
        assert result.episode.session_id == "deploy-session-42"

    def test_remember_default_source_is_agent(self, mem):
        """Default source for the facade is 'agent'."""
        result = mem.remember("Testing default source label.", force=True)
        assert result.success
        assert result.episode.source == "agent"


# ------------------------------------------------------------------
# recall
# ------------------------------------------------------------------


class TestRecall:
    """Test the recall (query) API."""

    def test_recall_empty_system(self, mem):
        """Querying an empty system returns a result (no crash)."""
        result = mem.recall("What am I learning?", synthesize=False)
        assert result is not None
        assert isinstance(result.episodes, list)

    def test_recall_after_ingest(self, mem):
        """After storing memories, recall should return a QueryResult."""
        mem.remember("I love hiking in the mountains on weekends.", force=True)
        mem.remember("Started training for a marathon next month.", force=True)

        result = mem.recall("What are my hobbies?", synthesize=False)
        assert result is not None
        # With mock embeddings we can't guarantee semantic relevance,
        # but the query should complete without error.
        assert hasattr(result, "episodes")
        assert hasattr(result, "facts")
        assert hasattr(result, "query_type")

    def test_recall_narrative(self, mem):
        """recall_narrative returns a QueryResult."""
        mem.remember("Day 1 of Korean: learned basic greetings.", force=True)
        mem.remember("Day 2 of Korean: learned numbers 1-10.", force=True)

        result = mem.recall_narrative("korean")
        assert result is not None
        assert result.query_type == "narrative"

    def test_quick_lookup(self, mem):
        """quick_lookup returns a list of Fact objects."""
        mem.remember("My favourite programming language is Python.", force=True)
        facts = mem.quick_lookup("favourite language")
        assert isinstance(facts, list)


# ------------------------------------------------------------------
# get_context
# ------------------------------------------------------------------


class TestGetContext:
    """Test the get_context API."""

    def test_get_context_returns_dict(self, mem):
        """get_context returns a dict with expected keys."""
        ctx = mem.get_context("nonexistent_topic")
        assert isinstance(ctx, dict)
        assert "topic" in ctx
        assert "recent_episodes" in ctx
        assert "facts" in ctx
        assert ctx["topic"] == "nonexistent_topic"


# ------------------------------------------------------------------
# consolidate
# ------------------------------------------------------------------


class TestConsolidate:
    """Test the consolidate API."""

    def test_consolidate_empty(self, mem):
        """Consolidation on empty system returns empty list."""
        results = mem.consolidate()
        assert isinstance(results, list)

    def test_consolidate_returns_results(self, mem):
        """After ingesting enough episodes, consolidation returns results."""
        for i in range(6):
            mem.remember(
                f"Korean lesson {i + 1}: practiced vocabulary and grammar drills.",
                force=True,
            )
        results = mem.consolidate()
        assert isinstance(results, list)


# ------------------------------------------------------------------
# forget
# ------------------------------------------------------------------


class TestForget:
    """Test the forget (soft-delete) API."""

    def test_forget_episode(self, mem):
        """Forgetting an episode deactivates it."""
        result = mem.remember("Secret I want to forget.", force=True)
        episode_id = result.episode.id

        forgotten = mem.forget(episode_id=episode_id)
        assert forgotten is True

        # Verify it's deactivated
        episode = mem.database.get_episode(episode_id)
        assert episode is not None
        assert episode.is_active is False

    def test_forget_nonexistent_episode(self, mem):
        """Forgetting a non-existent episode returns False."""
        forgotten = mem.forget(episode_id="nonexistent-id-1234")
        assert forgotten is False

    def test_forget_requires_exactly_one_id(self, mem):
        """Providing neither or both IDs raises ValueError."""
        with pytest.raises(ValueError, match="exactly one"):
            mem.forget()

        with pytest.raises(ValueError, match="exactly one"):
            mem.forget(episode_id="a", fact_id="b")

    def test_forget_fact(self, mem):
        """Forgetting a fact deactivates it."""
        # Ingest and consolidate to create facts
        for i in range(6):
            mem.remember(
                f"I enjoy cooking Italian food, especially pasta dish {i + 1}.", force=True
            )
        mem.consolidate()

        # Get a fact (if any were created by mock LLM)
        facts = mem.database.get_facts()
        if facts:
            fact_id = facts[0].id
            forgotten = mem.forget(fact_id=fact_id)
            assert forgotten is True
            fact = mem.database.get_fact(fact_id)
            assert fact.is_active is False


# ------------------------------------------------------------------
# stats / topics
# ------------------------------------------------------------------


class TestIntrospection:
    """Test stats and topics."""

    def test_stats_returns_dict(self, mem):
        """stats() returns a dict with sub-dicts."""
        s = mem.stats()
        assert isinstance(s, dict)
        assert "database" in s
        assert "vector_store" in s

    def test_topics_empty(self, mem):
        """topics() on empty system returns empty list."""
        t = mem.topics()
        assert isinstance(t, list)

    def test_topics_after_ingest(self, mem):
        """topics() reflects ingested data."""
        mem.remember("Started a new Python project at work today.", force=True)
        t = mem.topics()
        # With mock LLM the topics depend on the mock extractor output,
        # but the method should complete without error.
        assert isinstance(t, list)


# ------------------------------------------------------------------
# escape hatches
# ------------------------------------------------------------------


class TestEscapeHatches:
    """Test direct access to underlying components."""

    def test_database_access(self, mem):
        """database property returns the live Database instance."""
        db = mem.database
        assert db is not None
        # Should be able to call get_statistics
        stats = db.get_statistics()
        assert isinstance(stats, dict)

    def test_vector_store_access(self, mem):
        """vector_store property returns the live VectorStore instance."""
        vs = mem.vector_store
        assert vs is not None
        stats = vs.get_statistics()
        assert isinstance(stats, dict)
