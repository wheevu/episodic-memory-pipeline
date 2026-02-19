"""
Tests for input sanitization in the ingestion pipeline.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from config import Config
from src.embeddings import get_embedding_provider
from src.ingestion import IngestionPipeline
from src.llm import get_llm_provider
from src.storage import LanceStore
from src.utils import sanitize_entities, sanitize_topics


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test data."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create a test configuration."""
    config = Config()
    config.lance_db_path = temp_dir / "lancedb"
    config.embedding_dimension = 384
    return config


@pytest.fixture
def pipeline(test_config: Config) -> IngestionPipeline:
    """Create a test pipeline with mock providers."""
    lance_store = LanceStore(test_config.lance_db_path, embedding_dimension=384)
    embedding_provider = get_embedding_provider("mock", dimension=384)
    llm = get_llm_provider("mock")

    return IngestionPipeline(
        lance_store=lance_store,
        embedding_provider=embedding_provider,
        llm=llm,
    )


class TestInputLengthValidation:
    """Tests for input length validation."""

    def test_normal_length_input(self, pipeline: IngestionPipeline) -> None:
        """Test that normal-length inputs are accepted."""
        result = pipeline.ingest("This is a normal length input", force=True)
        assert result.success

    def test_empty_input_rejected(self, pipeline: IngestionPipeline) -> None:
        """Test that empty inputs are rejected."""
        result = pipeline.ingest("", force=True)
        assert not result.success
        assert "Empty input" in result.reason

    def test_whitespace_only_input_rejected(self, pipeline: IngestionPipeline) -> None:
        """Test that whitespace-only inputs are rejected."""
        result = pipeline.ingest("   \t\n  ", force=True)
        assert not result.success
        assert "Empty input" in result.reason

    def test_extremely_long_input_rejected(self, pipeline: IngestionPipeline) -> None:
        """Test that extremely long inputs are rejected."""
        # Create text longer than MAX_TEXT_LENGTH (50,000 chars)
        long_text = "a" * 60_000
        result = pipeline.ingest(long_text, force=True)
        assert not result.success
        assert "too long" in result.reason.lower()

    def test_max_length_boundary(self, pipeline: IngestionPipeline) -> None:
        """Test input at the maximum allowed length."""
        # Create text exactly at MAX_TEXT_LENGTH (50,000 chars)
        boundary_text = "a" * 50_000
        result = pipeline.ingest(boundary_text, force=True)
        assert result.success


class TestTopicsSanitization:
    """Tests for topics sanitization."""

    def test_normal_topics(self) -> None:
        """Test that normal topics are preserved."""
        topics = ["work", "learning", "health"]
        result = sanitize_topics(topics)
        assert result == ["work", "learning", "health"]

    def test_topics_with_whitespace(self) -> None:
        """Test that whitespace is stripped from topics."""
        topics = ["  work  ", "\tlearning\n", " health "]
        result = sanitize_topics(topics)
        assert result == ["work", "learning", "health"]

    def test_none_topics(self) -> None:
        """Test that None topics return empty list."""
        result = sanitize_topics(None)
        assert result == []

    def test_non_string_topics_filtered(self) -> None:
        """Test that non-string topics are filtered out."""
        topics = ["work", 123, None, "learning", {"dict": "value"}]
        result = sanitize_topics(topics)
        assert result == ["work", "learning"]

    def test_empty_topics_filtered(self) -> None:
        """Test that empty string topics are filtered out."""
        topics = ["work", "", "   ", "learning"]
        result = sanitize_topics(topics)
        assert result == ["work", "learning"]

    def test_topics_truncated_to_max_length(self) -> None:
        """Test that long topics are truncated."""
        long_topic = "a" * 200
        topics = [long_topic]
        result = sanitize_topics(topics, max_length=100)
        assert len(result) == 1
        assert len(result[0]) == 100
        assert result[0] == "a" * 100

    def test_topics_limited_to_max_count(self) -> None:
        """Test that topic count is limited."""
        topics = [f"topic{i}" for i in range(30)]
        result = sanitize_topics(topics, max_topics=20)
        assert len(result) == 20
        assert result == [f"topic{i}" for i in range(20)]


class TestEntitiesSanitization:
    """Tests for entities sanitization."""

    def test_normal_entities(self) -> None:
        """Test that normal entities are preserved."""
        entities = ["John Doe", "OpenAI", "Python"]
        result = sanitize_entities(entities)
        assert result == ["John Doe", "OpenAI", "Python"]

    def test_entities_with_whitespace(self) -> None:
        """Test that whitespace is stripped from entities."""
        entities = ["  John  ", "\tOpenAI\n", " Python "]
        result = sanitize_entities(entities)
        assert result == ["John", "OpenAI", "Python"]

    def test_none_entities(self) -> None:
        """Test that None entities return empty list."""
        result = sanitize_entities(None)
        assert result == []

    def test_non_string_entities_filtered(self) -> None:
        """Test that non-string entities are filtered out."""
        entities = ["John", 456, None, "OpenAI", ["list"]]
        result = sanitize_entities(entities)
        assert result == ["John", "OpenAI"]

    def test_entities_truncated_to_max_length(self) -> None:
        """Test that long entities are truncated."""
        long_entity = "a" * 200
        entities = [long_entity]
        result = sanitize_entities(entities, max_length=100)
        assert len(result) == 1
        assert len(result[0]) == 100

    def test_entities_limited_to_max_count(self) -> None:
        """Test that entity count is limited."""
        entities = [f"entity{i}" for i in range(60)]
        result = sanitize_entities(entities, max_entities=50)
        assert len(result) == 50
        assert result == [f"entity{i}" for i in range(50)]

    def test_entities_allow_more_than_topics(self) -> None:
        """Test that entities allow a higher count than topics."""
        # Default max_topics=20, max_entities=50
        topics = [f"topic{i}" for i in range(30)]
        entities = [f"entity{i}" for i in range(60)]

        sanitized_topics = sanitize_topics(topics)
        sanitized_entities = sanitize_entities(entities)

        assert len(sanitized_topics) == 20  # Limited to 20
        assert len(sanitized_entities) == 50  # Limited to 50
