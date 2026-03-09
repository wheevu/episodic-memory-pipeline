"""Safety and correctness tests for LanceStore filtering and updates."""

from datetime import datetime, timezone

import numpy as np

from src.models import Episode, Fact
from src.models.episode import MemoryType
from src.models.fact import FactCategory


def _vec(dim: int = 384) -> np.ndarray:
    return np.ones(dim, dtype=np.float32)


def test_fact_topic_filter_escapes_quotes(lance_store) -> None:
    """Topics containing single quotes should be queryable."""
    topic = "bob's-goals"
    fact = Fact(content="User plans a trip", category=FactCategory.GOAL, topic=topic)
    lance_store.save_fact(fact, _vec(), source_episode_ids=[])

    rows = lance_store.get_facts(topic=topic)

    assert len(rows) == 1
    assert rows[0].id == fact.id


def test_fact_topic_filter_blocks_injected_where(lance_store) -> None:
    """Injection-like topic strings should not broaden query scope."""
    fact_a = Fact(content="A", category=FactCategory.GOAL, topic="safe")
    fact_b = Fact(content="B", category=FactCategory.GOAL, topic="other")
    lance_store.save_fact(fact_a, _vec(), source_episode_ids=[])
    lance_store.save_fact(fact_b, _vec(), source_episode_ids=[])

    injected_topic = "safe' OR is_active = true OR 'x"
    rows = lance_store.get_facts(topic=injected_topic)

    assert rows == []


def test_episode_update_escapes_id_quotes(lance_store) -> None:
    """Update filters should handle IDs with quotes safely."""
    episode = Episode(
        id="episode'quoted",
        raw_input="raw",
        content="content",
        memory_type=MemoryType.EPISODIC,
        occurred_at=datetime.now(timezone.utc),
    )
    lance_store.save_episode(episode, _vec())

    lance_store.set_episode_active(episode.id, False)

    fetched = lance_store.get_episode(episode.id)
    assert fetched is not None
    assert fetched.is_active is False


def test_batch_id_lookup_escapes_quotes(lance_store) -> None:
    """IN-clause ID lookup should handle quote-containing IDs safely."""
    episode = Episode(
        id="episode'one",
        raw_input="raw",
        content="content",
        memory_type=MemoryType.EPISODIC,
        occurred_at=datetime.now(timezone.utc),
    )
    lance_store.save_episode(episode, _vec())

    found = lance_store.get_episodes_by_ids([episode.id])

    assert episode.id in found
