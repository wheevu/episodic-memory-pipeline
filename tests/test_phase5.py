"""
Phase 5 tests — Regression tests for Phases 1-4 fixes.

Test groups:
  1. Provenance Integrity  — upsert preserves FK links (Fix 1)
  2. CLI Import Smoke      — no SyntaxError on import (Fix 4)
  3. Consolidation E2E     — full consolidate_topic flow (Fixes 2, 5)
  4. Inactive Filtering    — deactivated records excluded from search (Fix 6)
  5. Vector/DB Consistency — diagnostic helpers work correctly (Fix 7)
  6. Schema Idempotency    — schema loads twice without error
"""

import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models import Episode, Fact, MemoryType, Summary
from src.storage import Database


# ---------------------------------------------------------------------------
# Shared fixtures (no FAISS required)
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def database(temp_dir):
    """Create a test database."""
    return Database(temp_dir / "test.db")


# ---------------------------------------------------------------------------
# Test 1 — Provenance Integrity (upsert survival)
# Verifies Fix 1: ON CONFLICT DO UPDATE preserves provenance FK rows.
# ---------------------------------------------------------------------------


class TestProvenanceIntegrity:
    """Verify that re-saving (upserting) records does NOT destroy provenance links."""

    def test_resave_fact_preserves_episode_facts_link(self, database):
        """Re-saving a fact with the same ID must keep episode_facts rows."""
        # Create an episode that the fact will reference
        episode = Episode(
            raw_input="I started a new job",
            content="Started new job at Google",
            memory_type=MemoryType.EPISODIC,
            topics=["career"],
        )
        database.save_episode(episode)

        # Create a fact linked to that episode
        fact = Fact(
            content="User works at Google",
            category="personal",
            topic="career",
            confidence=0.9,
        )
        database.save_fact(fact, source_episode_ids=[episode.id])

        # Verify the link exists
        with database._connection() as conn:
            row = conn.execute(
                "SELECT * FROM episode_facts WHERE fact_id = ? AND episode_id = ?",
                (fact.id, episode.id),
            ).fetchone()
        assert row is not None, "episode_facts link should exist after initial save"

        # Re-save the same fact with updated content (simulates upsert)
        fact.content = "User works at Google as a senior engineer"
        database.save_fact(fact, source_episode_ids=[episode.id])

        # The link row must still exist
        with database._connection() as conn:
            row = conn.execute(
                "SELECT * FROM episode_facts WHERE fact_id = ? AND episode_id = ?",
                (fact.id, episode.id),
            ).fetchone()
        assert row is not None, "episode_facts link must survive upsert"

        # The content should be updated
        retrieved = database.get_fact(fact.id)
        assert retrieved is not None
        assert "senior engineer" in retrieved.content

    def test_resave_fact_preserves_created_at(self, database):
        """Re-saving a fact must preserve the original created_at timestamp."""
        fact = Fact(
            content="Original fact",
            category="personal",
            topic="test",
            confidence=0.8,
        )
        database.save_fact(fact)
        original = database.get_fact(fact.id)
        original_created_at = original.created_at

        # Re-save with a later updated_at
        fact.content = "Updated fact"
        fact.updated_at = datetime.now(timezone.utc)
        database.save_fact(fact)

        updated = database.get_fact(fact.id)
        assert updated.created_at == original_created_at
        assert updated.content == "Updated fact"

    def test_resave_summary_preserves_episode_summaries_link(self, database):
        """Re-saving a summary must keep episode_summaries rows."""
        # Create episodes
        ep1 = Episode(
            raw_input="Event 1",
            content="First event",
            memory_type=MemoryType.EPISODIC,
            topics=["test"],
        )
        ep2 = Episode(
            raw_input="Event 2",
            content="Second event",
            memory_type=MemoryType.EPISODIC,
            topics=["test"],
        )
        database.save_episode(ep1)
        database.save_episode(ep2)

        # Create a summary linked to both episodes
        summary = Summary(
            content="A summary of events",
            topic="test",
            time_start=datetime.now(timezone.utc),
            time_end=datetime.now(timezone.utc),
            episode_count=2,
        )
        database.save_summary(
            summary,
            source_episode_ids=[ep1.id, ep2.id],
            key_episode_ids=[ep1.id],
        )

        # Verify links
        with database._connection() as conn:
            links = conn.execute(
                "SELECT * FROM episode_summaries WHERE summary_id = ?",
                (summary.id,),
            ).fetchall()
        assert len(links) == 2, "Both episode_summaries links should exist"

        # Re-save the summary with updated content
        summary.content = "Updated summary"
        database.save_summary(summary)

        # Links must survive the upsert
        with database._connection() as conn:
            links = conn.execute(
                "SELECT * FROM episode_summaries WHERE summary_id = ?",
                (summary.id,),
            ).fetchall()
        assert len(links) == 2, "episode_summaries links must survive upsert"

        # Content should be updated
        retrieved = database.get_summary(summary.id)
        assert retrieved.content == "Updated summary"

    def test_resave_episode_preserves_episode_facts_link(self, database):
        """Re-saving an episode must keep episode_facts rows pointing at it."""
        episode = Episode(
            raw_input="Learning Python",
            content="User is learning Python",
            memory_type=MemoryType.EPISODIC,
            topics=["programming"],
        )
        database.save_episode(episode)

        fact = Fact(
            content="User knows Python",
            category="knowledge",
            topic="programming",
            confidence=0.9,
        )
        database.save_fact(fact, source_episode_ids=[episode.id])

        # Re-save the episode with updated content
        episode.content = "User is learning advanced Python"
        database.save_episode(episode)

        # The fact → episode link must survive
        retrieved_fact = database.get_fact(fact.id)
        assert episode.id in retrieved_fact.source_episode_ids

    def test_multiple_episodes_linked_to_fact(self, database):
        """A fact with multiple source episodes keeps all links after upsert."""
        episodes = []
        for i in range(3):
            ep = Episode(
                raw_input=f"Event {i}",
                content=f"Content {i}",
                memory_type=MemoryType.EPISODIC,
                topics=["multi"],
            )
            database.save_episode(ep)
            episodes.append(ep)

        fact = Fact(
            content="Multi-sourced fact",
            category="knowledge",
            topic="multi",
        )
        database.save_fact(fact, source_episode_ids=[ep.id for ep in episodes])

        # Verify all 3 links exist
        retrieved = database.get_fact(fact.id)
        assert len(retrieved.source_episode_ids) == 3

        # Re-save fact
        fact.confidence = 0.95
        database.save_fact(fact)

        # All 3 links must survive
        retrieved = database.get_fact(fact.id)
        assert len(retrieved.source_episode_ids) == 3


# ---------------------------------------------------------------------------
# Test 2 — CLI Import Smoke Test
# Verifies Fix 4: f-string syntax error in ingest.py is fixed.
# ---------------------------------------------------------------------------


class TestCLIImportSmoke:
    """Verify CLI modules import without SyntaxError."""

    def test_import_src_cli(self):
        """Importing src.cli must not crash."""
        import src.cli  # noqa: F401

    def test_import_src_cli_commands(self):
        """Importing src.cli.commands must not crash."""
        import src.cli.commands  # noqa: F401

    def test_import_ingest_command(self):
        """Importing the ingest command module must not crash (was SyntaxError)."""
        import src.cli.commands.ingest  # noqa: F401

    def test_cli_group_has_expected_commands(self):
        """The CLI group should register core commands."""
        from src.cli import cli

        # cli is a Click group — check command names
        command_names = set(cli.commands.keys()) if hasattr(cli, "commands") else set()
        expected = {"ingest", "query", "stats"}
        missing = expected - command_names
        assert not missing, f"CLI group is missing commands: {missing}"


# ---------------------------------------------------------------------------
# Test 3 — Consolidation End-to-End with Mock LLM
# Verifies Fix 2 (contradiction handling) and Fix 5 (supersession guard).
# Requires FAISS.
# ---------------------------------------------------------------------------


@pytest.mark.requires_faiss
class TestConsolidationEndToEnd:
    """Full consolidation flow: ingest → consolidate → verify."""

    @pytest.fixture
    def components(self, temp_dir):
        """Bootstrap all components needed for consolidation."""
        from src.embeddings import get_embedding_provider
        from src.llm import get_llm_provider
        from src.storage import VectorStore

        db = Database(temp_dir / "test.db")
        vs = VectorStore(temp_dir / "test.faiss", dimension=384)
        emb = get_embedding_provider("mock", dimension=384)
        llm = get_llm_provider("mock")
        return db, vs, emb, llm

    def test_consolidate_topic_processes_episodes(self, components):
        """Consolidation should process episodes, create summary, mark as consolidated."""
        db, vs, emb, llm = components
        from src.consolidation import ConsolidationPipeline
        from src.ingestion import IngestionPipeline

        ingestion = IngestionPipeline(db, vs, emb, llm)

        # Ingest 5+ episodes (mock LLM assigns topic "general")
        texts = [
            "Started learning Korean today",
            "Practiced Korean vocabulary for an hour",
            "Had my first Korean conversation",
            "Watched a Korean drama without subtitles",
            "Enrolled in a Korean class",
        ]
        for text in texts:
            result = ingestion.ingest(text, force=True)
            assert result.success, f"Failed to ingest: {text}"

        # Verify episodes are stored and unconsolidated
        stats = db.get_statistics()
        assert stats["total_episodes"] >= 5
        assert stats["unconsolidated_episodes"] >= 5

        # Run consolidation
        consolidator = ConsolidationPipeline(db, vs, emb, llm, episode_threshold=1)
        result = consolidator.consolidate_topic("general")

        # Verify result
        assert result.episodes_processed >= 5
        assert result.summaries_created == 1
        assert result.duration_seconds >= 0

        # Verify a summary was created with source episode links
        summaries = db.get_summaries(topic="general")
        assert len(summaries) >= 1
        summary = db.get_summary(summaries[0].id)
        assert len(summary.source_episode_ids) >= 5

        # Verify episodes are now marked consolidated
        unconsolidated = db.get_unconsolidated_episodes(topic="general")
        assert len(unconsolidated) == 0

    def test_consolidate_empty_topic_returns_zero(self, components):
        """Consolidating a topic with no episodes returns a zero-count result."""
        db, vs, emb, llm = components
        from src.consolidation import ConsolidationPipeline

        consolidator = ConsolidationPipeline(db, vs, emb, llm)
        result = consolidator.consolidate_topic("nonexistent")

        assert result.episodes_processed == 0
        assert result.summaries_created == 0
        assert result.facts_extracted == 0

    def test_consolidate_does_not_reprocess_episodes(self, components):
        """A second consolidation of the same topic processes zero episodes."""
        db, vs, emb, llm = components
        from src.consolidation import ConsolidationPipeline
        from src.ingestion import IngestionPipeline

        ingestion = IngestionPipeline(db, vs, emb, llm)
        for i in range(5):
            ingestion.ingest(f"Learning event number {i}", force=True)

        consolidator = ConsolidationPipeline(db, vs, emb, llm, episode_threshold=1)

        # First consolidation
        result1 = consolidator.consolidate_topic("general")
        assert result1.episodes_processed >= 5

        # Second consolidation — should find nothing to process
        result2 = consolidator.consolidate_topic("general")
        assert result2.episodes_processed == 0


# ---------------------------------------------------------------------------
# Test 4 — Inactive Filtering in Retrieval
# Verifies Fix 6: deactivated records excluded from semantic search.
# Requires FAISS.
# ---------------------------------------------------------------------------


@pytest.mark.requires_faiss
class TestInactiveFiltering:
    """Deactivated records must be excluded from semantic search results."""

    @pytest.fixture
    def search_components(self, temp_dir):
        """Bootstrap components for retrieval tests."""
        from src.embeddings import get_embedding_provider
        from src.llm import get_llm_provider
        from src.retrieval.semantic import SemanticRetriever
        from src.storage import VectorStore

        db = Database(temp_dir / "test.db")
        vs = VectorStore(temp_dir / "test.faiss", dimension=384)
        emb = get_embedding_provider("mock", dimension=384)
        llm = get_llm_provider("mock")
        retriever = SemanticRetriever(db, vs, emb, top_k=10, similarity_threshold=-1.0)
        return db, vs, emb, llm, retriever

    def test_deactivated_episode_excluded_from_search(self, search_components):
        """An episode that is deactivated should not appear in semantic search."""
        db, vs, emb, _llm, retriever = search_components
        from src.ingestion import IngestionPipeline

        llm = _llm
        ingestion = IngestionPipeline(db, vs, emb, llm)

        # Ingest an episode
        result = ingestion.ingest("I am learning Korean today", force=True)
        assert result.success
        episode_id = result.episode.id

        # Search — should find it
        sem_result = retriever.search("Korean learning")
        found_ids = {ep.id for ep in sem_result.episodes}
        assert episode_id in found_ids, "Active episode should appear in search"

        # Deactivate
        db.set_episode_active(episode_id, False)

        # Search again — should NOT find it
        sem_result = retriever.search("Korean learning")
        found_ids = {ep.id for ep in sem_result.episodes}
        assert episode_id not in found_ids, "Deactivated episode must be excluded"

    def test_reactivated_episode_appears_in_search(self, search_components):
        """An episode that is reactivated should appear in search again."""
        db, vs, emb, llm, retriever = search_components
        from src.ingestion import IngestionPipeline

        ingestion = IngestionPipeline(db, vs, emb, llm)
        result = ingestion.ingest("I moved to Seoul last month", force=True)
        assert result.success
        episode_id = result.episode.id

        # Deactivate then reactivate
        db.set_episode_active(episode_id, False)
        db.set_episode_active(episode_id, True)

        # Should find it again
        sem_result = retriever.search("Seoul")
        found_ids = {ep.id for ep in sem_result.episodes}
        assert episode_id in found_ids, "Reactivated episode should appear in search"

    def test_deactivated_fact_excluded_from_search(self, search_components):
        """A deactivated fact should not appear in fact search results."""
        db, vs, emb, _llm, retriever = search_components

        # Create and index a fact directly
        fact = Fact(
            content="User speaks Korean fluently",
            category="knowledge",
            topic="languages",
            confidence=0.9,
        )
        db.save_fact(fact)
        embedding = emb.embed_text(fact.to_embedding_text())
        embedding_id = vs.add("facts", fact.id, embedding)
        db.update_embedding_id("facts", fact.id, embedding_id)

        # Search for it — should find it
        sem_result = retriever.search("Korean language skills", search_episodes=False)
        found_ids = {f.id for f in sem_result.facts}
        assert fact.id in found_ids, "Active fact should appear in search"

        # Deactivate
        db.set_fact_active(fact.id, False)

        # Search again — should NOT find it
        sem_result = retriever.search("Korean language skills", search_episodes=False)
        found_ids = {f.id for f in sem_result.facts}
        assert fact.id not in found_ids, "Deactivated fact must be excluded"


# ---------------------------------------------------------------------------
# Test 5 — Vector/DB Consistency Check
# Verifies Fix 7: get_active_record_ids and get_indexed_ids work correctly.
# Requires FAISS.
# ---------------------------------------------------------------------------


@pytest.mark.requires_faiss
class TestVectorDBConsistency:
    """Diagnostic methods for checking DB ↔ FAISS consistency."""

    @pytest.fixture
    def consistency_components(self, temp_dir):
        """Bootstrap components for consistency tests."""
        from src.embeddings import get_embedding_provider
        from src.llm import get_llm_provider
        from src.storage import VectorStore

        db = Database(temp_dir / "test.db")
        vs = VectorStore(temp_dir / "test.faiss", dimension=384)
        emb = get_embedding_provider("mock", dimension=384)
        llm = get_llm_provider("mock")
        return db, vs, emb, llm

    def test_consistent_after_ingestion(self, consistency_components):
        """After ingesting episodes, DB active IDs should match vector store IDs."""
        db, vs, emb, llm = consistency_components
        from src.ingestion import IngestionPipeline

        ingestion = IngestionPipeline(db, vs, emb, llm)

        for i in range(3):
            result = ingestion.ingest(f"Memory event number {i}", force=True)
            assert result.success

        db_ids = db.get_active_record_ids("episodes")
        vs_ids = vs.get_indexed_ids("episodes")

        assert db_ids == vs_ids, f"DB active IDs {db_ids} != VS indexed IDs {vs_ids}"

    def test_detects_divergence_after_deactivation(self, consistency_components):
        """Deactivating an episode in DB should cause divergence with vector store."""
        db, vs, emb, llm = consistency_components
        from src.ingestion import IngestionPipeline

        ingestion = IngestionPipeline(db, vs, emb, llm)
        result = ingestion.ingest("A memory to deactivate", force=True)
        assert result.success
        episode_id = result.episode.id

        # Before deactivation — consistent
        db_ids = db.get_active_record_ids("episodes")
        vs_ids = vs.get_indexed_ids("episodes")
        assert db_ids == vs_ids

        # Deactivate in DB only
        db.set_episode_active(episode_id, False)

        # Now divergent: vector store still has the ID, DB doesn't report it
        db_ids = db.get_active_record_ids("episodes")
        vs_ids = vs.get_indexed_ids("episodes")
        assert episode_id not in db_ids
        assert episode_id in vs_ids
        assert db_ids != vs_ids

    def test_get_active_record_ids_invalid_table_raises(self, consistency_components):
        """Passing an invalid table name should raise ValueError."""
        db, _vs, _emb, _llm = consistency_components

        with pytest.raises(ValueError, match="Invalid table"):
            db.get_active_record_ids("nonexistent")

    def test_get_indexed_ids_invalid_index_raises(self, consistency_components):
        """Passing an invalid index name should raise ValueError."""
        _db, vs, _emb, _llm = consistency_components

        with pytest.raises(ValueError, match="Unknown index"):
            vs.get_indexed_ids("nonexistent")

    def test_empty_stores_are_consistent(self, consistency_components):
        """Empty DB and vector store should both return empty sets."""
        db, vs, _emb, _llm = consistency_components

        for table in ["episodes", "facts", "summaries"]:
            db_ids = db.get_active_record_ids(table)
            vs_ids = vs.get_indexed_ids(table)
            assert db_ids == vs_ids == set()


# ---------------------------------------------------------------------------
# Test 6 — Schema Loading / Idempotency
# Verifies that schema.sql creates all tables and can be loaded twice.
# ---------------------------------------------------------------------------


class TestSchemaIdempotency:
    """Schema creation and double-initialization safety."""

    def test_all_expected_tables_created(self, database):
        """The schema should create all core and provenance tables."""
        expected_tables = {
            "episodes",
            "facts",
            "summaries",
            "episode_facts",
            "episode_summaries",
            "summary_facts",
            "consolidation_runs",
            "topics",
        }

        with database._connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            actual_tables = {row["name"] for row in cursor.fetchall()}

        missing = expected_tables - actual_tables
        assert not missing, f"Missing tables: {missing}"

    def test_expected_views_created(self, database):
        """The schema should create the analytical views."""
        expected_views = {
            "v_active_episodes",
            "v_facts_with_sources",
            "v_summaries_detailed",
        }

        with database._connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
            actual_views = {row["name"] for row in cursor.fetchall()}

        missing = expected_views - actual_views
        assert not missing, f"Missing views: {missing}"

    def test_foreign_keys_enabled(self, database):
        """PRAGMA foreign_keys should be ON."""
        with database._connection() as conn:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1, "Foreign keys must be enabled"

    def test_double_initialization_is_safe(self, temp_dir):
        """Creating two Database instances on the same file must not crash."""
        db_path = temp_dir / "double.db"
        db1 = Database(db_path)
        db2 = Database(db_path)  # Should not raise

        # Both should work
        episode = Episode(
            raw_input="Test",
            content="Test content",
            memory_type=MemoryType.EPISODIC,
            topics=["test"],
        )
        db1.save_episode(episode)
        retrieved = db2.get_episode(episode.id)
        assert retrieved is not None
        assert retrieved.id == episode.id

    def test_cascade_deletes_exist_on_episode_facts(self, database):
        """Deleting an episode should cascade-delete episode_facts rows.

        Note: Our upsert fix (ON CONFLICT DO UPDATE) avoids triggering this,
        but the schema constraint itself must still be correct.
        """
        episode = Episode(
            raw_input="Cascade test",
            content="Testing cascades",
            memory_type=MemoryType.EPISODIC,
            topics=["test"],
        )
        database.save_episode(episode)

        fact = Fact(
            content="Cascade fact",
            category="personal",
            topic="test",
        )
        database.save_fact(fact, source_episode_ids=[episode.id])

        # Verify link exists
        with database._connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM episode_facts WHERE episode_id = ?",
                (episode.id,),
            ).fetchone()[0]
        assert count == 1

        # Hard-delete the episode (not what production code does, but tests CASCADE)
        with database._connection() as conn:
            conn.execute("DELETE FROM episodes WHERE id = ?", (episode.id,))

        # The link row should be gone due to ON DELETE CASCADE
        with database._connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM episode_facts WHERE episode_id = ?",
                (episode.id,),
            ).fetchone()[0]
        assert count == 0, "CASCADE delete should remove episode_facts rows"

    def test_set_fact_active_method(self, database):
        """set_fact_active should toggle the is_active flag."""
        fact = Fact(
            content="Toggle me",
            category="personal",
            topic="test",
        )
        database.save_fact(fact)

        # Should start active
        retrieved = database.get_fact(fact.id)
        assert retrieved.is_active is True

        # Deactivate
        database.set_fact_active(fact.id, False)
        retrieved = database.get_fact(fact.id)
        assert retrieved.is_active is False

        # Reactivate
        database.set_fact_active(fact.id, True)
        retrieved = database.get_fact(fact.id)
        assert retrieved.is_active is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
