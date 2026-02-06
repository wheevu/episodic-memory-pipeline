"""
SQLite database layer for structured memory storage.

Handles persistence of Episodes, Facts, Summaries, and their relationships.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from ..models import Episode, Fact, Summary


class Database:
    """SQLite database manager for the memory pipeline."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the database and ensure the schema exists.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._ensure_directory()
        self._initialize_schema()

    def _ensure_directory(self) -> None:
        """Ensure the database parent directory exists.

        Returns:
            None.
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            import logging

            logging.getLogger(__name__).error("Failed to create database directory: %s", e)
            raise

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection with foreign keys enabled.

        This centralizes transaction handling: commit on success, rollback on error.

        Yields:
            An open `sqlite3.Connection`.

        Raises:
            Exception: Re-raises any exception encountered within the context after rollback.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        """Initialize the database schema from the repository's `schema.sql`.

        Returns:
            None.

        Raises:
            FileNotFoundError: If schema.sql cannot be found.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Try multiple locations for schema.sql
        search_paths = [
            # Relative to this file (src/storage/)
            Path(__file__).parent.parent.parent / "schema.sql",
            # Relative to package root (when installed)
            Path(__file__).parent.parent / "schema.sql",
            # Relative to current working directory
            Path.cwd() / "schema.sql",
            # From package resources if available
        ]

        schema_path = None
        for path in search_paths:
            if path.exists():
                schema_path = path
                logger.debug("Found schema.sql at: %s", path)
                break

        # Try importlib.resources as fallback for installed packages
        if schema_path is None:
            try:
                import importlib.resources as pkg_resources

                # Try to load from package data
                try:
                    schema_text = pkg_resources.read_text("episodic_memory_pipeline", "schema.sql")
                    with self._connection() as conn:
                        conn.executescript(schema_text)
                    logger.info("Loaded schema from package resources")
                    return
                except (FileNotFoundError, ModuleNotFoundError):
                    pass
            except ImportError:
                pass

        if schema_path is None:
            error_msg = (
                "Could not find schema.sql. Searched in: "
                + ", ".join(str(p) for p in search_paths)
                + ". This is required to initialize the database."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(schema_path) as f:
                schema_sql = f.read()

            with self._connection() as conn:
                conn.executescript(schema_sql)

            logger.info("Database schema initialized from: %s", schema_path)
        except Exception as e:
            logger.error("Failed to initialize database schema: %s", e)
            raise

    def _apply_topic_delta(self, conn: sqlite3.Connection, topic: str, delta: int) -> None:
        """Apply an increment/decrement to a topic's episode count.

        Args:
            conn: Active SQLite connection.
            topic: Topic name to update.
            delta: Positive or negative increment.

        Returns:
            None.
        """
        if delta == 0:
            return
        if delta > 0:
            conn.execute(
                """
                INSERT INTO topics (name, episode_count)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    episode_count = episode_count + ?
                """,
                (topic, delta, delta),
            )
        else:
            conn.execute(
                "UPDATE topics SET episode_count = episode_count + ? WHERE name = ?", (delta, topic)
            )
            conn.execute("DELETE FROM topics WHERE name = ? AND episode_count <= 0", (topic,))

    def _reconcile_topic_counts(
        self,
        conn: sqlite3.Connection,
        previous_topics: set[str],
        new_topics: set[str],
        previous_active: bool,
        new_active: bool,
    ) -> None:
        """Reconcile topic counts when an episode changes.

        Args:
            conn: Active SQLite connection.
            previous_topics: Topics before the change.
            new_topics: Topics after the change.
            previous_active: Previous active state.
            new_active: New active state.

        Returns:
            None.
        """
        if previous_active and not new_active:
            removed = previous_topics
            added = set()
        elif not previous_active and new_active:
            removed = set()
            added = new_topics
        elif not previous_active and not new_active:
            removed = set()
            added = set()
        else:
            removed = previous_topics - new_topics
            added = new_topics - previous_topics

        for topic in added:
            self._apply_topic_delta(conn, topic, 1)
        for topic in removed:
            self._apply_topic_delta(conn, topic, -1)

    # =========================================================================
    # Episode Operations
    # =========================================================================

    def save_episode(self, episode: Episode) -> str:
        """Save an episode, inserting if new or updating mutable fields if it exists.

        Uses INSERT ... ON CONFLICT to preserve the row identity and avoid
        cascading deletes on provenance tables (episode_facts, episode_summaries).

        Args:
            episode: Episode to persist.

        Returns:
            The episode's ID.
        """
        row = episode.to_db_row()

        with self._connection() as conn:
            existing = conn.execute(
                "SELECT topics, is_active FROM episodes WHERE id = ?", (episode.id,)
            ).fetchone()
            previous_topics = set()
            previous_active = False
            if existing:
                stored_topics = existing["topics"]
                if isinstance(stored_topics, str):
                    stored_topics = json.loads(stored_topics)
                previous_topics = set(stored_topics or [])
                previous_active = bool(existing["is_active"])

            conn.execute(
                """
                INSERT INTO episodes (
                    id, created_at, occurred_at, raw_input, content,
                    memory_type, topics, entities, confidence, importance,
                    source, session_id, is_active, consolidated, embedding_id
                ) VALUES (
                    :id, :created_at, :occurred_at, :raw_input, :content,
                    :memory_type, :topics, :entities, :confidence, :importance,
                    :source, :session_id, :is_active, :consolidated, :embedding_id
                )
                ON CONFLICT(id) DO UPDATE SET
                    occurred_at = excluded.occurred_at,
                    raw_input = excluded.raw_input,
                    content = excluded.content,
                    memory_type = excluded.memory_type,
                    topics = excluded.topics,
                    entities = excluded.entities,
                    confidence = excluded.confidence,
                    importance = excluded.importance,
                    source = excluded.source,
                    session_id = excluded.session_id,
                    is_active = excluded.is_active,
                    consolidated = excluded.consolidated,
                    embedding_id = excluded.embedding_id
            """,
                row,
            )

            new_topics = set(episode.topics or [])
            new_active = bool(episode.is_active)
            self._reconcile_topic_counts(
                conn,
                previous_topics,
                new_topics,
                previous_active,
                new_active,
            )

        return episode.id

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID.

        Args:
            episode_id: Episode identifier to look up.

        Returns:
            The episode if found; otherwise None.
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
            row = cursor.fetchone()

            if row:
                return Episode.from_db_row(dict(row))
            return None

    def get_episodes_by_ids(self, episode_ids: list[str]) -> dict[str, Episode]:
        """Retrieve multiple episodes by ID in a single query.

        Args:
            episode_ids: Episode identifiers to look up.

        Returns:
            A dictionary mapping episode ID to Episode for found records.
        """
        if not episode_ids:
            return {}

        placeholders = ",".join("?" * len(episode_ids))
        with self._connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM episodes WHERE id IN ({placeholders})", episode_ids
            )
            return {row["id"]: Episode.from_db_row(dict(row)) for row in cursor.fetchall()}

    def get_episodes(
        self,
        topic: Optional[str] = None,
        memory_type: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        consolidated: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Episode]:
        """Query episodes with optional filters.

        Args:
            topic: Filter by topic using a substring match in the stored JSON array.
            memory_type: Filter by memory type.
            since: Only include episodes occurring at/after this time.
            until: Only include episodes occurring at/before this time.
            consolidated: Filter by consolidation status if provided.
            limit: Maximum number of results to return.
            offset: Pagination offset.

        Returns:
            A list of matching episodes ordered by `occurred_at` descending.
        """
        conditions = ["is_active = TRUE"]
        params: list[object] = []

        if topic:
            conditions.append(
                "EXISTS (SELECT 1 FROM json_each(episodes.topics) WHERE json_each.value = ?)"
            )
            params.append(topic)

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        if since:
            conditions.append("occurred_at >= ?")
            params.append(since.isoformat())

        if until:
            conditions.append("occurred_at <= ?")
            params.append(until.isoformat())

        if consolidated is not None:
            conditions.append("consolidated = ?")
            params.append(consolidated)

        query = f"""
            SELECT * FROM episodes
            WHERE {" AND ".join(conditions)}
            ORDER BY occurred_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return [Episode.from_db_row(dict(row)) for row in cursor.fetchall()]

    def set_episode_active(self, episode_id: str, is_active: bool) -> None:
        """Set an episode's active flag and reconcile topic counts.

        Args:
            episode_id: Episode ID to update.
            is_active: New active state.

        Returns:
            None.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT topics, is_active FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
            if not row:
                return

            current_active = bool(row["is_active"])
            if current_active == is_active:
                return

            stored_topics = row["topics"]
            if isinstance(stored_topics, str):
                stored_topics = json.loads(stored_topics)
            topics = set(stored_topics or [])

            conn.execute("UPDATE episodes SET is_active = ? WHERE id = ?", (is_active, episode_id))
            self._reconcile_topic_counts(
                conn,
                topics,
                topics,
                current_active,
                is_active,
            )

    def mark_episodes_consolidated(self, episode_ids: list[str]) -> None:
        """Mark episodes as consolidated in bulk.

        Args:
            episode_ids: Episode IDs to mark as consolidated.

        Returns:
            None.
        """
        if not episode_ids:
            return

        placeholders = ",".join("?" * len(episode_ids))
        with self._connection() as conn:
            conn.execute(
                f"UPDATE episodes SET consolidated = TRUE WHERE id IN ({placeholders})", episode_ids
            )

    def get_unconsolidated_episodes(
        self, topic: Optional[str] = None, limit: int = 100
    ) -> list[Episode]:
        """Return episodes that have not yet been consolidated.

        Args:
            topic: Optional topic filter.
            limit: Maximum number of episodes to return.

        Returns:
            A list of unconsolidated episodes.
        """
        return self.get_episodes(topic=topic, consolidated=False, limit=limit)

    # =========================================================================
    # Fact Operations
    # =========================================================================

    def save_fact(self, fact: Fact, source_episode_ids: Optional[list[str]] = None) -> str:
        """Save a fact, inserting if new or updating mutable fields if it exists.

        Uses INSERT ... ON CONFLICT to preserve the row identity and avoid
        cascading deletes on provenance tables (episode_facts, summary_facts).

        Args:
            fact: Fact to persist.
            source_episode_ids: Episode IDs that support this fact.

        Returns:
            The fact's ID.
        """
        row = fact.to_db_row()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO facts (
                    id, created_at, updated_at, content, category,
                    topic, entities, confidence, valid_from, valid_until,
                    is_active, superseded_by, embedding_id
                ) VALUES (
                    :id, :created_at, :updated_at, :content, :category,
                    :topic, :entities, :confidence, :valid_from, :valid_until,
                    :is_active, :superseded_by, :embedding_id
                )
                ON CONFLICT(id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    content = excluded.content,
                    category = excluded.category,
                    topic = excluded.topic,
                    entities = excluded.entities,
                    confidence = excluded.confidence,
                    valid_from = excluded.valid_from,
                    valid_until = excluded.valid_until,
                    is_active = excluded.is_active,
                    superseded_by = excluded.superseded_by,
                    embedding_id = excluded.embedding_id
            """,
                row,
            )

            # Link to source episodes
            if source_episode_ids:
                for episode_id in source_episode_ids:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO episode_facts (episode_id, fact_id, relationship)
                        VALUES (?, ?, 'source')
                    """,
                        (episode_id, fact.id),
                    )

        return fact.id

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by ID (including its source episode links).

        Args:
            fact_id: Fact identifier to look up.

        Returns:
            The fact if found; otherwise None.
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
            row = cursor.fetchone()

            if row:
                fact = Fact.from_db_row(dict(row))
                # Load source episode IDs
                cursor = conn.execute(
                    "SELECT episode_id FROM episode_facts WHERE fact_id = ?", (fact_id,)
                )
                fact.source_episode_ids = [r["episode_id"] for r in cursor.fetchall()]
                return fact
            return None

    def get_facts_by_ids(self, fact_ids: list[str]) -> dict[str, Fact]:
        """Retrieve multiple facts by ID in a single query.

        Source episode links are loaded in a single batch query.

        Args:
            fact_ids: Fact identifiers to look up.

        Returns:
            A dictionary mapping fact ID to Fact for found records.
        """
        if not fact_ids:
            return {}

        placeholders = ",".join("?" * len(fact_ids))
        with self._connection() as conn:
            cursor = conn.execute(f"SELECT * FROM facts WHERE id IN ({placeholders})", fact_ids)
            facts = {row["id"]: Fact.from_db_row(dict(row)) for row in cursor.fetchall()}

            # Batch-load source episode links
            if facts:
                cursor = conn.execute(
                    f"SELECT fact_id, episode_id FROM episode_facts WHERE fact_id IN ({placeholders})",
                    list(facts.keys()),
                )
                for row in cursor.fetchall():
                    if row["fact_id"] in facts:
                        facts[row["fact_id"]].source_episode_ids.append(row["episode_id"])

            return facts

    def get_facts(
        self,
        topic: Optional[str] = None,
        category: Optional[str] = None,
        current_only: bool = True,
        limit: int = 100,
    ) -> list[Fact]:
        """Query facts with optional filters.

        Args:
            topic: Optional topic filter.
            category: Optional category filter.
            current_only: If True, exclude expired or superseded facts.
            limit: Maximum number of facts to return.

        Returns:
            A list of matching facts ordered by `updated_at` descending.
        """
        conditions = ["is_active = TRUE"]
        params: list[object] = []

        if topic:
            conditions.append("topic = ?")
            params.append(topic)

        if category:
            conditions.append("category = ?")
            params.append(category)

        if current_only:
            conditions.append("(valid_until IS NULL OR valid_until > ?)")
            params.append(datetime.now(timezone.utc).isoformat())
            conditions.append("superseded_by IS NULL")

        query = f"""
            SELECT * FROM facts
            WHERE {" AND ".join(conditions)}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return [Fact.from_db_row(dict(row)) for row in cursor.fetchall()]

    def find_similar_facts(self, content: str, topic: str) -> list[Fact]:
        """Return a small set of recent facts for duplicate checking within a topic.

        This is intentionally a lightweight heuristic; semantic similarity is handled
        by the vector store, while this supports quick "nearby candidates" lookups.

        Args:
            content: Fact text content (currently unused by this heuristic).
            topic: Topic to constrain the search.

        Returns:
            A list of recent facts in the same topic.
        """
        # Simple text-based similarity check (vector search handles semantic)
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM facts
                WHERE topic = ? AND is_active = TRUE
                ORDER BY updated_at DESC
                LIMIT 10
            """,
                (topic,),
            )
            return [Fact.from_db_row(dict(row)) for row in cursor.fetchall()]

    def supersede_fact(self, old_fact_id: str, new_fact: Fact) -> None:
        """Mark a fact as superseded by another fact.

        Args:
            old_fact_id: ID of the fact being superseded.
            new_fact: The replacing fact.

        Returns:
            None.
        """
        with self._connection() as conn:
            conn.execute(
                "UPDATE facts SET superseded_by = ?, is_active = FALSE WHERE id = ?",
                (new_fact.id, old_fact_id),
            )

    def set_fact_active(self, fact_id: str, is_active: bool) -> None:
        """Set a fact's active flag.

        Args:
            fact_id: Fact ID to update.
            is_active: New active state.

        Returns:
            None.
        """
        with self._connection() as conn:
            conn.execute(
                "UPDATE facts SET is_active = ? WHERE id = ?",
                (is_active, fact_id),
            )

    # =========================================================================
    # Summary Operations
    # =========================================================================

    def save_summary(
        self,
        summary: Summary,
        source_episode_ids: Optional[list[str]] = None,
        key_episode_ids: Optional[list[str]] = None,
    ) -> str:
        """Save a summary, inserting if new or updating mutable fields if it exists.

        Uses INSERT ... ON CONFLICT to preserve the row identity and avoid
        cascading deletes on provenance tables (episode_summaries, summary_facts).

        Args:
            summary: Summary to persist.
            source_episode_ids: Episode IDs that contributed to this summary.
            key_episode_ids: Subset of source episode IDs marked as key events.

        Returns:
            The summary's ID.
        """
        row = summary.to_db_row()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO summaries (
                    id, created_at, updated_at, content, topic,
                    time_start, time_end, episode_count, key_events,
                    parent_summary_id, summary_level, is_active, embedding_id
                ) VALUES (
                    :id, :created_at, :updated_at, :content, :topic,
                    :time_start, :time_end, :episode_count, :key_events,
                    :parent_summary_id, :summary_level, :is_active, :embedding_id
                )
                ON CONFLICT(id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    content = excluded.content,
                    topic = excluded.topic,
                    time_start = excluded.time_start,
                    time_end = excluded.time_end,
                    episode_count = excluded.episode_count,
                    key_events = excluded.key_events,
                    parent_summary_id = excluded.parent_summary_id,
                    summary_level = excluded.summary_level,
                    is_active = excluded.is_active,
                    embedding_id = excluded.embedding_id
            """,
                row,
            )

            # Link to source episodes
            if source_episode_ids:
                key_set = set(key_episode_ids or [])
                for episode_id in source_episode_ids:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO episode_summaries
                        (episode_id, summary_id, is_key_event)
                        VALUES (?, ?, ?)
                    """,
                        (episode_id, summary.id, episode_id in key_set),
                    )

            # Update topic's last consolidation time
            conn.execute(
                """
                UPDATE topics SET last_consolidation = ?
                WHERE name = ?
            """,
                (datetime.now(timezone.utc).isoformat(), summary.topic),
            )

        return summary.id

    def get_summary(self, summary_id: str) -> Optional[Summary]:
        """Retrieve a summary by ID (including its source episode links).

        Args:
            summary_id: Summary identifier to look up.

        Returns:
            The summary if found; otherwise None.
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM summaries WHERE id = ?", (summary_id,))
            row = cursor.fetchone()

            if row:
                summary = Summary.from_db_row(dict(row))
                cursor = conn.execute(
                    "SELECT episode_id FROM episode_summaries WHERE summary_id = ?", (summary_id,)
                )
                summary.source_episode_ids = [r["episode_id"] for r in cursor.fetchall()]
                return summary
            return None

    def get_summaries_by_ids(self, summary_ids: list[str]) -> dict[str, Summary]:
        """Retrieve multiple summaries by ID in a single query.

        Source episode links are loaded in a single batch query.

        Args:
            summary_ids: Summary identifiers to look up.

        Returns:
            A dictionary mapping summary ID to Summary for found records.
        """
        if not summary_ids:
            return {}

        placeholders = ",".join("?" * len(summary_ids))
        with self._connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM summaries WHERE id IN ({placeholders})", summary_ids
            )
            summaries = {row["id"]: Summary.from_db_row(dict(row)) for row in cursor.fetchall()}

            # Batch-load source episode links
            if summaries:
                cursor = conn.execute(
                    f"SELECT summary_id, episode_id FROM episode_summaries WHERE summary_id IN ({placeholders})",
                    list(summaries.keys()),
                )
                for row in cursor.fetchall():
                    if row["summary_id"] in summaries:
                        summaries[row["summary_id"]].source_episode_ids.append(row["episode_id"])

            return summaries

    def get_summaries(
        self,
        topic: Optional[str] = None,
        level: Optional[int] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[Summary]:
        """Query summaries with optional filters.

        Args:
            topic: Optional topic filter.
            level: Optional summary level filter.
            since: Only include summaries ending at/after this time.
            limit: Maximum number of summaries to return.

        Returns:
            A list of matching summaries ordered by `time_end` descending.
        """
        conditions = ["is_active = TRUE"]
        params: list[object] = []

        if topic:
            conditions.append("topic = ?")
            params.append(topic)

        if level:
            conditions.append("summary_level = ?")
            params.append(level)

        if since:
            conditions.append("time_end >= ?")
            params.append(since.isoformat())

        query = f"""
            SELECT * FROM summaries
            WHERE {" AND ".join(conditions)}
            ORDER BY time_end DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return [Summary.from_db_row(dict(row)) for row in cursor.fetchall()]

    def get_latest_summary(self, topic: str) -> Optional[Summary]:
        """Return the most recent summary for a topic.

        Args:
            topic: Topic name to fetch the latest summary for.

        Returns:
            The latest summary if one exists; otherwise None.
        """
        summaries = self.get_summaries(topic=topic, limit=1)
        return summaries[0] if summaries else None

    # =========================================================================
    # Topic Operations
    # =========================================================================

    def get_topics(self) -> list[dict]:
        """Return all known topics with basic statistics.

        Returns:
            A list of topic dictionaries ordered by `episode_count` descending.
        """
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT name, description, episode_count, last_consolidation
                FROM topics
                ORDER BY episode_count DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_topics_needing_consolidation(
        self, min_episodes: int = 5, max_age_days: int = 7
    ) -> list[str]:
        """Find topics that should be consolidated based on episode volume or staleness.

        Criteria:
        - Has at least `min_episodes` unconsolidated episodes, OR
        - Has not been consolidated in `max_age_days`.

        Args:
            min_episodes: Minimum unconsolidated episode count to trigger consolidation.
            max_age_days: Maximum allowed days since last consolidation.

        Returns:
            A list of topic names needing consolidation.
        """
        with self._connection() as conn:
            # Topics with enough unconsolidated episodes
            cursor = conn.execute(
                """
                SELECT DISTINCT json_each.value as topic
                FROM episodes, json_each(episodes.topics)
                WHERE consolidated = FALSE AND is_active = TRUE
                GROUP BY json_each.value
                HAVING COUNT(*) >= ?
            """,
                (min_episodes,),
            )

            topics = {row["topic"] for row in cursor.fetchall()}

            # Topics that haven't been consolidated recently
            cursor = conn.execute(
                """
                SELECT name FROM topics
                WHERE last_consolidation IS NULL
                   OR last_consolidation < datetime('now', ? || ' days')
            """,
                (f"-{max_age_days}",),
            )

            topics.update(row["name"] for row in cursor.fetchall())

            return list(topics)

    # =========================================================================
    # Utility Operations
    # =========================================================================

    def update_embedding_id(self, table: str, record_id: str, embedding_id: int) -> None:
        """Update the `embedding_id` field for a record in the given table.

        Args:
            table: One of {"episodes", "facts", "summaries"}.
            record_id: Primary key of the record to update.
            embedding_id: FAISS internal ID to store.

        Returns:
            None.

        Raises:
            ValueError: If `table` is not one of the allowed table names.
        """
        valid_tables = {"episodes", "facts", "summaries"}
        if table not in valid_tables:
            raise ValueError(f"Invalid table: {table}")

        with self._connection() as conn:
            conn.execute(
                f"UPDATE {table} SET embedding_id = ? WHERE id = ?", (embedding_id, record_id)
            )

    def get_statistics(self) -> dict:
        """Return basic database statistics for monitoring/debugging.

        Returns:
            A dictionary of aggregate counts (episodes, facts, summaries, topics).
        """
        with self._connection() as conn:
            stats = {}

            cursor = conn.execute("SELECT COUNT(*) FROM episodes WHERE is_active = TRUE")
            stats["total_episodes"] = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE consolidated = FALSE AND is_active = TRUE"
            )
            stats["unconsolidated_episodes"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM facts WHERE is_active = TRUE")
            stats["total_facts"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM summaries WHERE is_active = TRUE")
            stats["total_summaries"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM topics")
            stats["total_topics"] = cursor.fetchone()[0]

            return stats

    def get_active_record_ids(self, table: str) -> set[str]:
        """Return the set of active record IDs for a given table.

        Useful for consistency checks against the vector store.

        Args:
            table: One of {"episodes", "facts", "summaries"}.

        Returns:
            A set of active record IDs.

        Raises:
            ValueError: If `table` is not one of the allowed table names.
        """
        valid_tables = {"episodes", "facts", "summaries"}
        if table not in valid_tables:
            raise ValueError(f"Invalid table: {table}")

        with self._connection() as conn:
            cursor = conn.execute(
                f"SELECT id FROM {table} WHERE is_active = TRUE AND embedding_id IS NOT NULL"
            )
            return {row["id"] for row in cursor.fetchall()}
