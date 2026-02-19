"""
LanceDB unified vector + metadata store for the episodic memory pipeline.

Replaces the separate SQLite (Database) and FAISS (VectorStore) layers with
a single LanceDB-backed store that handles both structured metadata and
vector similarity search.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)


def _lance():
    """Lazy import of lancedb to allow the module to be imported without the package installed."""
    try:
        import lancedb  # type: ignore[import]

        return lancedb
    except ImportError as err:
        raise ImportError("lancedb package required. Install with: pip install lancedb") from err


class LanceStore:
    """
    Unified LanceDB store for episodes, facts, and summaries.

    Provides both structured-metadata queries and native vector similarity
    search in a single backend, replacing the paired SQLite + FAISS design.
    """

    TABLE_EPISODES = "episodes"
    TABLE_FACTS = "facts"
    TABLE_SUMMARIES = "summaries"

    def __init__(self, db_path: Path, embedding_dimension: int) -> None:
        """Open (or create) a LanceDB database and ensure tables exist.

        Args:
            db_path: Directory path for the LanceDB database.
            embedding_dimension: Dimensionality expected for all embedding vectors.
        """
        self.db_path = db_path
        self.embedding_dimension = embedding_dimension
        db_path.mkdir(parents=True, exist_ok=True)
        lancedb = _lance()
        self._db = lancedb.connect(str(db_path))
        self._ensure_tables()

    # =========================================================================
    # Internal: schema and table bootstrapping
    # =========================================================================

    def _episode_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("occurred_at", pa.string()),
                pa.field("raw_input", pa.string()),
                pa.field("content", pa.string()),
                pa.field("memory_type", pa.string()),
                pa.field("topics", pa.list_(pa.string())),
                pa.field("entities", pa.list_(pa.string())),
                pa.field("confidence", pa.float32()),
                pa.field("importance", pa.float32()),
                pa.field("source", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("is_active", pa.bool_()),
                pa.field("consolidated", pa.bool_()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dimension)),
            ]
        )

    def _fact_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("content", pa.string()),
                pa.field("category", pa.string()),
                pa.field("topic", pa.string()),
                pa.field("entities", pa.list_(pa.string())),
                pa.field("confidence", pa.float32()),
                pa.field("valid_from", pa.string()),
                pa.field("valid_until", pa.string()),
                pa.field("is_active", pa.bool_()),
                pa.field("superseded_by", pa.string()),
                pa.field("source_episode_ids", pa.list_(pa.string())),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dimension)),
            ]
        )

    def _summary_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("content", pa.string()),
                pa.field("topic", pa.string()),
                pa.field("time_start", pa.string()),
                pa.field("time_end", pa.string()),
                pa.field("episode_count", pa.int32()),
                pa.field("key_events", pa.list_(pa.string())),
                pa.field("parent_summary_id", pa.string()),
                pa.field("summary_level", pa.int32()),
                pa.field("is_active", pa.bool_()),
                pa.field("source_episode_ids", pa.list_(pa.string())),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dimension)),
            ]
        )

    def _ensure_tables(self) -> None:
        """Create tables if they do not already exist."""
        existing = set(self._db.table_names())
        for name, schema in [
            (self.TABLE_EPISODES, self._episode_schema()),
            (self.TABLE_FACTS, self._fact_schema()),
            (self.TABLE_SUMMARIES, self._summary_schema()),
        ]:
            if name not in existing:
                empty = pa.table(
                    {field.name: pa.array([], type=field.type) for field in schema},
                    schema=schema,
                )
                self._db.create_table(name, data=empty, schema=schema)
                logger.debug("Created LanceDB table: %s", name)

    def _tbl(self, name: str):
        """Return an opened LanceDB table handle."""
        return self._db.open_table(name)

    # =========================================================================
    # Write helpers
    # =========================================================================

    @staticmethod
    def _norm(vec: np.ndarray) -> list[float]:
        """L2-normalise a vector and return as a Python list for storage."""
        vec = vec.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    # =========================================================================
    # Write: save / upsert
    # =========================================================================

    def save_episode(self, episode, vector: np.ndarray) -> str:
        """Atomically upsert an episode with its embedding vector.

        Args:
            episode: Episode model instance.
            vector: Embedding vector for this episode.

        Returns:
            The episode's ID.
        """
        tbl = self._tbl(self.TABLE_EPISODES)
        row = {
            "id": episode.id,
            "created_at": episode.created_at.isoformat(),
            "occurred_at": episode.occurred_at.isoformat(),
            "raw_input": episode.raw_input,
            "content": episode.content,
            "memory_type": str(episode.memory_type),
            "topics": list(episode.topics or []),
            "entities": list(episode.entities or []),
            "confidence": float(episode.confidence),
            "importance": float(episode.importance),
            "source": episode.source or "chat",
            "session_id": episode.session_id or "",
            "is_active": bool(episode.is_active),
            "consolidated": bool(episode.consolidated),
            "vector": self._norm(vector),
        }
        tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
            [row]
        )
        return episode.id

    def save_fact(
        self,
        fact,
        vector: np.ndarray,
        source_episode_ids: Optional[list[str]] = None,
    ) -> str:
        """Atomically upsert a fact with its embedding vector.

        Args:
            fact: Fact model instance.
            vector: Embedding vector for this fact.
            source_episode_ids: Episode IDs that support this fact.

        Returns:
            The fact's ID.
        """
        tbl = self._tbl(self.TABLE_FACTS)
        row = {
            "id": fact.id,
            "created_at": fact.created_at.isoformat(),
            "updated_at": fact.updated_at.isoformat(),
            "content": fact.content,
            "category": str(fact.category),
            "topic": fact.topic,
            "entities": list(fact.entities or []),
            "confidence": float(fact.confidence),
            "valid_from": fact.valid_from.isoformat() if fact.valid_from else "",
            "valid_until": fact.valid_until.isoformat() if fact.valid_until else "",
            "is_active": bool(fact.is_active),
            "superseded_by": fact.superseded_by or "",
            "source_episode_ids": list(source_episode_ids or fact.source_episode_ids or []),
            "vector": self._norm(vector),
        }
        tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
            [row]
        )
        return fact.id

    def save_summary(
        self,
        summary,
        vector: np.ndarray,
        source_episode_ids: Optional[list[str]] = None,
    ) -> str:
        """Atomically upsert a summary with its embedding vector.

        Args:
            summary: Summary model instance.
            vector: Embedding vector for this summary.
            source_episode_ids: Episode IDs that contributed to this summary.

        Returns:
            The summary's ID.
        """
        tbl = self._tbl(self.TABLE_SUMMARIES)
        row = {
            "id": summary.id,
            "created_at": summary.created_at.isoformat(),
            "updated_at": summary.updated_at.isoformat(),
            "content": summary.content,
            "topic": summary.topic,
            "time_start": summary.time_start.isoformat(),
            "time_end": summary.time_end.isoformat(),
            "episode_count": int(summary.episode_count),
            "key_events": list(summary.key_events or []),
            "parent_summary_id": summary.parent_summary_id or "",
            "summary_level": int(summary.summary_level),
            "is_active": bool(summary.is_active),
            "source_episode_ids": list(source_episode_ids or summary.source_episode_ids or []),
            "vector": self._norm(vector),
        }
        tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
            [row]
        )
        return summary.id

    # =========================================================================
    # Write: updates
    # =========================================================================

    def set_episode_active(self, episode_id: str, is_active: bool) -> None:
        """Toggle the is_active flag on an episode.

        Args:
            episode_id: Episode ID to update.
            is_active: New active state.
        """
        tbl = self._tbl(self.TABLE_EPISODES)
        tbl.update(where=f"id = '{episode_id}'", values={"is_active": is_active})

    def set_fact_active(self, fact_id: str, is_active: bool) -> None:
        """Toggle the is_active flag on a fact.

        Args:
            fact_id: Fact ID to update.
            is_active: New active state.
        """
        tbl = self._tbl(self.TABLE_FACTS)
        tbl.update(where=f"id = '{fact_id}'", values={"is_active": is_active})

    def mark_episodes_consolidated(self, episode_ids: list[str]) -> None:
        """Mark a list of episodes as consolidated.

        Args:
            episode_ids: Episode IDs to mark.
        """
        if not episode_ids:
            return
        tbl = self._tbl(self.TABLE_EPISODES)
        for episode_id in episode_ids:
            tbl.update(where=f"id = '{episode_id}'", values={"consolidated": True})

    def supersede_fact(self, old_fact_id: str, new_fact) -> None:
        """Mark a fact as superseded by a newer fact.

        Args:
            old_fact_id: ID of the fact being superseded.
            new_fact: The replacing Fact instance.
        """
        tbl = self._tbl(self.TABLE_FACTS)
        tbl.update(
            where=f"id = '{old_fact_id}'",
            values={"superseded_by": new_fact.id, "is_active": False},
        )

    # =========================================================================
    # Read helpers
    # =========================================================================

    @staticmethod
    def _parse_dt(val: Optional[str]) -> Optional[datetime]:
        """Parse an ISO-format datetime string or return None."""
        if not val:
            return None
        try:
            return datetime.fromisoformat(val)
        except (ValueError, TypeError):
            return None

    def _row_to_episode(self, row: dict):
        """Reconstruct an Episode from a LanceDB row dict."""
        from ..models.episode import Episode, MemoryType

        return Episode(
            id=row["id"],
            created_at=self._parse_dt(row["created_at"]) or datetime.now(timezone.utc),
            occurred_at=self._parse_dt(row["occurred_at"]) or datetime.now(timezone.utc),
            raw_input=row["raw_input"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            topics=list(row.get("topics") or []),
            entities=list(row.get("entities") or []),
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            source=row.get("source", "chat"),
            session_id=row.get("session_id") or None,
            is_active=bool(row["is_active"]),
            consolidated=bool(row["consolidated"]),
        )

    def _row_to_fact(self, row: dict):
        """Reconstruct a Fact from a LanceDB row dict."""
        from ..models.fact import Fact, FactCategory

        return Fact(
            id=row["id"],
            created_at=self._parse_dt(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=self._parse_dt(row["updated_at"]) or datetime.now(timezone.utc),
            content=row["content"],
            category=FactCategory(row["category"]),
            topic=row["topic"],
            entities=list(row.get("entities") or []),
            confidence=float(row["confidence"]),
            valid_from=self._parse_dt(row.get("valid_from")),
            valid_until=self._parse_dt(row.get("valid_until")),
            is_active=bool(row["is_active"]),
            superseded_by=row.get("superseded_by") or None,
            source_episode_ids=list(row.get("source_episode_ids") or []),
        )

    def _row_to_summary(self, row: dict):
        """Reconstruct a Summary from a LanceDB row dict."""
        from ..models.summary import Summary

        return Summary(
            id=row["id"],
            created_at=self._parse_dt(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=self._parse_dt(row["updated_at"]) or datetime.now(timezone.utc),
            content=row["content"],
            topic=row["topic"],
            time_start=self._parse_dt(row["time_start"]) or datetime.now(timezone.utc),
            time_end=self._parse_dt(row["time_end"]) or datetime.now(timezone.utc),
            episode_count=int(row["episode_count"]),
            key_events=list(row.get("key_events") or []),
            parent_summary_id=row.get("parent_summary_id") or None,
            summary_level=int(row["summary_level"]),
            is_active=bool(row["is_active"]),
            source_episode_ids=list(row.get("source_episode_ids") or []),
        )

    # =========================================================================
    # Read: single record
    # =========================================================================

    def get_episode(self, episode_id: str):
        """Retrieve an episode by ID.

        Args:
            episode_id: Episode identifier to look up.

        Returns:
            The Episode if found; otherwise None.
        """
        tbl = self._tbl(self.TABLE_EPISODES)
        rows = tbl.search().where(f"id = '{episode_id}'").limit(1).to_list()
        return self._row_to_episode(rows[0]) if rows else None

    def get_fact(self, fact_id: str):
        """Retrieve a fact by ID.

        Args:
            fact_id: Fact identifier to look up.

        Returns:
            The Fact if found; otherwise None.
        """
        tbl = self._tbl(self.TABLE_FACTS)
        rows = tbl.search().where(f"id = '{fact_id}'").limit(1).to_list()
        return self._row_to_fact(rows[0]) if rows else None

    def get_summary(self, summary_id: str):
        """Retrieve a summary by ID.

        Args:
            summary_id: Summary identifier to look up.

        Returns:
            The Summary if found; otherwise None.
        """
        tbl = self._tbl(self.TABLE_SUMMARIES)
        rows = tbl.search().where(f"id = '{summary_id}'").limit(1).to_list()
        return self._row_to_summary(rows[0]) if rows else None

    # =========================================================================
    # Read: filtered collections
    # =========================================================================

    def get_episodes(
        self,
        topic: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list:
        """Query episodes with optional filters.

        Args:
            topic: Filter to episodes containing this topic.
            since: Only include episodes occurring at/after this time.
            until: Only include episodes occurring at/before this time.
            active_only: If True, exclude inactive episodes.
            limit: Maximum number of results to return.

        Returns:
            A list of Episode objects ordered by occurred_at descending.
        """
        tbl = self._tbl(self.TABLE_EPISODES)
        conditions = []
        if active_only:
            conditions.append("is_active = true")
        if since:
            conditions.append(f"occurred_at >= '{since.isoformat()}'")
        if until:
            conditions.append(f"occurred_at <= '{until.isoformat()}'")
        where = " AND ".join(conditions) if conditions else None

        query = tbl.search().limit(limit * 3 if topic else limit)
        if where:
            query = query.where(where)
        rows = query.to_list()

        # Topic filter (array membership)
        if topic:
            rows = [r for r in rows if topic in (r.get("topics") or [])]

        # Sort by occurred_at descending and slice
        rows.sort(key=lambda r: r.get("occurred_at", ""), reverse=True)
        rows = rows[:limit]

        return [self._row_to_episode(r) for r in rows]

    def get_facts(
        self,
        topic: Optional[str] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list:
        """Query facts with optional filters.

        Args:
            topic: Optional topic filter.
            active_only: If True, exclude inactive/superseded facts.
            limit: Maximum number of facts to return.

        Returns:
            A list of Fact objects ordered by updated_at descending.
        """
        tbl = self._tbl(self.TABLE_FACTS)
        conditions = []
        if active_only:
            conditions.append("is_active = true")
        if topic:
            conditions.append(f"topic = '{topic}'")
        where = " AND ".join(conditions) if conditions else None

        query = tbl.search().limit(limit)
        if where:
            query = query.where(where)
        rows = query.to_list()
        rows.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
        return [self._row_to_fact(r) for r in rows]

    def get_summaries(
        self,
        topic: Optional[str] = None,
        level: Optional[int] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> list:
        """Query summaries with optional filters.

        Args:
            topic: Optional topic filter.
            level: Optional summary level filter.
            since: Only include summaries ending at/after this time.
            limit: Maximum number of summaries to return.

        Returns:
            A list of Summary objects ordered by time_end descending.
        """
        tbl = self._tbl(self.TABLE_SUMMARIES)
        conditions = ["is_active = true"]
        if topic:
            conditions.append(f"topic = '{topic}'")
        if level is not None:
            conditions.append(f"summary_level = {level}")
        if since:
            conditions.append(f"time_end >= '{since.isoformat()}'")
        where = " AND ".join(conditions)

        rows = tbl.search().where(where).limit(limit).to_list()
        rows.sort(key=lambda r: r.get("time_end", ""), reverse=True)
        return [self._row_to_summary(r) for r in rows]

    def get_unconsolidated_episodes(self, topic: Optional[str] = None) -> list:
        """Return episodes that have not been consolidated yet.

        Args:
            topic: Optional topic filter.

        Returns:
            A list of unconsolidated Episode objects.
        """
        tbl = self._tbl(self.TABLE_EPISODES)
        conditions = ["is_active = true", "consolidated = false"]
        if topic:
            conditions.append(f"array_has_all(topics, ARRAY['{topic}'])")
        where = " AND ".join(conditions)
        rows = tbl.search().where(where).limit(500).to_list()

        if topic:
            rows = [r for r in rows if topic in (r.get("topics") or [])]

        rows.sort(key=lambda r: r.get("occurred_at", ""), reverse=True)
        return [self._row_to_episode(r) for r in rows]

    def get_episodes_by_ids(self, episode_ids: list[str]) -> dict:
        """Retrieve multiple episodes by ID.

        Args:
            episode_ids: List of episode identifiers.

        Returns:
            A dict mapping episode_id -> Episode for found records.
        """
        if not episode_ids:
            return {}
        tbl = self._tbl(self.TABLE_EPISODES)
        id_list = ", ".join(f"'{i}'" for i in episode_ids)
        rows = tbl.search().where(f"id IN ({id_list})").limit(len(episode_ids)).to_list()
        return {r["id"]: self._row_to_episode(r) for r in rows}

    def get_facts_by_ids(self, fact_ids: list[str]) -> dict:
        """Retrieve multiple facts by ID.

        Args:
            fact_ids: List of fact identifiers.

        Returns:
            A dict mapping fact_id -> Fact for found records.
        """
        if not fact_ids:
            return {}
        tbl = self._tbl(self.TABLE_FACTS)
        id_list = ", ".join(f"'{i}'" for i in fact_ids)
        rows = tbl.search().where(f"id IN ({id_list})").limit(len(fact_ids)).to_list()
        return {r["id"]: self._row_to_fact(r) for r in rows}

    def get_summaries_by_ids(self, summary_ids: list[str]) -> dict:
        """Retrieve multiple summaries by ID.

        Args:
            summary_ids: List of summary identifiers.

        Returns:
            A dict mapping summary_id -> Summary for found records.
        """
        if not summary_ids:
            return {}
        tbl = self._tbl(self.TABLE_SUMMARIES)
        id_list = ", ".join(f"'{i}'" for i in summary_ids)
        rows = tbl.search().where(f"id IN ({id_list})").limit(len(summary_ids)).to_list()
        return {r["id"]: self._row_to_summary(r) for r in rows}

    def get_latest_summary(self, topic: str):
        """Return the most recent summary for a topic.

        Args:
            topic: Topic name.

        Returns:
            The latest Summary if one exists; otherwise None.
        """
        summaries = self.get_summaries(topic=topic, limit=1)
        return summaries[0] if summaries else None

    # =========================================================================
    # Topic helpers
    # =========================================================================

    def get_topics(self) -> list[dict]:
        """Return all topics with episode counts.

        Returns:
            A list of dicts with at minimum "name" and "episode_count" keys.
        """
        tbl = self._tbl(self.TABLE_EPISODES)
        rows = tbl.search().where("is_active = true").limit(10000).to_list()

        topic_counts: dict[str, int] = {}
        for row in rows:
            for t in row.get("topics") or []:
                topic_counts[t] = topic_counts.get(t, 0) + 1

        return [
            {"name": name, "episode_count": count}
            for name, count in sorted(topic_counts.items(), key=lambda x: -x[1])
        ]

    def get_topic_counts(self) -> dict[str, int]:
        """Return a mapping of topic -> active episode count.

        Returns:
            Dict of topic name to count.
        """
        return {t["name"]: t["episode_count"] for t in self.get_topics()}

    def get_topics_needing_consolidation(
        self, min_episodes: int = 5, max_age_days: int = 7
    ) -> list[str]:
        """Find topics that should be consolidated.

        Criteria: has at least `min_episodes` unconsolidated active episodes.

        Args:
            min_episodes: Minimum unconsolidated episode count to trigger consolidation.
            max_age_days: Not used in this implementation (kept for API compatibility).

        Returns:
            A list of topic names needing consolidation.
        """
        tbl = self._tbl(self.TABLE_EPISODES)
        rows = (
            tbl.search().where("is_active = true AND consolidated = false").limit(10000).to_list()
        )

        topic_counts: dict[str, int] = {}
        for row in rows:
            for t in row.get("topics") or []:
                topic_counts[t] = topic_counts.get(t, 0) + 1

        return [topic for topic, count in topic_counts.items() if count >= min_episodes]

    # =========================================================================
    # Vector search
    # =========================================================================

    def search(
        self,
        table: str,
        query_vector: np.ndarray,
        k: int = 10,
        where: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """Vector similarity search on a named table.

        Args:
            table: One of "episodes", "facts", "summaries".
            query_vector: Query embedding; will be L2-normalised internally.
            k: Maximum number of results to return.
            where: Optional SQL filter string (applied before ranking).

        Returns:
            A list of (record_id, similarity_score) tuples.
        """
        tbl = self._tbl(table)
        norm_vec = self._norm(query_vector)

        q = tbl.search(norm_vec).limit(k)
        if where:
            q = q.where(where)

        try:
            rows = q.to_list()
        except Exception as exc:
            logger.warning("Vector search error on %s: %s", table, exc)
            return []

        results = []
        for row in rows:
            score = float(row.get("_distance", 0.0))
            # LanceDB returns L2 distance; convert to cosine similarity for normalised vectors:
            # cosine_sim = 1 - (L2^2 / 2)  (for unit vectors)
            cosine_sim = max(-1.0, 1.0 - score / 2.0)
            results.append((row["id"], cosine_sim))

        return results

    # =========================================================================
    # Maintenance
    # =========================================================================

    def get_statistics(self) -> dict:
        """Return aggregate counts across all tables.

        Returns:
            A dict with total_episodes, unconsolidated_episodes, total_facts,
            total_summaries, and total_topics.
        """
        ep_tbl = self._tbl(self.TABLE_EPISODES)
        fact_tbl = self._tbl(self.TABLE_FACTS)
        sum_tbl = self._tbl(self.TABLE_SUMMARIES)

        total_episodes = len(ep_tbl.search().where("is_active = true").limit(100000).to_list())
        unconsolidated = len(
            ep_tbl.search()
            .where("is_active = true AND consolidated = false")
            .limit(100000)
            .to_list()
        )
        total_facts = len(fact_tbl.search().where("is_active = true").limit(100000).to_list())
        total_summaries = len(sum_tbl.search().where("is_active = true").limit(100000).to_list())
        topic_counts = self.get_topic_counts()

        return {
            "total_episodes": total_episodes,
            "unconsolidated_episodes": unconsolidated,
            "total_facts": total_facts,
            "total_summaries": total_summaries,
            "total_topics": len(topic_counts),
        }

    def optimize(self, table: str) -> None:
        """Create an HNSW index on the vector column for faster search.

        Args:
            table: Table name to index.
        """
        tbl = self._tbl(table)
        try:
            tbl.create_index(
                metric="cosine",
                index_type="IVF_HNSW_SQ",
            )
            logger.info("Created HNSW index on table '%s'", table)
        except Exception as exc:
            logger.warning("Could not create index on '%s': %s", table, exc)

    def close(self) -> None:
        """Close the LanceDB connection (no-op; LanceDB manages connections internally)."""
        pass
