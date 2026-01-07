"""
Database schema and operations for SQLite storage.

Design rationale for SQLite:
- Local-first: No server required, single file
- ACID compliant: Safe concurrent access
- Perfect for single-user personal assistant
- Easy backup (just copy the file)
- Can migrate to Postgres later if needed (schema is compatible)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from contextlib import contextmanager

from memory.models import (
    Episode, Fact, Summary, 
    MemoryType, MemoryStatus
)
from memory.config import get_config


# =============================================================================
# SQL Schema Definitions
# =============================================================================

SCHEMA_SQL = """
-- Episodes: The foundational episodic memories
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,  -- episodic, fact, goal, preference, reflection
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Structured extraction (JSON)
    extracted_info TEXT DEFAULT '{}',  -- JSON object
    topics TEXT DEFAULT '[]',          -- JSON array
    entities TEXT DEFAULT '[]',        -- JSON array
    
    -- Metadata
    confidence REAL DEFAULT 1.0,
    source TEXT DEFAULT 'user_input',
    status TEXT DEFAULT 'active',  -- active, consolidated, superseded, archived
    
    -- Vector store reference
    embedding_id INTEGER,
    
    -- Session grouping
    session_id TEXT,
    
    -- Indexes for common queries
    CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_memory_type ON episodes(memory_type);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);

-- Facts: Semantic memories extracted from episodes
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    topic TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Provenance (JSON array of episode IDs)
    source_episode_ids TEXT DEFAULT '[]',
    
    -- Metadata
    confidence REAL DEFAULT 1.0,
    fact_type TEXT DEFAULT 'general',  -- general, preference, skill, relationship, etc.
    status TEXT DEFAULT 'active',
    
    -- Vector store reference
    embedding_id INTEGER,
    
    -- Conflict tracking
    superseded_by TEXT,  -- ID of fact that replaced this
    
    CHECK (confidence >= 0 AND confidence <= 1),
    FOREIGN KEY (superseded_by) REFERENCES facts(id)
);

CREATE INDEX IF NOT EXISTS idx_facts_topic ON facts(topic);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
CREATE INDEX IF NOT EXISTS idx_facts_fact_type ON facts(fact_type);

-- Summaries: Consolidated narrative memories
CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    topic TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Time range covered
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    
    -- Provenance (JSON arrays)
    source_episode_ids TEXT DEFAULT '[]',
    extracted_fact_ids TEXT DEFAULT '[]',
    
    -- Metadata
    episode_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    
    -- Vector store reference
    embedding_id INTEGER,
    
    -- Hierarchy
    parent_summary_id TEXT,
    
    FOREIGN KEY (parent_summary_id) REFERENCES summaries(id)
);

CREATE INDEX IF NOT EXISTS idx_summaries_topic ON summaries(topic);
CREATE INDEX IF NOT EXISTS idx_summaries_period ON summaries(period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_summaries_status ON summaries(status);

-- Episode-Fact provenance junction (for efficient reverse lookups)
CREATE TABLE IF NOT EXISTS episode_fact_links (
    episode_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (episode_id, fact_id),
    FOREIGN KEY (episode_id) REFERENCES episodes(id),
    FOREIGN KEY (fact_id) REFERENCES facts(id)
);

-- Episode-Summary provenance junction
CREATE TABLE IF NOT EXISTS episode_summary_links (
    episode_id TEXT NOT NULL,
    summary_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (episode_id, summary_id),
    FOREIGN KEY (episode_id) REFERENCES episodes(id),
    FOREIGN KEY (summary_id) REFERENCES summaries(id)
);

-- Topics table for controlled vocabulary
CREATE TABLE IF NOT EXISTS topics (
    name TEXT PRIMARY KEY,
    description TEXT,
    parent_topic TEXT,
    episode_count INTEGER DEFAULT 0,
    last_seen TIMESTAMP,
    FOREIGN KEY (parent_topic) REFERENCES topics(name)
);

-- Metadata table for system state
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# =============================================================================
# Database Connection Management
# =============================================================================

class DatabaseConnection:
    """Manages SQLite database connection and provides CRUD operations."""
    
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize the schema-backed database connection manager.

        Args:
            db_path: Optional path to the SQLite database file. If not provided,
                the configured default path is used.

        Returns:
            None.
        """
        self.db_path = db_path or get_config().database_path
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Create tables if they don't exist.

        Returns:
            None.
        """
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
    
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections.

        Returns:
            An iterator yielding a live `sqlite3.Connection`.
        """
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # =========================================================================
    # Episode Operations
    # =========================================================================
    
    def save_episode(self, episode: Episode) -> Episode:
        """Insert or update an episode.

        Args:
            episode: Episode to persist.

        Returns:
            The saved episode.
        """
        with self.connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO episodes (
                    id, content, memory_type, created_at,
                    extracted_info, topics, entities,
                    confidence, source, status, embedding_id, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.id,
                episode.content,
                episode.memory_type.value,
                episode.created_at.isoformat(),
                json.dumps(episode.extracted_info),
                json.dumps(episode.topics),
                json.dumps(episode.entities),
                episode.confidence,
                episode.source,
                episode.status.value,
                episode.embedding_id,
                episode.session_id,
            ))
            
            # Update topic counts
            for topic in episode.topics:
                conn.execute("""
                    INSERT INTO topics (name, episode_count, last_seen)
                    VALUES (?, 1, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        episode_count = episode_count + 1,
                        last_seen = excluded.last_seen
                """, (topic, episode.created_at.isoformat()))
            
            conn.commit()
        return episode
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID.

        Args:
            episode_id: Episode identifier.

        Returns:
            Matching `Episode` if found; otherwise None.
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
            if row:
                return self._row_to_episode(row)
        return None
    
    def get_episodes(
        self,
        memory_type: Optional[MemoryType] = None,
        status: Optional[MemoryStatus] = None,
        topic: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Episode]:
        """Query episodes with optional filters.

        Args:
            memory_type: Optional memory type filter.
            status: Optional status filter.
            topic: Optional topic filter (matches within stored JSON array string).
            since: Optional lower bound for `created_at`.
            until: Optional upper bound for `created_at`.
            limit: Maximum number of episodes to return.
            offset: Offset for pagination.

        Returns:
            List of matching episodes ordered by `created_at` descending.
        """
        query = "SELECT * FROM episodes WHERE 1=1"
        params = []
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if topic:
            query += " AND topics LIKE ?"
            params.append(f'%"{topic}"%')
        if since:
            query += " AND created_at >= ?"
            params.append(since.isoformat())
        if until:
            query += " AND created_at <= ?"
            params.append(until.isoformat())
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_episode(row) for row in rows]
    
    def get_episodes_by_ids(self, episode_ids: List[str]) -> List[Episode]:
        """Retrieve multiple episodes by their IDs.

        Args:
            episode_ids: Episode identifiers to fetch.

        Returns:
            List of matching episodes (order by `created_at` descending).
        """
        if not episode_ids:
            return []
        
        placeholders = ",".join("?" * len(episode_ids))
        query = f"SELECT * FROM episodes WHERE id IN ({placeholders}) ORDER BY created_at DESC"
        
        with self.connect() as conn:
            rows = conn.execute(query, episode_ids).fetchall()
            return [self._row_to_episode(row) for row in rows]
    
    def update_episode_status(self, episode_id: str, status: MemoryStatus) -> None:
        """Update the status of an episode.

        Args:
            episode_id: Episode identifier.
            status: New status to set.

        Returns:
            None.
        """
        with self.connect() as conn:
            conn.execute(
                "UPDATE episodes SET status = ? WHERE id = ?",
                (status.value, episode_id)
            )
            conn.commit()
    
    def update_episode_embedding_id(self, episode_id: str, embedding_id: int) -> None:
        """Update the embedding ID for an episode.

        Args:
            episode_id: Episode identifier.
            embedding_id: Vector store embedding identifier.

        Returns:
            None.
        """
        with self.connect() as conn:
            conn.execute(
                "UPDATE episodes SET embedding_id = ? WHERE id = ?",
                (embedding_id, episode_id)
            )
            conn.commit()
    
    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """Convert a database row to an `Episode` object.

        Args:
            row: SQLite row containing episode fields.

        Returns:
            Parsed `Episode`.
        """
        return Episode(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            extracted_info=json.loads(row["extracted_info"] or "{}"),
            topics=json.loads(row["topics"] or "[]"),
            entities=json.loads(row["entities"] or "[]"),
            confidence=row["confidence"],
            source=row["source"],
            status=MemoryStatus(row["status"]),
            embedding_id=row["embedding_id"],
            session_id=row["session_id"],
        )
    
    # =========================================================================
    # Fact Operations
    # =========================================================================
    
    def save_fact(self, fact: Fact) -> Fact:
        """Insert or update a fact.

        Args:
            fact: Fact to persist.

        Returns:
            The saved fact.
        """
        with self.connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO facts (
                    id, content, topic, created_at, updated_at,
                    source_episode_ids, confidence, fact_type, status,
                    embedding_id, superseded_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.id,
                fact.content,
                fact.topic,
                fact.created_at.isoformat(),
                fact.updated_at.isoformat(),
                json.dumps(fact.source_episode_ids),
                fact.confidence,
                fact.fact_type,
                fact.status.value,
                fact.embedding_id,
                fact.superseded_by,
            ))
            
            # Update junction table
            for episode_id in fact.source_episode_ids:
                conn.execute("""
                    INSERT OR IGNORE INTO episode_fact_links (episode_id, fact_id)
                    VALUES (?, ?)
                """, (episode_id, fact.id))
            
            conn.commit()
        return fact
    
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by ID.

        Args:
            fact_id: Fact identifier.

        Returns:
            Matching `Fact` if found; otherwise None.
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM facts WHERE id = ?", (fact_id,)
            ).fetchone()
            if row:
                return self._row_to_fact(row)
        return None
    
    def get_facts(
        self,
        topic: Optional[str] = None,
        fact_type: Optional[str] = None,
        status: Optional[MemoryStatus] = None,
        limit: int = 100,
    ) -> List[Fact]:
        """Query facts with optional filters.

        Args:
            topic: Optional topic filter.
            fact_type: Optional fact type filter.
            status: Optional status filter.
            limit: Maximum number of facts to return.

        Returns:
            List of matching facts ordered by `updated_at` descending.
        """
        query = "SELECT * FROM facts WHERE 1=1"
        params = []
        
        if topic:
            query += " AND topic = ?"
            params.append(topic)
        if fact_type:
            query += " AND fact_type = ?"
            params.append(fact_type)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_fact(row) for row in rows]
    
    def get_facts_by_ids(self, fact_ids: List[str]) -> List[Fact]:
        """Retrieve multiple facts by their IDs.

        Args:
            fact_ids: Fact identifiers to fetch.

        Returns:
            List of matching facts.
        """
        if not fact_ids:
            return []
        
        placeholders = ",".join("?" * len(fact_ids))
        query = f"SELECT * FROM facts WHERE id IN ({placeholders})"
        
        with self.connect() as conn:
            rows = conn.execute(query, fact_ids).fetchall()
            return [self._row_to_fact(row) for row in rows]
    
    def update_fact_embedding_id(self, fact_id: str, embedding_id: int) -> None:
        """Update the embedding ID for a fact.

        Args:
            fact_id: Fact identifier.
            embedding_id: Vector store embedding identifier.

        Returns:
            None.
        """
        with self.connect() as conn:
            conn.execute(
                "UPDATE facts SET embedding_id = ? WHERE id = ?",
                (embedding_id, fact_id)
            )
            conn.commit()
    
    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        """Convert a database row to a `Fact` object.

        Args:
            row: SQLite row containing fact fields.

        Returns:
            Parsed `Fact`.
        """
        return Fact(
            id=row["id"],
            content=row["content"],
            topic=row["topic"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source_episode_ids=json.loads(row["source_episode_ids"] or "[]"),
            confidence=row["confidence"],
            fact_type=row["fact_type"],
            status=MemoryStatus(row["status"]),
            embedding_id=row["embedding_id"],
            superseded_by=row["superseded_by"],
        )
    
    # =========================================================================
    # Summary Operations
    # =========================================================================
    
    def save_summary(self, summary: Summary) -> Summary:
        """Insert or update a summary.

        Args:
            summary: Summary to persist.

        Returns:
            The saved summary.
        """
        with self.connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO summaries (
                    id, content, topic, created_at, updated_at,
                    period_start, period_end,
                    source_episode_ids, extracted_fact_ids,
                    episode_count, status, embedding_id, parent_summary_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.id,
                summary.content,
                summary.topic,
                summary.created_at.isoformat(),
                summary.updated_at.isoformat(),
                summary.period_start.isoformat(),
                summary.period_end.isoformat(),
                json.dumps(summary.source_episode_ids),
                json.dumps(summary.extracted_fact_ids),
                summary.episode_count,
                summary.status.value,
                summary.embedding_id,
                summary.parent_summary_id,
            ))
            
            # Update junction table
            for episode_id in summary.source_episode_ids:
                conn.execute("""
                    INSERT OR IGNORE INTO episode_summary_links (episode_id, summary_id)
                    VALUES (?, ?)
                """, (episode_id, summary.id))
            
            conn.commit()
        return summary
    
    def get_summary(self, summary_id: str) -> Optional[Summary]:
        """Retrieve a summary by ID.

        Args:
            summary_id: Summary identifier.

        Returns:
            Matching `Summary` if found; otherwise None.
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM summaries WHERE id = ?", (summary_id,)
            ).fetchone()
            if row:
                return self._row_to_summary(row)
        return None
    
    def get_summaries(
        self,
        topic: Optional[str] = None,
        status: Optional[MemoryStatus] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[Summary]:
        """Query summaries with optional filters.

        Args:
            topic: Optional topic filter.
            status: Optional status filter.
            since: Optional lower bound for `period_end`.
            limit: Maximum number of summaries to return.

        Returns:
            List of matching summaries ordered by `period_end` descending.
        """
        query = "SELECT * FROM summaries WHERE 1=1"
        params = []
        
        if topic:
            query += " AND topic = ?"
            params.append(topic)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if since:
            query += " AND period_end >= ?"
            params.append(since.isoformat())
        
        query += " ORDER BY period_end DESC LIMIT ?"
        params.append(limit)
        
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_summary(row) for row in rows]
    
    def get_summaries_by_ids(self, summary_ids: List[str]) -> List[Summary]:
        """Retrieve multiple summaries by their IDs.

        Args:
            summary_ids: Summary identifiers to fetch.

        Returns:
            List of matching summaries.
        """
        if not summary_ids:
            return []
        
        placeholders = ",".join("?" * len(summary_ids))
        query = f"SELECT * FROM summaries WHERE id IN ({placeholders})"
        
        with self.connect() as conn:
            rows = conn.execute(query, summary_ids).fetchall()
            return [self._row_to_summary(row) for row in rows]
    
    def update_summary_embedding_id(self, summary_id: str, embedding_id: int) -> None:
        """Update the embedding ID for a summary.

        Args:
            summary_id: Summary identifier.
            embedding_id: Vector store embedding identifier.

        Returns:
            None.
        """
        with self.connect() as conn:
            conn.execute(
                "UPDATE summaries SET embedding_id = ? WHERE id = ?",
                (embedding_id, summary_id)
            )
            conn.commit()
    
    def _row_to_summary(self, row: sqlite3.Row) -> Summary:
        """Convert a database row to a `Summary` object.

        Args:
            row: SQLite row containing summary fields.

        Returns:
            Parsed `Summary`.
        """
        return Summary(
            id=row["id"],
            content=row["content"],
            topic=row["topic"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            period_start=datetime.fromisoformat(row["period_start"]),
            period_end=datetime.fromisoformat(row["period_end"]),
            source_episode_ids=json.loads(row["source_episode_ids"] or "[]"),
            extracted_fact_ids=json.loads(row["extracted_fact_ids"] or "[]"),
            episode_count=row["episode_count"],
            status=MemoryStatus(row["status"]),
            embedding_id=row["embedding_id"],
            parent_summary_id=row["parent_summary_id"],
        )
    
    # =========================================================================
    # Topic Operations
    # =========================================================================
    
    def get_all_topics(self) -> List[Dict[str, Any]]:
        """Get all topics with their counts.

        Returns:
            List of topic dictionaries (row mappings).
        """
        with self.connect() as conn:
            rows = conn.execute("""
                SELECT name, description, parent_topic, episode_count, last_seen
                FROM topics ORDER BY episode_count DESC
            """).fetchall()
            return [dict(row) for row in rows]
    
    def get_topics_for_episodes(self, episode_ids: List[str]) -> List[str]:
        """Get all unique topics from a set of episodes.

        Args:
            episode_ids: Episode identifiers to collect topics from.

        Returns:
            List of unique topic strings.
        """
        if not episode_ids:
            return []
        
        placeholders = ",".join("?" * len(episode_ids))
        query = f"SELECT topics FROM episodes WHERE id IN ({placeholders})"
        
        all_topics = set()
        with self.connect() as conn:
            rows = conn.execute(query, episode_ids).fetchall()
            for row in rows:
                topics = json.loads(row["topics"] or "[]")
                all_topics.update(topics)
        
        return list(all_topics)
    
    # =========================================================================
    # Metadata Operations
    # =========================================================================
    
    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            None.
        """
        with self.connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, datetime.utcnow().isoformat()))
            conn.commit()
    
    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value.

        Args:
            key: Metadata key.

        Returns:
            Metadata value if present; otherwise None.
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None
    
    # =========================================================================
    # Utility Operations
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary of counts for core tables.
        """
        with self.connect() as conn:
            episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            summary_count = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            topic_count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            
            return {
                "episodes": episode_count,
                "facts": fact_count,
                "summaries": summary_count,
                "topics": topic_count,
            }

