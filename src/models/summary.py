"""
Summary model - represents consolidated narrative memory.

A summary weaves together multiple episodes into a coherent narrative
about a topic over a time period.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Summary(BaseModel):
    """
    A consolidated narrative summary.

    Design notes:
    - Summaries are topic-scoped and time-bounded
    - They can form hierarchies (weekly → monthly → quarterly)
    - key_events captures the most important moments
    - Summaries link back to all source episodes for provenance
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Content
    content: str  # The summary narrative

    # Scope
    topic: str  # What topic this summarizes
    time_start: datetime  # Coverage start
    time_end: datetime  # Coverage end

    # Metadata
    episode_count: int = 0  # How many episodes contributed
    key_events: list[str] = Field(default_factory=list)  # Key event snippets

    # Hierarchy
    parent_summary_id: Optional[str] = None
    summary_level: int = 1  # 1=weekly, 2=monthly, 3=quarterly

    # State
    is_active: bool = True

    # Vector storage reference
    embedding_id: Optional[int] = None

    # Provenance (populated on retrieval)
    source_episode_ids: list[str] = Field(default_factory=list)

    @property
    def time_span_days(self) -> float:
        """Calculate the time span covered in days.

        Returns:
            The number of days covered by this summary.
        """
        return (self.time_end - self.time_start).total_seconds() / 86400

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding.

        Returns:
            A compact string representation used for embedding generation.
        """
        parts = [self.content]
        parts.append(f"Topic: {self.topic}")
        if self.key_events:
            parts.append(f"Key events: {'; '.join(self.key_events[:3])}")
        return " | ".join(parts)

    def to_db_row(self) -> dict:
        """Convert the summary to a database row dictionary.

        Returns:
            A dictionary suitable for parameterized SQL insertion/update.
        """
        import json

        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "content": self.content,
            "topic": self.topic,
            "time_start": self.time_start.isoformat(),
            "time_end": self.time_end.isoformat(),
            "episode_count": self.episode_count,
            "key_events": json.dumps(self.key_events),
            "parent_summary_id": self.parent_summary_id,
            "summary_level": self.summary_level,
            "is_active": self.is_active,
            "embedding_id": self.embedding_id,
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "Summary":
        """Create a `Summary` from a database row dictionary.

        Args:
            row: Database row mapping (dict or dict-like) containing summary fields.

        Returns:
            A populated `Summary` instance.
        """
        import json

        def parse_datetime(val: Optional[object]) -> Optional[datetime]:
            """Parse a database datetime field into a `datetime` instance.

            Args:
                val: Value from the database row; may be None, a datetime, or ISO string.

            Returns:
                A `datetime` if parsable; otherwise None.
            """
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        return cls(
            id=row["id"],
            created_at=parse_datetime(row["created_at"]) or datetime.utcnow(),
            updated_at=parse_datetime(row["updated_at"]) or datetime.utcnow(),
            content=row["content"],
            topic=row["topic"],
            time_start=parse_datetime(row["time_start"]),
            time_end=parse_datetime(row["time_end"]),
            episode_count=row["episode_count"],
            key_events=json.loads(row["key_events"])
            if isinstance(row["key_events"], str)
            else row["key_events"],
            parent_summary_id=row.get("parent_summary_id"),
            summary_level=row["summary_level"],
            is_active=bool(row["is_active"]),
            embedding_id=row.get("embedding_id"),
        )
