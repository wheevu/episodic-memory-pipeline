"""
Summary model - represents consolidated narrative memory.

A summary weaves together multiple episodes into a coherent narrative
about a topic over a time period.
"""

from datetime import datetime, timezone
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
