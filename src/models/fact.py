"""
Fact model - represents semantic memory (stable knowledge).

A fact is a distilled piece of knowledge extracted from one or more episodes.
Facts are more durable than episodes and represent what we "know" vs what "happened".
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class FactCategory(str, Enum):
    """Categories of factual knowledge."""

    PERSONAL = "personal"  # About the user
    PREFERENCE = "preference"  # User preferences
    RELATIONSHIP = "relationship"  # People/entities user knows
    KNOWLEDGE = "knowledge"  # Things user knows
    CONTEXT = "context"  # Situational context
    GOAL = "goal"  # Long-term goals


class Fact(BaseModel):
    """
    A semantic memory - a stable piece of knowledge.

    Design notes:
    - Facts are extracted from episodes via consolidation
    - They have temporal validity (valid_from, valid_until)
    - Supersession allows tracking how facts evolve over time
    - Confidence degrades if not reinforced by new episodes
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Content
    content: str  # The factual statement
    category: FactCategory = FactCategory.PERSONAL

    # Metadata
    topic: str  # Primary topic
    entities: list[str] = Field(default_factory=list)

    # Confidence and validity
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None  # NULL = currently valid

    # State
    is_active: bool = True
    superseded_by: Optional[str] = None  # ID of newer fact

    # Vector storage reference
    embedding_id: Optional[int] = None

    # Provenance (populated on retrieval)
    source_episode_ids: list[str] = Field(default_factory=list)

    class Config:
        """Pydantic model configuration."""

        use_enum_values = True

    @property
    def is_current(self) -> bool:
        """Return True if the fact is currently valid.

        Returns:
            True if within validity window, active, and not superseded.
        """
        now = datetime.now(timezone.utc)
        if self.valid_until and self.valid_until < now:
            return False
        if self.valid_from and self.valid_from > now:
            return False
        return self.is_active and self.superseded_by is None

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding.

        Returns:
            A compact string representation used for embedding generation.
        """
        parts = [self.content]
        parts.append(f"Category: {self.category}")
        parts.append(f"Topic: {self.topic}")
        return " | ".join(parts)

    def to_db_row(self) -> dict:
        """Convert the fact to a database row dictionary.

        Returns:
            A dictionary suitable for parameterized SQL insertion/update.
        """
        import json

        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "content": self.content,
            "category": self.category,
            "topic": self.topic,
            "entities": json.dumps(self.entities),
            "confidence": self.confidence,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "is_active": self.is_active,
            "superseded_by": self.superseded_by,
            "embedding_id": self.embedding_id,
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "Fact":
        """Create a `Fact` from a database row dictionary.

        Args:
            row: Database row mapping (dict or dict-like) containing fact fields.

        Returns:
            A populated `Fact` instance.
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
            created_at=parse_datetime(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=parse_datetime(row["updated_at"]) or datetime.now(timezone.utc),
            content=row["content"],
            category=FactCategory(row["category"]),
            topic=row["topic"],
            entities=json.loads(row["entities"])
            if isinstance(row["entities"], str)
            else row["entities"],
            confidence=row["confidence"],
            valid_from=parse_datetime(row.get("valid_from")),
            valid_until=parse_datetime(row.get("valid_until")),
            is_active=bool(row["is_active"]),
            superseded_by=row.get("superseded_by"),
            embedding_id=row.get("embedding_id"),
        )
