"""
Episode model - represents a single episodic memory.

An episode is a timestamped event capturing what happened, when, and in what context.
This is the atomic unit of memory in the system.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Classification of memory content type."""

    EPISODIC = "episodic"  # Event that happened
    FACT = "fact"  # Factual statement
    GOAL = "goal"  # User's goal or intention
    PREFERENCE = "preference"  # User's preference
    REFLECTION = "reflection"  # Meta-cognitive reflection


class Episode(BaseModel):
    """
    An episodic memory - a single event or piece of information with context.

    Design notes:
    - raw_input preserves original text for provenance
    - content is the processed/cleaned version for retrieval
    - occurred_at may differ from created_at (e.g., "yesterday I went...")
    - topics and entities enable filtering without vector search
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    occurred_at: datetime = Field(default_factory=datetime.utcnow)

    # Content
    raw_input: str  # Original input text
    content: str  # Processed content

    # Classification
    memory_type: MemoryType = MemoryType.EPISODIC

    # Metadata
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    # Context
    source: str = "chat"
    session_id: Optional[str] = None

    # State
    is_active: bool = True
    consolidated: bool = False

    # Vector storage reference
    embedding_id: Optional[int] = None

    class Config:
        """Pydantic model configuration."""

        use_enum_values = True

    def to_embedding_text(self) -> str:
        """
        Generate text representation for embedding.
        Includes context that aids semantic similarity.

        Returns:
            A compact string representation used for embedding generation.
        """
        parts = [self.content]

        if self.topics:
            parts.append(f"Topics: {', '.join(self.topics)}")

        if self.memory_type != MemoryType.EPISODIC:
            parts.append(f"Type: {self.memory_type}")

        return " | ".join(parts)

    def to_db_row(self) -> dict:
        """Convert the episode to a database row dictionary.

        Returns:
            A dictionary suitable for parameterized SQL insertion/update.
        """
        import json

        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "occurred_at": self.occurred_at.isoformat(),
            "raw_input": self.raw_input,
            "content": self.content,
            "memory_type": self.memory_type,
            "topics": json.dumps(self.topics),
            "entities": json.dumps(self.entities),
            "confidence": self.confidence,
            "importance": self.importance,
            "source": self.source,
            "session_id": self.session_id,
            "is_active": self.is_active,
            "consolidated": self.consolidated,
            "embedding_id": self.embedding_id,
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "Episode":
        """Create an `Episode` from a database row dictionary.

        Args:
            row: Database row mapping (dict or dict-like) containing episode fields.

        Returns:
            A populated `Episode` instance.
        """
        import json

        return cls(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"])
            if isinstance(row["created_at"], str)
            else row["created_at"],
            occurred_at=datetime.fromisoformat(row["occurred_at"])
            if isinstance(row["occurred_at"], str)
            else row["occurred_at"],
            raw_input=row["raw_input"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            topics=json.loads(row["topics"]) if isinstance(row["topics"], str) else row["topics"],
            entities=json.loads(row["entities"])
            if isinstance(row["entities"], str)
            else row["entities"],
            confidence=row["confidence"],
            importance=row["importance"],
            source=row["source"],
            session_id=row.get("session_id"),
            is_active=bool(row["is_active"]),
            consolidated=bool(row["consolidated"]),
            embedding_id=row.get("embedding_id"),
        )
