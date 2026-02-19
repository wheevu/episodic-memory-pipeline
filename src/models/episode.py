"""
Episode model - represents a single episodic memory.

An episode is a timestamped event capturing what happened, when, and in what context.
This is the atomic unit of memory in the system.
"""

from datetime import datetime, timezone
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
